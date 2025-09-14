import pandas as pd
import streamlit as st
import plotly.express as px  # (not used below, but kept if you add Plotly charts later)
import numpy as np
import matplotlib.pyplot as plt
from dca_oop import ARPS
from datetime import timedelta

# ------------------------------- Utilities --------------------------------- #


def safe_index(options, value, fallback=0):
    """Return the index of `value` in `options`, or `fallback` if not present."""
    try:
        return list(options).index(value)
    except ValueError:
        return fallback


def load_data(filepath, well_names_col, well_name, prod_col_name, date_col_name):
    """
    Read an Excel file and return a 2-column DataFrame ['date', 'production']
    filtered to the specified well.
    """
    df = pd.read_excel(filepath, parse_dates=True)

    # Filter to the chosen well and keep only the desired columns
    df = df[df[well_names_col] == well_name][[date_col_name, prod_col_name]].copy()

    # Standardize column names for downstream code
    df.columns = ["date", "production"]
    return df

def arps_rate(t, Qi, Di, b):
    t = np.asarray(t, dtype=float)
    if np.isclose(b, 0.0):          # Exponential
        return Qi * np.exp(-Di * t)
    elif np.isclose(b, 1.0):        # Harmonic
        return Qi / (1.0 + Di * t)
    else:                           # Hyperbolic (general b)
        return Qi / np.power(1.0 + b * Di * t, 1.0 / b)

# ===== Common helpers =====

_MODEL_ORDER = np.array(["ex", "hp", "hr"])

def _params_df_to_arrays(parameters: pd.DataFrame):
    """Return arrays Qi, Di, b aligned to _MODEL_ORDER."""
    # assumes 'Model' has exactly {ex,hp,hr}
    params_sorted = parameters.set_index("Model").loc[_MODEL_ORDER]
    Qi = params_sorted["Qi"].to_numpy(float)
    Di = params_sorted["Di"].to_numpy(float)
    b  = params_sorted["b" ].to_numpy(float)
    return Qi, Di, b

def _arps_rate_vec(t: np.ndarray, Qi: np.ndarray, Di: np.ndarray, b: np.ndarray):
    """
    Vectorized ARPS.
    t: (T,)
    Qi, Di, b: (3,) aligned to ['ex','hp','hr']
    return q: (3, T)
    """
    t = t[None, :]                         # (1,T)
    Qi = Qi[:, None]                       # (3,1)
    Di = Di[:, None]
    b  = b[:,  None]

    # Exponential (b==0) and Harmonic (b==1) handled cleanly:
    q = np.empty((3, t.shape[1]), dtype=float)

    # exponential
    q[0] = (Qi[0] * np.exp(-Di[0] * t)).ravel()
    # hyperbolic general
    q[1] = (Qi[1] / np.power(1.0 + np.clip(b[1], 1e-30, None)*Di[1]*t, 1.0/np.clip(b[1], 1e-30, None))).ravel()
    # harmonic
    q[2] = (Qi[2] / (1.0 + Di[2] * t)).ravel()
    return q

def forecast_all_models(parameters, flow_rates, start_date, end_date, is_initial: bool = True):
    """
    Returns DataFrame: ['Date','ex','hp','hr'].
    Optimized: vectorized with arrays; no per-row loops.
    """
    # dates & times
    fdates = pd.to_datetime(flow_rates["Date"])
    t0 = fdates.min()

    start_date = pd.to_datetime(start_date)
    end_date   = pd.to_datetime(end_date)
    dates = pd.date_range(start=start_date, end=end_date, freq="D")
    t_rel = (dates - start_date).days.to_numpy(dtype=float)
    t_start = float((start_date - t0).days)

    Qi, Di, b = _params_df_to_arrays(parameters)

    if is_initial:
        # compute qi*, D* at t_start
        qi_star = np.array([
            Qi[0] * np.exp(-Di[0] * t_start),                                 # ex
            Qi[1] / np.power(1.0 + np.clip(b[1], 1e-30, None)*Di[1]*t_start, 1.0/np.clip(b[1], 1e-30, None)),  # hp
            Qi[2] / (1.0 + Di[2] * t_start)                                    # hr
        ], dtype=float)

        D_star = np.array([
            Di[0],                                      # ex
            Di[1] / (1.0 + np.clip(b[1], 1e-30, None)*Di[1]*t_start),  # hp
            Di[2] / (1.0 + Di[2] * t_start)             # hr
        ], dtype=float)

        q = _arps_rate_vec(t_rel, qi_star, D_star, b)
    else:
        q = _arps_rate_vec(t_rel, Qi, Di, b)

    out = pd.DataFrame({
        "Date": dates,
        "ex": q[0],
        "hp": q[1],
        "hr": q[2],
    })
    return out

def monte_carlo_simulation(parameters, flow_rates,
                           start_date, end_date,
                           n_scenarios=1000,
                           discount_rate=0.10,
                           price_mean=70, price_std=10,
                           is_initial=True,
                           clamp_b=(1e-6, 2.0)):
    """
    Same signature/output as before, but vectorized and ~O(10x) faster for large N.
    """
    rng = np.random.default_rng()

    # Precompute time axis and discount factors
    start_date = pd.to_datetime(start_date)
    end_date   = pd.to_datetime(end_date)
    dates = pd.date_range(start=start_date, end=end_date, freq="D")
    T = len(dates)
    t_rel_days = np.arange(T, dtype=float)
    discount_factors = 1.0 / np.power(1.0 + discount_rate, t_rel_days/365.0)

    Qi_mu, Di_mu, b_mu = _params_df_to_arrays(parameters)

    # STDs (fallback to zeros if missing)
    def _std(col):
        if col in parameters.columns:
            return parameters.set_index("Model").loc[_MODEL_ORDER, col].to_numpy(float)
        return np.zeros(3, dtype=float)

    Qi_sd = _std("Qi_std")
    Di_sd = _std("Di_std")
    b_sd  = _std("b_std")

    # Storage
    out_total = np.empty((3, n_scenarios), dtype=float)
    out_npv   = np.empty((3, n_scenarios), dtype=float)

    # Pre-grab t0 for restart math inside forecast
    fdates = pd.to_datetime(flow_rates["Date"])
    t0 = fdates.min()
    t_start = float((start_date - t0).days)

    for k in range(n_scenarios):
        price_path = rng.normal(price_mean, price_std, T)

        # sample parameters (vectorized across 3 models)
        Qi = np.maximum(rng.normal(Qi_mu, Qi_sd), 1e-8)
        Di = np.maximum(rng.normal(Di_mu, Di_sd), 1e-10)

        b  = rng.normal(b_mu, b_sd)
        # enforce canonical b for ex/hr
        b[0] = 0.0  # ex
        b[2] = 1.0  # hr
        # clamp hyperbolic only
        b[1] = np.clip(b[1], clamp_b[0], clamp_b[1])

        # rates
        t_rel = np.arange(T, dtype=float)
        if is_initial:
            qi_star = np.array([
                Qi[0] * np.exp(-Di[0]*t_start),
                Qi[1] / np.power(1.0 + np.clip(b[1], 1e-30, None)*Di[1]*t_start, 1.0/np.clip(b[1], 1e-30, None)),
                Qi[2] / (1.0 + Di[2]*t_start)
            ], dtype=float)

            D_star = np.array([
                Di[0],
                Di[1] / (1.0 + np.clip(b[1], 1e-30, None)*Di[1]*t_start),
                Di[2] / (1.0 + Di[2]*t_start)
            ], dtype=float)
            q = _arps_rate_vec(t_rel, qi_star, D_star, b)
        else:
            q = _arps_rate_vec(t_rel, Qi, Di, b)

        # metrics for all 3 models at once
        total_prod = np.trapz(q, dx=1.0, axis=1)                    # (3,)
        npv = (q * price_path[None, :] * discount_factors[None, :]).sum(axis=1)

        out_total[:, k] = total_prod
        out_npv[:, k]   = npv

    return pd.DataFrame({
        "ex_TotalProd": out_total[0], "ex_NPV": out_npv[0],
        "hp_TotalProd": out_total[1], "hp_NPV": out_npv[1],
        "hr_TotalProd": out_total[2], "hr_NPV": out_npv[2],
    })

def monte_carlo_two_segment_simulation(
    parameters, flow_rates,
    start_date, split_date, end_date,
    n_scenarios=1000,
    discount_rate=0.10,
    price_mean=70, price_std=10,
    clamp_b=(1e-6, 2.0),
    seg2_overrides=None  # {"ex"| "hp" | "hr": (Qi,Di,b)} OR {"ex"|...: (Qi_mu,Di_mu,b_mu, Qi_std,Di_std,b_std)}
):
    """
    Two-segment Monte Carlo using forecast_all_models twice per scenario.

    Segment 1: [start_date, split_date]   (is_initial=True)  -> use each model’s own curve.
    Segment 2: [split_date, end_date]     (is_initial=False) -> if seg2_overrides is set for ONE model,
                                                               use that model's post-split curve for ALL models.

    Returns DataFrame:
      ['ex_TotalProd','ex_NPV','hp_TotalProd','hp_NPV','hr_TotalProd','hr_NPV']
    """
    import numpy as np
    import pandas as pd

    # ---------- Basic validation ----------
    req_cols = {"Model", "Qi", "Di", "b"}
    if not req_cols.issubset(parameters.columns):
        raise ValueError(f"`parameters` must have columns {req_cols}. Got {list(parameters.columns)}.")

    models = ["ex", "hp", "hr"]
    if set(models) - set(parameters["Model"].unique()):
        raise ValueError("`parameters['Model']` must contain exactly 'ex','hp','hr' rows.")

    if "Date" not in flow_rates.columns:
        raise ValueError("`flow_rates` must have a 'Date' column.")

    start_date = pd.to_datetime(start_date)
    split_date = pd.to_datetime(split_date)
    end_date   = pd.to_datetime(end_date)
    if not (start_date <= split_date <= end_date):
        raise ValueError("Require start_date <= split_date <= end_date.")

    # time axis & discounting (whole cashflow window)
    dates_full = pd.date_range(start=start_date, end=end_date, freq="D")
    n_full = len(dates_full)
    t_rel_days = np.arange(n_full, dtype=float)
    discount_factors = 1.0 / np.power((1.0 + discount_rate), t_rel_days/365.0)

    # interpret override (must be 0 or 1 key if provided)
    override_key = None
    override_spec = None
    if seg2_overrides is not None:
        if not isinstance(seg2_overrides, dict) or len(seg2_overrides) != 1:
            raise ValueError("seg2_overrides must be a dict with EXACTLY one key among {'ex','hp','hr'}.")
        (override_key, override_spec), = seg2_overrides.items()
        if override_key not in models:
            raise ValueError(f"Invalid override key '{override_key}'. Must be one of {models}.")
        if not isinstance(override_spec, (tuple, list)) or len(override_spec) not in (3, 6):
            raise ValueError("Override for the chosen model must be a 3-tuple (Qi,Di,b) "
                             "or 6-tuple (Qi_mu,Di_mu,b_mu, Qi_std,Di_std,b_std).")

    # handy helper to read std columns safely
    def _get_std(df, model, col):
        try:
            v = float(df.loc[df["Model"] == model, col].values[0])
            return max(v, 0.0)
        except Exception:
            return 0.0

    rng = np.random.default_rng()

    results = {m: {"TotalProd": [], "NPV": []} for m in models}

    for _ in range(n_scenarios):
        # price path across full window
        price_path = rng.normal(price_mean, price_std, n_full)

        # --- draw base params for seg1 (and default seg2) ---
        params_draw = parameters.copy(deep=True)
        for m in models:
            # means
            Qi_mu = float(params_draw.loc[params_draw["Model"] == m, "Qi"].values[0])
            Di_mu = float(params_draw.loc[params_draw["Model"] == m, "Di"].values[0])
            b_mu  = float(params_draw.loc[params_draw["Model"] == m, "b" ].values[0])
            # stds
            Qi_std = _get_std(params_draw, m, "Qi_std")
            Di_std = _get_std(params_draw, m, "Di_std")
            b_std  = _get_std(params_draw, m, "b_std")
            # sample
            Qi = max(rng.normal(Qi_mu, Qi_std), 1e-8)
            Di = max(rng.normal(Di_mu, Di_std), 1e-10)
            if m == "ex":
                b = 0.0
            elif m == "hr":
                b = 1.0
            else:
                b = float(np.clip(rng.normal(b_mu, b_std), clamp_b[0], clamp_b[1]))
            params_draw.loc[params_draw["Model"] == m, ["Qi","Di","b"]] = [Qi, Di, b]

        # --- segment 1 forecast ---
        seg1 = forecast_all_models(
            params_draw, flow_rates, start_date=start_date, end_date=split_date, is_initial=True
        )  # columns: Date, ex, hp, hr

        # --- segment 2 parameters (apply override if any) ---
        params_seg2 = params_draw.copy(deep=True)
        if override_key is not None:
            m = override_key
            spec = override_spec
            if len(spec) == 3:
                Qi2, Di2, b2 = map(float, spec)
            else:
                Qi_mu2, Di_mu2, b_mu2, Qi_sd2, Di_sd2, b_sd2 = map(float, spec)
                Qi2 = max(rng.normal(Qi_mu2, Qi_sd2), 1e-8)
                Di2 = max(rng.normal(Di_mu2, Di_sd2), 1e-10)
                if m == "ex":
                    b2 = 0.0
                elif m == "hr":
                    b2 = 1.0
                else:
                    b2 = float(np.clip(rng.normal(b_mu2, b_sd2), clamp_b[0], clamp_b[1]))
            # enforce canonical b
            if m == "ex": b2 = 0.0
            if m == "hr": b2 = 1.0
            params_seg2.loc[params_seg2["Model"] == m, ["Qi","Di","b"]] = [Qi2, Di2, b2]

        # --- segment 2 forecast ---
        seg2 = forecast_all_models(
            params_seg2, flow_rates, start_date=split_date, end_date=end_date, is_initial=False
        )

        # ---------- stitch and compute metrics ----------
        # ensure "seg2 wins at split day": blank seg1 from split_date onward
        pre_idxed = seg1.set_index("Date")
        post_idxed = seg2.set_index("Date")

        # Reindex to full window
        pre_all  = pre_idxed.reindex(dates_full)
        post_all = post_idxed.reindex(dates_full)

        # drop seg1 values at/after split so seg2 provides split-day value
        pre_all.loc[dates_full >= split_date, ["ex","hp","hr"]] = np.nan

        if override_key is None:
            # normal: each model uses its own post segment
            merged = pre_all[["ex","hp","hr"]].fillna(post_all[["ex","hp","hr"]])
            merged = merged.astype(float)
            for m in models:
                q = merged[m].to_numpy(float)
                total_prod = float(np.trapz(q, dx=1.0))
                npv = float(np.sum(q * price_path * discount_factors))
                results[m]["TotalProd"].append(total_prod)
                results[m]["NPV"].append(npv)
        else:
            # override: ALL models use the chosen model's post segment
            ov_series = post_all[override_key].astype(float)
            for m in models:
                pre_series = pre_all[m].astype(float)
                q_series = pre_series.fillna(ov_series)  # post-split replaced by override model
                q = q_series.to_numpy(float)
                total_prod = float(np.trapz(q, dx=1.0))
                npv = float(np.sum(q * price_path * discount_factors))
                results[m]["TotalProd"].append(total_prod)
                results[m]["NPV"].append(npv)

    # pack results
    out = {}
    for m in models:
        out[f"{m}_TotalProd"] = results[m]["TotalProd"]
        out[f"{m}_NPV"]       = results[m]["NPV"]
    return pd.DataFrame(out)



def summarize_and_plot(mc_results, model, metric="TotalProd"):
    """
    Plot histogram + CDF for a given model and metric (TotalProd or NPV).
    Adds vertical lines for mean, P10, P90.
    """
    # extract the column
    col = f"{model}_{metric}"
    data = mc_results[col].values

    # compute stats
    mean_val = np.mean(data)
    p10 = np.percentile(data, 10)
    p90 = np.percentile(data, 90)

    # histogram
    fig, ax = plt.subplots(figsize=(16, 10), dpi=500)
    plt.hist(data, bins=30, alpha=0.7, color="skyblue", edgecolor="k")
    plt.axvline(mean_val, color="red", linestyle="--", label=f"Mean: {mean_val:,.0f}")
    plt.axvline(p10, color="green", linestyle="--", label=f"P10: {p10:,.0f}")
    plt.axvline(p90, color="blue", linestyle="--", label=f"P90: {p90:,.0f}")
    plt.title(f"Histogram of {col}", fontsize=14)
    plt.xlabel(metric)
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(alpha=0.3)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    return {"mean": mean_val, "P10": p10, "P90": p90}

def two_segment_preview_all_models(parameters, flow_rates,
                                   start_date, split_date, end_date,
                                   chosen_model,      # "ex" | "hp" | "hr"
                                   user_Qi_mean, user_Di_mean, user_b_mean):
    """
    Segment 1 (start_date → split_date):
        - Use fitted parameters
        - is_initial=True (restart at start_date)

    Segment 2 (split_date → end_date):
        - Override the chosen model with user means
        - is_initial=False (use new params but keep elapsed-time handling from forecast_all_models)
        - Return only the chosen model for the post-split part, as '<model>_after'
    """
    # --- normalize inputs
    start_date = pd.to_datetime(start_date)
    split_date = pd.to_datetime(split_date)
    end_date   = pd.to_datetime(end_date)

    # ---------- Segment 1: fitted params, is_initial = True ----------
    seg1 = forecast_all_models(parameters, flow_rates,
                               start_date=start_date,
                               end_date=split_date,
                               is_initial=True)   # all three models

    # ---------- Segment 2: user overrides for chosen model, is_initial = False ----------
    # copy parameters and replace the chosen model row with user means
    params2 = parameters.copy()

    # enforce canonical b for exp/harm
    if chosen_model == "ex":
        user_b_mean = 0.0
    elif chosen_model == "hr":
        user_b_mean = 1.0

    params2.loc[params2["Model"] == chosen_model, ["Qi", "Di", "b"]] = [
        float(user_Qi_mean), float(user_Di_mean), float(user_b_mean)
    ]

    seg2 = forecast_all_models(params2, flow_rates,
                               start_date=split_date,
                               end_date=end_date,
                               is_initial=False)  # only keep chosen model

    # ---------- Build combined preview ----------
    # pre-split: ex/hp/hr, NaNs for post rows
    pre = seg1[["Date", "ex", "hp", "hr"]].copy()
    pre_post_pad = pd.DataFrame({
        "Date": seg2["Date"],
        "ex":  np.nan, "hp": np.nan, "hr": np.nan
    })

    # post-split: chosen model as '<model>_after', NaNs for pre rows
    post_only = pd.DataFrame({
        "Date": seg2["Date"],
        f"{chosen_model}_after": seg2[chosen_model].values
    })
    pre_only_pad = pd.DataFrame({
        "Date": seg1["Date"],
        f"{chosen_model}_after": np.nan
    })

    # concatenate with aligned columns
    left  = pd.concat([pre, pre_post_pad], ignore_index=True)
    right = pd.concat([pre_only_pad, post_only], ignore_index=True)

    out = left.merge(right, on="Date", how="inner")

    return out