import streamlit as st
import pandas as pd
import plotly.express as px  # (not used below, but kept if you add Plotly charts later)
import numpy as np
import matplotlib.pyplot as plt
from dca_oop import ARPS
from datetime import timedelta

# ------------------------------- App Header -------------------------------- #

st.set_page_config(page_title="DCA")
st.title("Decline Curve Analysis")

st.markdown(
    """
An app to perform Decline Curve Analysis using ARPS models for conventional reservoirs.

- Upload your data  
- Specify whether it's field (multi‐well) or one well  
- Choose the parameters  
- See smoothed series, fit interval, and plots
"""
)

# ------------------------------- Utilities --------------------------------- #
def monte_carlo_two_segment_simulation(
    parameters, flow_rates,
    start_date, split_date, end_date,
    n_scenarios=1000,
    discount_rate=0.10,
    price_mean=70, price_std=10,
    clamp_b=(1e-6, 2.0),
    seg2_overrides=None  # e.g. {"hp": (Qi_mu, Di_mu, b_mu, Qi_std, Di_std, b_std)} or {"hp": (Qi,Di,b)}
):
    """
    Two-segment Monte Carlo using forecast_all_models(...) twice per scenario.

    Segment 1: [start_date, split_date]   with is_initial=True  -> use each model's own curve.
    Segment 2: [split_date, end_date]     with is_initial=False -> use ONLY the overridden model's curve for ALL models.

    seg2_overrides:
      - dict with EXACTLY one key in {'ex','hp','hr'}.
      - 3-tuple  (Qi, Di, b)   => fixed values in segment 2 (for that model).
      - 6-tuple  (Qi_mu, Di_mu, b_mu, Qi_std, Di_std, b_std) => sample per scenario in segment 2 (for that model).
      - For 'ex' and 'hr', b is forced to 0 or 1 regardless of what's passed.

    Returns a DataFrame with columns:
      ['ex_TotalProd','ex_NPV','hp_TotalProd','hp_NPV','hr_TotalProd','hr_NPV']
    """
    import numpy as np
    import pandas as pd

    # --- time axes ---
    start_date = pd.to_datetime(start_date)
    split_date = pd.to_datetime(split_date)
    end_date   = pd.to_datetime(end_date)

    dates_full = pd.date_range(start=start_date, end=end_date, freq="D")
    n_full = len(dates_full)

    # discount factors from forecast window start
    t_rel_days = np.arange(n_full, dtype=float)
    discount_factors = 1.0 / np.power((1.0 + discount_rate), t_rel_days/365.0)

    models = ["ex", "hp", "hr"]
    results = {m: {"TotalProd": [], "NPV": []} for m in models}

    # ---- validate & normalize overrides: exactly ONE model must be specified (or None) ----
    override_key = None
    override_spec = None
    if seg2_overrides is not None:
        if not isinstance(seg2_overrides, dict) or len(seg2_overrides) != 1:
            raise ValueError("seg2_overrides must be a dict with EXACTLY one key among {'ex','hp','hr'}.")
        (override_key, override_spec), = seg2_overrides.items()
        if override_key not in models:
            raise ValueError(f"Invalid override key '{override_key}'. Must be one of {models}.")
        if not isinstance(override_spec, (tuple, list)) or len(override_spec) not in (3, 6):
            raise ValueError(
                f"Override for '{override_key}' must be a 3-tuple (Qi,Di,b) or "
                f"6-tuple (Qi_mu,Di_mu,b_mu,Qi_std,Di_std,b_std)."
            )

    def _get_std(df, model, col):
        try:
            v = float(df.loc[df["Model"] == model, col].values[0])
            return max(v, 0.0)
        except Exception:
            return 0.0

    for _ in range(n_scenarios):
        # price path over the whole window
        price_path = np.random.normal(price_mean, price_std, n_full)

        # --- draw base params for this scenario (used in seg1; also default seg2) ---
        params_draw = parameters.copy(deep=True)
        for m in models:
            Qi_mu = float(params_draw.loc[params_draw["Model"] == m, "Qi"].values[0])
            Di_mu = float(params_draw.loc[params_draw["Model"] == m, "Di"].values[0])
            b_mu  = float(params_draw.loc[params_draw["Model"] == m, "b" ].values[0])

            Qi_std = _get_std(params_draw, m, "Qi_std")
            Di_std = _get_std(params_draw, m, "Di_std")
            b_std  = _get_std(params_draw, m, "b_std")

            Qi = max(np.random.normal(Qi_mu, Qi_std), 1e-8)
            Di = max(np.random.normal(Di_mu, Di_std), 1e-10)

            if m == "ex":
                b = 0.0
            elif m == "hr":
                b = 1.0
            else:
                b = np.random.normal(b_mu, b_std)
                b = float(np.clip(b, clamp_b[0], clamp_b[1]))

            params_draw.loc[params_draw["Model"] == m, ["Qi", "Di", "b"]] = [Qi, Di, b]

        # --- Segment 1 forecast (restart) ---
        seg1 = forecast_all_models(
            params_draw, flow_rates,
            start_date=start_date, end_date=split_date,
            is_initial=True
        )  # columns: Date, ex, hp, hr

        # --- Segment 2 params (apply override to the single chosen model, if any) ---
        params_seg2 = params_draw.copy(deep=True)
        if override_key is not None:
            m = override_key
            spec = override_spec
            if len(spec) == 3:
                Qi2, Di2, b2 = map(float, spec)
            else:  # len == 6  (mean,std sampling per scenario)
                Qi_mu2, Di_mu2, b_mu2, Qi_sd2, Di_sd2, b_sd2 = map(float, spec)
                Qi2 = max(np.random.normal(Qi_mu2, Qi_sd2), 1e-8)
                Di2 = max(np.random.normal(Di_mu2, Di_sd2), 1e-10)
                if m == "ex":
                    b2 = 0.0
                elif m == "hr":
                    b2 = 1.0
                else:
                    b2 = np.random.normal(b_mu2, b_sd2)
                    b2 = float(np.clip(b2, clamp_b[0], clamp_b[1]))
            # enforce canonical b for exp/harm regardless
            if m == "ex": b2 = 0.0
            if m == "hr": b2 = 1.0
            params_seg2.loc[params_seg2["Model"] == m, ["Qi","Di","b"]] = [Qi2, Di2, b2]

        # --- Segment 2 forecast (continue) ---
        seg2 = forecast_all_models(
            params_seg2, flow_rates,
            start_date=split_date, end_date=end_date,
            is_initial=False
        )  # columns: Date, ex, hp, hr

        # ===================== STITCHING FOR METRICS =====================
        # Base = pre-split (seg1) on full index; seg1 will be NaN after split
        pre = seg1.set_index("Date").reindex(dates_full)

        if override_key is None:
            # No special override → normal stitch (seg2 wins at split)
            merged = pd.concat([seg1, seg2], ignore_index=True).drop_duplicates("Date", keep="last")
            merged = merged.set_index("Date").reindex(dates_full).interpolate(limit_direction="both")
            # metrics per model
            for m in models:
                q = merged[m].to_numpy(float)
                total_prod = float(np.trapz(q, dx=1.0))
                revenues   = q * price_path
                npv        = float(np.sum(revenues * discount_factors))
                results[m]["TotalProd"].append(total_prod)
                results[m]["NPV"].append(npv)
        else:
            # Build the single override series from seg2 for the chosen model
            override_series = (
                seg2.set_index("Date")[override_key]
                    .reindex(dates_full)
                    .astype(float)
            )
            # For each model: use its seg1 curve before split; use override_series after split
            for m in models:
                series_pre = pre[m].astype(float)
                q_series = series_pre.fillna(override_series)  # <- key: post-split = override only
                q = q_series.to_numpy(float)
                total_prod = float(np.trapz(q, dx=1.0))
                revenues   = q * price_path
                npv        = float(np.sum(revenues * discount_factors))
                results[m]["TotalProd"].append(total_prod)
                results[m]["NPV"].append(npv)
        # ================================================================

    # pack results
    out = {}
    for m in models:
        out[f"{m}_TotalProd"] = results[m]["TotalProd"]
        out[f"{m}_NPV"]       = results[m]["NPV"]
    return pd.DataFrame(out)

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

def _safe_index(options, value, fallback=0):
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

def forecast_all_models(parameters, flow_rates, start_date, end_date, is_initial: bool = True):
    f = flow_rates.copy()
    f["Date"] = pd.to_datetime(f["Date"])
    t0 = f["Date"].min()

    start_date = pd.to_datetime(start_date)
    end_date   = pd.to_datetime(end_date)
    dates = pd.date_range(start=start_date, end=end_date, freq="D")

    # relative time within the prediction window
    t_rel = (dates - start_date).days.astype(float)

    # elapsed days from original origin to the new window start
    t_start = float((start_date - t0).days)

    out = pd.DataFrame({"Date": dates})

    for _, row in parameters.iterrows():
        model = row["Model"]
        Qi, Di, b = float(row["Qi"]), float(row["Di"]), float(row["b"])

        if is_initial:
            qi_star = arps_rate(t_start, Qi, Di, b)

            if np.isclose(b, 0.0):         # exponential
                D_star = Di
            elif np.isclose(b, 1.0):       # harmonic
                D_star = Di / (1.0 + Di * t_start)
            else:                           # hyperbolic
                D_star = Di / (1.0 + b * Di * t_start)

            out[model] = arps_rate(t_rel, qi_star, D_star, b)

        else:
            out[model] = arps_rate(t_rel, Qi, Di, b)

    return out
# ----- Monte Carlo using forecast_all_models -----
def monte_carlo_simulation(parameters, flow_rates,
                           start_date, end_date,
                           n_scenarios=1000,
                           discount_rate=0.10,
                           price_mean=70, price_std=10,
                           is_initial=True,
                           clamp_b=(1e-6, 2.0)):
    """
    Runs scenario draws by sampling Qi/Di/b per model, then calling forecast_all_models(...)
    to produce daily rates for ex/hp/hr. Computes TotalProd and NPV per scenario.

    Notes:
      - For 'ex' and 'hr', b is forced to 0 and 1 respectively (ignores b_std).
      - If *_std columns are missing, sampling falls back to the mean (std=0).
      - Discounting starts at the forecast window start_date.
    """
    # Time axis for the cashflow window
    start_date = pd.to_datetime(start_date)
    end_date   = pd.to_datetime(end_date)
    dates = pd.date_range(start=start_date, end=end_date, freq="D")
    n_steps = len(dates)
    # discount factors relative to the forecast start
    t_rel_days = np.arange(n_steps, dtype=float)
    discount_factors = 1.0 / np.power((1.0 + discount_rate), t_rel_days / 365.0)

    # Helper to safely read std columns (fallback to 0 if missing)
    def _get_std(df, model, colname):
        try:
            v = float(df.loc[df["Model"] == model, colname].values[0])
            return max(v, 0.0)
        except Exception:
            return 0.0

    models = ["ex", "hp", "hr"]
    # Storage
    results = {m: {"TotalProd": [], "NPV": []} for m in models}

    for _ in range(n_scenarios):
        # Price scenario (daily)
        price_path = np.random.normal(price_mean, price_std, n_steps)

        # Draw parameter set for this scenario
        params_draw = parameters.copy(deep=True)

        for m in models:
            # Means
            Qi_mu = float(params_draw.loc[params_draw["Model"] == m, "Qi"].values[0])
            Di_mu = float(params_draw.loc[params_draw["Model"] == m, "Di"].values[0])
            b_mu  = float(params_draw.loc[params_draw["Model"] == m, "b" ].values[0])

            # STDs (fallback to 0 if not present)
            Qi_std = _get_std(params_draw, m, "Qi_std")
            Di_std = _get_std(params_draw, m, "Di_std")
            b_std  = _get_std(params_draw, m, "b_std")

            # Sample
            Qi = np.random.normal(Qi_mu, Qi_std)
            Di = np.random.normal(Di_mu, Di_std)

            # Enforce valid positives
            Qi = max(Qi, 1e-8)
            Di = max(Di, 1e-10)

            # b handling
            if m == "ex":
                b = 0.0
            elif m == "hr":
                b = 1.0
            else:  # "hp"
                b = np.random.normal(b_mu, b_std)
                # clamp to reasonable range
                b = float(np.clip(b, clamp_b[0], clamp_b[1]))

            # Write back the draw
            params_draw.loc[params_draw["Model"] == m, ["Qi", "Di", "b"]] = [Qi, Di, b]

        # Forecast rates for this scenario (daily)
        scen_forecast = forecast_all_models(
            params_draw, flow_rates,
            start_date=start_date, end_date=end_date,
            is_initial=is_initial
        )
        # scen_forecast columns: ["Date","ex","hp","hr"]

        # Cashflow & metrics per model
        for m in models:
            q = scen_forecast[m].to_numpy(float)  # bbl/day
            total_prod = float(np.trapz(q, dx=1.0))  # bbl over window
            revenues = q * price_path               # $/day
            npv = float(np.sum(revenues * discount_factors))

            results[m]["TotalProd"].append(total_prod)
            results[m]["NPV"].append(npv)

    # Pack to DataFrame
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




# ---------------------------- Default Inputs -------------------------------- #

well_names_col_default = "NPD_WELL_BORE_NAME"
well_name_default      = "15/9-F-14"
prod_col_name_default  = "BORE_OIL_VOL"
date_col_name_default  = "DATEPRD"

# ------------------------------ Sidebar UI --------------------------------- #

st.sidebar.markdown("## Data Input `Production data`")
file = st.sidebar.file_uploader("Upload Excel file", type=["xlsx", "xls"])

# ------------------------------ Main Logic --------------------------------- #

if file:
    @st.cache_data
    def upload_file(f):
        """Cache Excel loading (first sheet by default)."""
        return pd.read_excel(f, sheet_name=0)

    df_raw = upload_file(file)

    # Column pickers (pre-filled with sensible defaults)
    cols = list(df_raw.columns)

    well_names_col = st.sidebar.selectbox(
        "Wells_name_column",
        cols,
        index=_safe_index(cols, well_names_col_default),
    )

    wells = df_raw[well_names_col].dropna().astype(str).unique().tolist()
    well_name = st.sidebar.selectbox(
        "Which well?",
        wells,
        index=_safe_index(wells, well_name_default),
    )

    prod_col_name = st.sidebar.selectbox(
        "Production column",
        cols,
        index=_safe_index(cols, prod_col_name_default),
    )

    date_col_name = st.sidebar.selectbox(
        "Date column",
        cols,
        index=_safe_index(cols, date_col_name_default),
    )

    # Load/shape the working DataFrame
    df = load_data(file, well_names_col, well_name, prod_col_name, date_col_name)
    st.table(df.head())

    # -------------------------- Smoothing Controls -------------------------- #

    st.markdown("## Smoothing the data using moving average")
    window_size = st.slider("Window size", min_value=10, max_value=200, value=100)
    stds = st.slider("Removing outliers (std)", min_value=1, max_value=10, value=3)

    # -------------------------- Fit Window Slider --------------------------- #

    dmin = pd.to_datetime(df["date"].min()).to_pydatetime()
    dmax = pd.to_datetime(df["date"].max()).to_pydatetime()

    fit_start, fit_end = st.slider(
        "Fitting window",
        min_value=dmin,
        max_value=dmax,
        value=(dmin, dmax),
        step=timedelta(days=1),
        format="YYYY-MM-DD",
    )

    fit_start = pd.to_datetime(fit_start)
    fit_end   = pd.to_datetime(fit_end)

    # --------------------------- Smoothing & Fit ---------------------------- #

    arps_all = ARPS(dataframe=df, production_col="production", date_column="date")
    df_smoothed_all = arps_all.smooth(window_size=window_size, stds=stds, trim=True).copy()
    df_smoothed_all["date"] = pd.to_datetime(df_smoothed_all["date"])

    # Build subset used for fitting
    mask = (df_smoothed_all["date"] >= fit_start) & (df_smoothed_all["date"] <= fit_end)
    df_fit = df_smoothed_all.loc[mask, ["date", "production"]].copy()

    # Safety check
    min_points = 5
    if len(df_fit) < min_points:
        st.error(
            f"Not enough data points in the fitting interval "
            f"({len(df_fit)} found, need at least {min_points})."
        )
        st.stop()

    # Re-fit and re-smooth ONLY the interval (to match class logic)
    arps_fit = ARPS(dataframe=df_fit, production_col="production", date_column="date")
    df_fit_smoothed = arps_fit.smooth(window_size=window_size, stds=stds, trim=True).copy()
    df_fit_smoothed["date"] = pd.to_datetime(df_fit_smoothed["date"])

    # ------------------------------- Plot ---------------------------------- #

    fig, ax = plt.subplots(figsize=(16, 10), dpi=500)

    # All data (raw + smoothed)
    ax.scatter(
        df_smoothed_all["date"], df_smoothed_all["production"],
        label="Original (all data)", alpha=0.45, color="gray", s=10
    )
    ax.plot(
        df_smoothed_all["date"], df_smoothed_all["production_rol_Av"],
        label="Smoothed (all data)", color="red", alpha=0.8, linewidth=2
    )

    # Fit-interval data (raw + smoothed)
    ax.scatter(
        df_fit["date"], df_fit["production"],
        label="Original (fit interval)", color="royalblue", s=18
    )
    ax.plot(
        df_fit_smoothed["date"], df_fit_smoothed["production_rol_Av"],
        label="Smoothed (fit interval)", color="seagreen", linewidth=2
    )

    # Shade fitting window
    ax.axvspan(fit_start, fit_end, color="gold", alpha=0.15, label="Fitting interval")

    # Formatting
    ax.set_title(f"Oil production well '{well_name}'", fontsize=18, loc="left", pad=12)
    ax.set_xlabel("Time")
    ax.set_ylabel("Oil production (bbl/day)")
    ax.grid(which="major", alpha=.5)
    ax.grid(which="minor", alpha=.1)
    ax.minorticks_on()
    ax.legend()

    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    arps_fit.prepocess_date_col(frequency="Daily")
    parameters, flow_rates, best_model = arps_fit.fit_all_models()
    st.table(parameters)

    fig, ax = plt.subplots(figsize=(16, 10), dpi=500)

    # convert to datetime for slicing
    fit_end_dt = pd.to_datetime(fit_end)

    # --- clip datasets up to fit_end ---
    df_all_clip = df_smoothed_all[df_smoothed_all["date"] <= fit_end_dt]
    flow_rates_clip = flow_rates[flow_rates["Date"] <= fit_end_dt]

    # full data (up to fit_end)
    plt.scatter(df_all_clip["date"], df_all_clip["production"],
                label="Original (all data)", alpha=0.45, color="gray")
    plt.plot(df_all_clip["date"], df_all_clip["production_rol_Av"],
             label="Smoothed (all data)", color="red", alpha=0.8)

    # fitted models (up to fit_end)
    plt.plot(flow_rates_clip["Date"], flow_rates_clip["Exponential"],
             label="Exponential fit", color="red")
    plt.plot(flow_rates_clip["Date"], flow_rates_clip["Hyperbolic"],
             label="Hyperbolic fit", color="blue")
    plt.plot(flow_rates_clip["Date"], flow_rates_clip["Harmonic"],
             label="Harmonic fit", color="green")

    # shade the fitting window
    plt.axvspan(pd.to_datetime(fit_start), pd.to_datetime(fit_end),
                color="gold", alpha=0.15, label="Fitting interval")

    # --- formatting ---
    plt.title("Fitting all Arps models and show the fitting",
              fontsize=18, color="#333", loc="left", pad=20)
    plt.xlabel("Date", fontsize=12, color="#333")
    plt.ylabel("Oil production (bbl/day)", fontsize=12, color="#333")

    plt.grid(which="major", color="#6666", linestyle="-", alpha=.5)
    plt.grid(which="minor", color="#9999", linestyle="-", alpha=.1)
    plt.minorticks_on()
    plt.legend()
    plt.xticks(rotation=45)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # --- Forecast window (range slider) ---
    fc_min = pd.to_datetime(fit_end).to_pydatetime()
    fc_default_end = (pd.to_datetime(fit_end) + pd.DateOffset(years=2)).to_pydatetime()
    fc_max = (pd.to_datetime(fit_end) + pd.DateOffset(years=10)).to_pydatetime()  # room to extend

    start_date, end_date = st.slider(
        "Forecast window",
        min_value=fc_min,
        max_value=fc_max,
        value=(fc_min, fc_default_end),
        step=timedelta(days=1),
        format="YYYY-MM-DD",
    )

    # keep as pandas Timestamps
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)


    forecast_df = forecast_all_models(parameters, flow_rates, start_date, end_date, is_initial=True)

    fig, ax = plt.subplots(figsize=(16, 10), dpi=500)

    # convert to datetime for slicing
    fit_end_dt = pd.to_datetime(fit_end)

    # --- clip datasets up to fit_end ---
    df_all_clip = df_smoothed_all[df_smoothed_all["date"] <= fit_end_dt]
    flow_rates_clip = flow_rates[flow_rates["Date"] <= fit_end_dt]

    # full data (up to fit_end)
    plt.scatter(df_all_clip["date"], df_all_clip["production"],
                label="Original (all data)", alpha=0.45, color="gray")
    plt.plot(df_all_clip["date"], df_all_clip["production_rol_Av"],
             label="Smoothed (all data)", color="red", alpha=0.8)

    # fitted + forecast models (same colors, one legend entry each)
    plt.plot(flow_rates_clip["Date"], flow_rates_clip["Exponential"],
             label="Exponential", color="red")
    plt.plot(forecast_df["Date"], forecast_df["ex"],
             color="red", linestyle="--", label="_nolegend_")

    plt.plot(flow_rates_clip["Date"], flow_rates_clip["Hyperbolic"],
             label="Hyperbolic", color="blue")
    plt.plot(forecast_df["Date"], forecast_df["hp"],
             color="blue", linestyle="--", label="_nolegend_")

    plt.plot(flow_rates_clip["Date"], flow_rates_clip["Harmonic"],
             label="Harmonic", color="green")
    plt.plot(forecast_df["Date"], forecast_df["hr"],
             color="green", linestyle="--", label="_nolegend_")

    # shade the fitting window
    plt.axvspan(pd.to_datetime(fit_start), pd.to_datetime(fit_end),
                color="gold", alpha=0.15, label="Fitting interval")

    # --- formatting ---
    plt.title("Arps fitting and forecasts", fontsize=18, color="#333", loc="left", pad=20)
    plt.xlabel("Date", fontsize=12, color="#333")
    plt.ylabel("Oil production (bbl/day)", fontsize=12, color="#333")

    plt.grid(which="major", color="#6666", linestyle="-", alpha=.5)
    plt.grid(which="minor", color="#9999", linestyle="-", alpha=.1)
    plt.minorticks_on()
    plt.legend()
    plt.xticks(rotation=45)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # --- Monte Carlo inputs (sidebar) ---
    st.sidebar.markdown("## Monte Carlo Parameters")

    n_scenarios = st.sidebar.number_input(
        "MC scenarios", min_value=100, max_value=100000, value=1000, step=100
    )

    discount_rate = st.sidebar.number_input(
        "Discount rate",
        min_value=0.00, max_value=1.00, value=0.10,
        step=0.01, format="%.2f"
    )

    oil_price_mean = st.sidebar.number_input(
        "Oil price mean ($/bbl)",
        min_value=0.00, max_value=1000.00, value=70.00,
        step=0.01, format="%.2f"
    )

    oil_price_std = st.sidebar.number_input(
        "Oil price std ($/bbl)",
        min_value=0.00, max_value=500.00, value=10.00,
        step=0.01, format="%.2f"
    )

    # ---------------- example usage ----------------
    # forecast 1000 scenarios from 2010-01-01 to 2015-12-31
    mc_results = monte_carlo_simulation(
        parameters=parameters,
        flow_rates=flow_rates,
        start_date=start_date,
        end_date=end_date,
        n_scenarios=n_scenarios,
        discount_rate=discount_rate,
        price_mean=oil_price_mean, price_std=oil_price_std
    )

    # --- Choose which model's Monte Carlo results to show ---
    model_map = {
        "Exponential (ex)": "ex",
        "Hyperbolic (hp)": "hp",
        "Harmonic (hr)": "hr",
    }
    model_label = st.sidebar.selectbox(
        "Monte Carlo: model to visualize",
        list(model_map.keys()),
        index=1,  # default: Hyperbolic
    )
    chosen_mc_model = model_map[model_label]

    # --- Show summaries/plots for the selected model ---
    st.markdown(f"### Monte Carlo results — {model_label}")
    summary_prod = summarize_and_plot(mc_results, chosen_mc_model, "TotalProd")
    summary_npv = summarize_and_plot(mc_results, chosen_mc_model, "NPV")



    # ========================== Segment-2 Override UI ===========================

    st.sidebar.markdown("## Segment-2 Override (optional)")
    use_seg2 = st.sidebar.checkbox("Enable two-segment override after split", value=False)

    if use_seg2:
        # Split date slider bounded by forecast window
        split_dt = st.sidebar.slider(
            "Split date",
            min_value=start_date.to_pydatetime(),
            max_value=end_date.to_pydatetime(),
            value=(start_date + (end_date - start_date) / 2).to_pydatetime(),
            step=timedelta(days=1),
            format="YYYY-MM-DD",
        )
        split_date = pd.to_datetime(split_dt)

        # Choose model for segment 2
        seg2_label = st.sidebar.selectbox(
            "Segment 2 decline model",
            ["Exponential (ex)", "Hyperbolic (hp)", "Harmonic (hr)"],
            index=1,
        )
        seg2_model = {"Exponential (ex)": "ex", "Hyperbolic (hp)": "hp", "Harmonic (hr)": "hr"}[seg2_label]

        # Segment-2 parameter means/stds (two-decimal or better)
        user_Qi_mean = st.sidebar.number_input(
            "Qi mean (bbl/day)", min_value=0.00, value=500.00, step=0.01, format="%.2f"
        )
        user_Qi_std = st.sidebar.number_input(
            "Qi std", min_value=0.00, value=50.00, step=0.01, format="%.2f"
        )

        user_Di_mean = st.sidebar.number_input(
            "Di mean (1/day)", min_value=0.000000, value=0.001000, step=0.000001, format="%.6f"
        )
        user_Di_std = st.sidebar.number_input(
            "Di std (1/day)", min_value=0.000000, value=0.000200, step=0.000001, format="%.6f"
        )

        # b handling: force to 0 for ex, 1 for hr; free only for hp
        if seg2_model == "hp":
            user_b_mean = st.sidebar.number_input(
                "b mean", min_value=0.00, max_value=2.00, value=0.65, step=0.01, format="%.2f"
            )
            user_b_std = st.sidebar.number_input(
                "b std", min_value=0.00, max_value=1.00, value=0.05, step=0.01, format="%.2f"
            )
        elif seg2_model == "ex":
            user_b_mean, user_b_std = 0.0, 0.0
            st.sidebar.info("For Exponential, b is fixed at 0.0.")
        else:  # "hr"
            user_b_mean, user_b_std = 1.0, 0.0
            st.sidebar.info("For Harmonic, b is fixed at 1.0.")

        # -------------------- Two-segment PREVIEW (visual check) --------------------
        preview = two_segment_preview_all_models(
            parameters, flow_rates,
            start_date=start_date,
            split_date=split_date,
            end_date=end_date,
            chosen_model=seg2_model,
            user_Qi_mean=user_Qi_mean,
            user_Di_mean=user_Di_mean,
            user_b_mean=user_b_mean,
        )

        fig, ax = plt.subplots(figsize=(16, 8), dpi=300)
        mask_ctx = (flow_rates["Date"] >= start_date) & (flow_rates["Date"] <= end_date)
        ax.plot(
            flow_rates.loc[mask_ctx, "Date"],
            flow_rates.loc[mask_ctx, "Original_Smoothed"],
            label="Original (smoothed)", color="black", linewidth=2, alpha=.7
        )
        colors = {"ex": "red", "hp": "blue", "hr": "green"}
        for m in ["ex", "hp", "hr"]:
            ax.plot(preview["Date"], preview[m], color=colors[m], linewidth=1.8, label=f"{m.upper()} (pre)")
        ax.plot(
            preview["Date"], preview[f"{seg2_model}_after"],
            color=colors[seg2_model], linestyle="--", linewidth=2.2,
            label=f"{seg2_model.upper()} (post, override)"
        )
        ax.axvline(split_date, color="#444", linestyle="--", alpha=.7, label="Split date")
        ax.axvspan(start_date, end_date, color="gold", alpha=.10, label="Forecast window")
        ax.set_title("Two-segment forecast: all models before split, chosen model after", loc="left")
        ax.set_xlabel("Date");
        ax.set_ylabel("Oil rate (bbl/day)")
        ax.grid(alpha=.3);
        ax.legend();
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True);
        plt.close(fig)

        # ---------------- Two-segment MONTE CARLO (override in segment 2) -----------
        seg2_overrides = {
            seg2_model: (
                float(user_Qi_mean), float(user_Di_mean), float(user_b_mean),
                float(user_Qi_std), float(user_Di_std), float(user_b_std)
            )
        }

        mc_two_seg = monte_carlo_two_segment_simulation(
            parameters=parameters,
            flow_rates=flow_rates,
            start_date=start_date,
            split_date=split_date,
            end_date=end_date,
            n_scenarios=n_scenarios,
            discount_rate=discount_rate,
            price_mean=oil_price_mean, price_std=oil_price_std,
            seg2_overrides=seg2_overrides
        )
        # --- summarize chosen model results ---
        summary_prod = summarize_and_plot(mc_two_seg, chosen_mc_model, "TotalProd")
        summary_npv = summarize_and_plot(mc_two_seg, chosen_mc_model, "NPV")
