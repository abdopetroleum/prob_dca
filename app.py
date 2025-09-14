import streamlit as st
import pandas as pd
import plotly.express as px  # (not used below, but kept if you add Plotly charts later)
import numpy as np
import matplotlib.pyplot as plt
from dca_oop import ARPS
from datetime import timedelta
from utilities import *
import seaborn as sns
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
from cycler import cycler

from pathlib import Path
DEFAULT_XLSX = Path(__file__).with_name("Volve production data.xlsx")  # same folder as this script

# ------------------------------- Theme & Colors ------------------------------- #
# Modern, accessible palette
HP   = "#2563EB"  # Hyperbolic  (blue-600)
EX   = "#EF4444"  # Exponential (red-500)
HR   = "#10B981"  # Harmonic    (emerald-500)
RAW  = "#94A3B8"  # raw points  (slate-400)
SMTH = "#0EA5E9"  # smoothed    (sky-500)

FIT_BAND = "#E0E7FF"  # subtle indigo-100 instead of yellow
FC_BAND  = "#E0F2FE"  # optional (forecast band) sky-100

sns.set_theme(style="whitegrid", context="talk", rc={
    "axes.facecolor":    "#FFFFFF",
    "figure.facecolor":  "#FFFFFF",
    "grid.color":        "#E5E7EB",
    "grid.linestyle":    "-",
    "grid.linewidth":    0.8,
    "axes.edgecolor":    "#111827",
    "axes.titlecolor":   "#111827",
    "text.color":        "#111827",
})
plt.rcParams["axes.prop_cycle"] = cycler(color=[HP, EX, HR, "#8B5CF6", "#F59E0B"])
plt.rcParams["lines.linewidth"] = 2.0
plt.rcParams["legend.frameon"]  = False

# ------------------------------- App Header -------------------------------- #
def _fmt_thousands(x, _): return f"{x:,.0f}"

def _decorate_date_axes(ax, title=None, ylabel="Oil rate (bbl/day)", logy=False):
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    ax.yaxis.set_major_formatter(FuncFormatter(_fmt_thousands))
    if logy: ax.set_yscale("log")
    if title: ax.set_title(title, loc="left", pad=8)
    ax.set_xlabel("Date"); ax.set_ylabel(ylabel)
    for spine in ("top","right"): ax.spines[spine].set_visible(False)
    return ax

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
    def upload_file_any(path_or_file):
        """Read first sheet from an uploaded file or a local path."""
        return pd.read_excel(path_or_file, sheet_name=0)


    # Decide which file to use
    if file is None:
        if DEFAULT_XLSX.exists():
            st.sidebar.info(f"Using default dataset: {DEFAULT_XLSX.name}")
            file_to_use = str(DEFAULT_XLSX)  # pandas accepts path string
        else:
            st.sidebar.error(
                "No file uploaded and default 'Volve production data.xlsx' not found. "
                "Please upload an Excel file."
            )
            st.stop()
    else:
        file_to_use = file  # Streamlit UploadedFile is fine for pandas

    # Load data (cached)
    df_raw = upload_file_any(file_to_use)

    # ---------------- Column pickers (pre-filled with sensible defaults) ----------------
    cols = list(df_raw.columns)

    well_names_col = st.sidebar.selectbox(
        "Wells_name_column",
        cols,
        index=safe_index(cols, well_names_col_default),
    )

    wells = df_raw[well_names_col].dropna().astype(str).unique().tolist()
    well_name = st.sidebar.selectbox(
        "Which well?",
        wells,
        index=safe_index(wells, well_name_default),
    )

    prod_col_name = st.sidebar.selectbox(
        "Production column",
        cols,
        index=safe_index(cols, prod_col_name_default),
    )

    date_col_name = st.sidebar.selectbox(
        "Date column",
        cols,
        index=safe_index(cols, date_col_name_default),
    )

    # Load/shape the working DataFrame
    df = load_data(file, well_names_col, well_name, prod_col_name, date_col_name)
    st.write(df)

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
    # fig 1
    fig, ax = plt.subplots(figsize=(16, 10), dpi=1080)

    # All data (raw + smoothed)
    ax.scatter(
        df_smoothed_all["date"], df_smoothed_all["production"],
        label="Original (all data)", alpha=0.45, color=RAW, s=10
    )
    ax.plot(
        df_smoothed_all["date"], df_smoothed_all["production_rol_Av"],
        label="Smoothed (all data)", color=SMTH, alpha=0.9, linewidth=2.4
    )

    # Fit-interval data (raw + smoothed)
    ax.scatter(
        df_fit["date"], df_fit["production"],
        label="Original (fit interval)", color=HP, s=18
    )
    ax.plot(
        df_fit_smoothed["date"], df_fit_smoothed["production_rol_Av"],
        label="Smoothed (fit interval)", color=HR, linewidth=2.2
    )

    # Shade fitting window
    ax.axvspan(fit_start, fit_end, color=FIT_BAND, alpha=0.6, label="Fitting interval")

    # Formatting
    ax.set_title(f"Oil production well '{well_name}'", fontsize=18, loc="left", pad=12)
    ax.set_xlabel("Time")
    ax.set_ylabel("Oil production (bbl/day)")
    ax.grid(which="major", alpha=.5)
    ax.grid(which="minor", alpha=.1)
    ax.minorticks_on()
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0.)

    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    arps_fit.prepocess_date_col(frequency="Daily")
    parameters, flow_rates, best_model = arps_fit.fit_all_models()
    st.table(parameters)

    # ------------------------------- Plot ---------------------------------- #
    # fig 2
    fig, ax = plt.subplots(figsize=(16, 10), dpi=500)

    # convert to datetime for slicing
    fit_end_dt = pd.to_datetime(fit_end)

    # --- clip datasets up to fit_end ---
    df_all_clip = df_smoothed_all[df_smoothed_all["date"] <= fit_end_dt]
    flow_rates_clip = flow_rates[flow_rates["Date"] <= fit_end_dt]

    # full data (up to fit_end)
    plt.scatter(df_all_clip["date"], df_all_clip["production"],
                label="Original (all data)", alpha=0.45, color=RAW)
    plt.plot(df_all_clip["date"], df_all_clip["production_rol_Av"],
             label="Smoothed (all data)", color=SMTH, alpha=0.9)

    # fitted models (up to fit_end)
    plt.plot(flow_rates_clip["Date"], flow_rates_clip["Exponential"],
             label="Exponential fit", color=EX)
    plt.plot(flow_rates_clip["Date"], flow_rates_clip["Hyperbolic"],
             label="Hyperbolic fit", color=HP)
    plt.plot(flow_rates_clip["Date"], flow_rates_clip["Harmonic"],
             label="Harmonic fit", color=HR)

    # shade the fitting window
    plt.axvspan(pd.to_datetime(fit_start), pd.to_datetime(fit_end),
                color=FIT_BAND, alpha=0.6, label="Fitting interval")

    # --- formatting ---
    plt.title("Fitting all Arps models and show the fitting",
              fontsize=18, color="#333", loc="left", pad=20)
    plt.xlabel("Date", fontsize=12, color="#333")
    plt.ylabel("Oil production (bbl/day)", fontsize=12, color="#333")

    plt.grid(which="major", color="#6666", linestyle="-", alpha=.5)
    plt.grid(which="minor", color="#9999", linestyle="-", alpha=.1)
    plt.minorticks_on()
    plt.legend(loc="upper left", bbox_to_anchor=(1.02, 1))
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

    # ------------------------------- Plot ---------------------------------- #
    # fig 3
    fig, ax = plt.subplots(figsize=(16, 10), dpi=500)

    # convert to datetime for slicing
    fit_end_dt = pd.to_datetime(fit_end)

    # --- clip datasets up to fit_end ---
    df_all_clip = df_smoothed_all[df_smoothed_all["date"] <= fit_end_dt]
    flow_rates_clip = flow_rates[flow_rates["Date"] <= fit_end_dt]

    # full data (up to fit_end)
    plt.scatter(df_all_clip["date"], df_all_clip["production"],
                label="Original (all data)", alpha=0.45, color=RAW)
    plt.plot(df_all_clip["date"], df_all_clip["production_rol_Av"],
             label="Smoothed (all data)", color=SMTH, alpha=0.9)

    # fitted + forecast models (same colors, one legend entry each)
    plt.plot(flow_rates_clip["Date"], flow_rates_clip["Exponential"],
             label="Exponential", color=EX)
    plt.plot(forecast_df["Date"], forecast_df["ex"],
             color=EX, linestyle="--", label="_nolegend_")

    plt.plot(flow_rates_clip["Date"], flow_rates_clip["Hyperbolic"],
             label="Hyperbolic", color=HP)
    plt.plot(forecast_df["Date"], forecast_df["hp"],
             color=HP, linestyle="--", label="_nolegend_")

    plt.plot(flow_rates_clip["Date"], flow_rates_clip["Harmonic"],
             label="Harmonic", color=HR)
    plt.plot(forecast_df["Date"], forecast_df["hr"],
             color=HR, linestyle="--", label="_nolegend_")

    # bands
    plt.axvspan(pd.to_datetime(fit_start), pd.to_datetime(fit_end),
                color=FIT_BAND, alpha=0.6, label="Fitting interval")
    # Optional: show forecast window subtly
    # plt.axvspan(pd.to_datetime(start_date), pd.to_datetime(end_date),
    #             color=FC_BAND, alpha=0.45, label="Forecast interval")

    # --- formatting ---
    plt.title("Arps fitting and forecasts", fontsize=18, color="#333", loc="left", pad=20)
    plt.xlabel("Date", fontsize=12, color="#333")
    plt.ylabel("Oil production (bbl/day)", fontsize=12, color="#333")

    plt.grid(which="major", color="#6666", linestyle="-", alpha=.5)
    plt.grid(which="minor", color="#9999", linestyle="-", alpha=.1)
    plt.minorticks_on()
    plt.legend(loc="upper left", bbox_to_anchor=(1.02, 1))
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

    # ---- Monte Carlo summary table (all three models) ----
    def _mc_summary_table(df: pd.DataFrame) -> pd.DataFrame:
        rows = []
        name_map = {"ex": "Exponential", "hp": "Hyperbolic", "hr": "Harmonic"}
        for m in ["ex", "hp", "hr"]:
            tp = df[f"{m}_TotalProd"].to_numpy(float)
            nv = df[f"{m}_NPV"].to_numpy(float)
            rows.append({
                "Model": name_map[m],
                "TotalProd mean (bbl)": np.mean(tp),
                "TotalProd P10 (bbl)": np.percentile(tp, 10),
                "TotalProd P50 (bbl)": np.percentile(tp, 50),
                "TotalProd P90 (bbl)": np.percentile(tp, 90),
                "NPV mean ($)": np.mean(nv),
                "NPV P10 ($)": np.percentile(nv, 10),
                "NPV P50 ($)": np.percentile(nv, 50),
                "NPV P90 ($)": np.percentile(nv, 90),
            })
        out = pd.DataFrame(rows)
        return out

    st.markdown("### Monte Carlo summary (all models)")
    mc_table = _mc_summary_table(mc_results)

    # Nicely formatted table
    st.dataframe(
        mc_table.style.format({
            "TotalProd mean (bbl)": "{:,.0f}",
            "TotalProd P10 (bbl)": "{:,.0f}",
            "TotalProd P50 (bbl)": "{:,.0f}",
            "TotalProd P90 (bbl)": "{:,.0f}",
            "NPV mean ($)": "${:,.0f}",
            "NPV P10 ($)": "${:,.0f}",
            "NPV P50 ($)": "${:,.0f}",
            "NPV P90 ($)": "${:,.0f}",
        }),
        use_container_width=True
    )

    # --- Show summaries/plots for the selected model ---
    # fig 4,5
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
        # ------------------------------- Plot ---------------------------------- #
        # fig 6
        fig, ax = plt.subplots(figsize=(16, 8), dpi=300)
        mask_ctx = (flow_rates["Date"] >= start_date) & (flow_rates["Date"] <= end_date)
        ax.plot(
            flow_rates.loc[mask_ctx, "Date"],
            flow_rates.loc[mask_ctx, "Original_Smoothed"],
            label="Original (smoothed)", color="black", linewidth=2, alpha=.7
        )
        colors = {"ex": EX, "hp": HP, "hr": HR}
        for m in ["ex", "hp", "hr"]:
            ax.plot(preview["Date"], preview[m], color=colors[m], linewidth=1.8, label=f"{m.upper()} (pre)")
        ax.plot(
            preview["Date"], preview[f"{seg2_model}_after"],
            color=colors[seg2_model], linestyle="--", linewidth=2.2,
            label=f"{seg2_model.upper()} (post, override)"
        )
        ax.axvline(split_date, color="#444", linestyle="--", alpha=.7, label="Split date")
        ax.axvspan(start_date, end_date, color=FC_BAND, alpha=.10, label="Forecast window")
        ax.set_title("Two-segment forecast: all models before split, chosen model after", loc="left")
        ax.set_xlabel("Date")
        ax.set_ylabel("Oil rate (bbl/day)")
        ax.grid(alpha=.3)
        ax.legend()
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
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
        # fig 7,8
        summary_prod = summarize_and_plot(mc_two_seg, chosen_mc_model, "TotalProd")
        summary_npv = summarize_and_plot(mc_two_seg, chosen_mc_model, "NPV")
