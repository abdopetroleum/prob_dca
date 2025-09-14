# Decline Curve Analysis (DCA) — Streamlit App

A lightweight Streamlit application for **Decline Curve Analysis** on conventional reservoirs using **Arps models** (Exponential, Hyperbolic, Harmonic). The app lets you smooth production data, fit models on a selected window, generate **forecasts**, and run **Monte Carlo** simulations (including an optional two‑segment override) to quantify uncertainty in **Total Production** and **NPV**.

---

## ✨ Features

- **Data input**
  - Upload an Excel file _or_ automatically fall back to the bundled sample: **`Volve production data.xlsx`**.
  - Column pickers for well name, date and production columns.
- **Smoothing**
  - Rolling average smoothing with outlier trimming (configurable window and std cutoff).
- **Model fitting (ARPS)**
  - Fits **Exponential (ex)**, **Hyperbolic (hp)** and **Harmonic (hr)** on a user‑selected fitting window.
  - Shows a parameters table (Qi, Di, b + optional stds) and model fit curves.
- **Forecasting**
  - Interactive **forecast window** starting from the end of the fitting window.
  - Forecasts all three models with consistent time handling.
- **Monte Carlo**
  - Runs scenarios by sampling Qi/Di/b from provided means/stds per model.
  - Draws daily price paths (normal) and discounts cashflows (daily) to compute **NPV**.
  - Summary table with **mean / P10 / P50 / P90** for both **TotalProd** and **NPV**.
  - Histograms & CDFs per model for quick diagnostics.
- **Two‑segment override (optional)**
  - Pick a **split date** and a **segment‑2 model** (ex/hp/hr).
  - Override its parameters (means/stds) after the split; **b is fixed** to 0 (ex) or 1 (hr) automatically.
  - Preview curves (all models pre‑split, chosen model post‑split) and run MC with the override.
- **Clean visuals**
  - Seaborn/Matplotlib theme with a modern color palette and subtle bands (no bright yellow).

---

## 📂 Project Structure

```
.
├── app.py                  # Streamlit app (UI + plots + MC + overrides)
├── dca_oop.py              # ARPS class: smoothing, preprocessing, fitting
├── utilities.py            # Helpers: safe_index, load_data, ARPS rate, forecasts, MC
├── notebook.ipynb          # (optional) experiments
└── Volve production data.xlsx   # Sample dataset used as default
```

> **Note**  
> The app expects `Volve production data.xlsx` to sit next to `app.py`. If you don’t upload a file, the app will read this sample by default.

---

## 🛠️ Installation

1) **Python** 3.9+ recommended.  
2) Install dependencies:

```bash
pip install streamlit pandas numpy matplotlib seaborn plotly openpyxl
```

> `openpyxl` is required to read Excel files.

---

## 🚀 Run the app

```bash
streamlit run app.py
```

Open the local URL printed in your terminal (usually http://localhost:8501).

---

## 🧭 How to use

1. **Load data**
   - Upload your Excel file (or let the app use the bundled sample).
   - Choose the columns: well name, production, date.
2. **Smooth & fit**
   - Set the **window size** and **std cutoff** for smoothing.
   - Select the **fitting window**. The app re‑fits ARPS models on this interval.
   - Inspect the **parameters table** and **fit plots**.
3. **Forecast**
   - Drag the **forecast window** slider to set the prediction horizon.
   - Review the combined **fit + forecast** figure.
4. **Monte Carlo**
   - Set **scenarios**, **discount rate**, **oil price mean/std**.
   - View the **summary table** (mean/P10/P50/P90) and histograms for the model you select.
5. **Two‑segment override (optional)**
   - Enable the override, choose **split date** and **segment‑2 model**.
   - Enter means/stds (for ex/hr, `b` is fixed to 0/1).
   - Preview and run MC with the override applied after the split.

---

## 📐 Methods (brief)

- **ARPS rate**:
  - Exponential: `q(t) = Qi * exp(-Di * t)`  
  - Hyperbolic:  `q(t) = Qi / (1 + b*Di*t)^(1/b)`  
  - Harmonic:    special case of hyperbolic with `b = 1`  
- **Forecasting logic**
  - `is_initial=True` restarts the curve at the forecast start with transformed `(Qi*, D*)` per model to maintain continuity.
  - `is_initial=False` continues from the provided `(Qi, Di, b)` values.
- **Monte Carlo**
  - Samples `Qi, Di, b` per model using normal draws from mean/std (stds default to 0 if missing).
  - For **ex** and **hr**, `b` is forced to `0` and `1` respectively.
  - Daily price path ~ `N(price_mean, price_std)`.  
  - **NPV** uses daily discount factors from the forecast start.
- **Two‑segment MC**
  - Pre‑split: all models use their sampled parameters.  
  - Post‑split: **only the chosen model** is used for all three series (via override), then metrics are computed.

---

## 🔧 Configuration notes

- Default sample: **`Volve production data.xlsx`**. If you move it, update the path logic in `app.py`.
- The app uses Streamlit caching for reading files to keep reloads fast.
- If you see `Invalid file path or buffer type: <class 'NoneType'>`, it means neither an upload nor the default file was found.

---

## 🧪 Tips

- Very long time series? Subsample points for the scatter to keep the UI snappy.
- Heavy MC runs? Lower `n_scenarios` or keep plots off-screen while running.
- Consider log‑scale (y) when comparing declines across wells (easy to add).

---

## 📄 License

MIT (or your preferred license).

---

## 🙌 Acknowledgements

- Sample dataset: **Volve** field (for demonstration).  
- Thanks to the Arps decline framework and the Python scientific stack.

