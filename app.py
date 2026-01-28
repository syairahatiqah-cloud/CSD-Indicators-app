# app.py
# ============================================================
# Streamlit App: CSD Rolling Indicators (Variance + PSD Low-Freq)
# (Computes AR1 too, included in CSV output, but not plotted)
#
# Matches your R logic:
# - datetime parsing: tries fixed formats first (day-first), then fallback
# - rolling window computation:
#     * requires >= max(4, valid_frac*window) finite points
#     * variance = sample variance (ddof=1)
#     * PSD low freq = mean of first k bins excluding DC
#     * dynamic low_k = min(8, max(2, floor(win/10)))
#
# Outputs:
# - Preview table
# - Interactive Plotly (Variance + PSD) + download HTML
# - Static Matplotlib (Variance + PSD) + download PNG
# - Download combined CSV
# - Download per-window ZIP (CSVs)
# ============================================================

import io
import re
import zipfile
import numpy as np
import pandas as pd
import streamlit as st

import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# -----------------------------
# Robust CSV loader (fix UnicodeDecodeError)
# -----------------------------
@st.cache_data(show_spinner=False)
def load_csv_bytes(file_bytes: bytes) -> pd.DataFrame:
    """
    Robust CSV loader:
    - tries multiple encodings (utf-8, utf-8-sig, utf-16, cp1252, latin1)
    - tries common separators (comma, semicolon, tab)
    """
    encodings = ["utf-8", "utf-8-sig", "utf-16", "cp1252", "latin1"]
    seps = [",", ";", "\t"]

    last_err = None
    for enc in encodings:
        for sep in seps:
            try:
                return pd.read_csv(
                    io.BytesIO(file_bytes),
                    dtype=str,
                    keep_default_na=False,
                    encoding=enc,
                    sep=sep,
                    engine="python",
                )
            except Exception as e:
                last_err = e

    # final fallback: separator auto-detect, forgiving encoding
    try:
        return pd.read_csv(
            io.BytesIO(file_bytes),
            dtype=str,
            keep_default_na=False,
            encoding="latin1",
            sep=None,
            engine="python",
        )
    except Exception as e:
        raise RuntimeError(f"Failed to read CSV. Last error: {last_err}") from e


# -----------------------------
# Datetime parsing (R-like order)
# -----------------------------
def parse_dt_series(s: pd.Series, tz: str = "Asia/Kuala_Lumpur") -> pd.Series:
    """
    Mimics your R parse_dt() order:
      %d/%m/%Y ... first, %m/%d/%Y ... last
    Adds date-only support too.
    """
    s = s.astype(str).str.strip()

    fmts = [
        "%d/%m/%Y %H:%M", "%d/%m/%Y %H",
        "%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S",
        "%m/%d/%Y %H:%M",
        "%d/%m/%Y", "%Y-%m-%d", "%m/%d/%Y"
    ]

    dt = pd.Series(pd.NaT, index=s.index)
    remaining = dt.isna()

    for f in fmts:
        if not remaining.any():
            break
        parsed = pd.to_datetime(s[remaining], format=f, errors="coerce")
        dt.loc[remaining] = parsed
        remaining = dt.isna()

    # last fallback: inference but force dayfirst=True (important!)
    if remaining.any():
        dt.loc[remaining] = pd.to_datetime(s[remaining], errors="coerce", dayfirst=True)

    # timezone handling (optional; safe if fails)
    if tz:
        try:
            if getattr(dt.dt, "tz", None) is None:
                dt = dt.dt.tz_localize(tz, ambiguous="NaT", nonexistent="NaT")
            else:
                dt = dt.dt.tz_convert(tz)
        except Exception:
            pass

    return dt


# -----------------------------
# CSD helpers
# -----------------------------
def calc_ar1(w: np.ndarray) -> float:
    w = w[np.isfinite(w)]
    if w.size < 2:
        return np.nan
    if np.nanstd(w) == 0:
        return np.nan
    return float(np.corrcoef(w[:-1], w[1:])[0, 1])


def calc_lowfreq_psd(w: np.ndarray, low_k: int = 8) -> float:
    """
    PSD low-frequency mean using rFFT power, exclude DC.
    (Not identical to R spectrum() method, but consistent and stable.)
    """
    w = w[np.isfinite(w)]
    if w.size < 4:
        return np.nan

    w = w - np.nanmean(w)
    spec = np.abs(np.fft.rfft(w)) ** 2
    if spec.size <= 1:
        return np.nan

    k = min(low_k, spec.size - 1)  # exclude DC at index 0
    if k < 1:
        return np.nan

    return float(np.nanmean(spec[1:k + 1]))


def csd_roll(x: np.ndarray, dt: pd.Series, window_hours: int, valid_frac: float = 0.8) -> pd.DataFrame:
    win = int(window_hours)
    low_k = min(8, max(2, win // 10))  # dynamic low_k like your R code

    n = len(x)
    out_len = n - win + 1
    if out_len < 1:
        return pd.DataFrame()

    var_roll = np.full(out_len, np.nan, dtype=float)
    ar1_roll = np.full(out_len, np.nan, dtype=float)
    psd_low  = np.full(out_len, np.nan, dtype=float)

    min_good = max(4, int(np.floor(valid_frac * win)))

    for i in range(out_len):
        w = x[i:i + win]
        good = np.isfinite(w).sum()
        if good < min_good:
            continue

        w2 = w[np.isfinite(w)]
        # Match R var() sample variance
        var_roll[i] = float(np.var(w2, ddof=1)) if w2.size > 1 else np.nan
        ar1_roll[i] = calc_ar1(w2)
        psd_low[i]  = calc_lowfreq_psd(w2, low_k=low_k)

    return pd.DataFrame({
        "datetime_start": dt.iloc[:out_len].values,
        "datetime_end":   dt.iloc[win - 1: win - 1 + out_len].values,
        "window_hours":   win,
        "low_k_used":     low_k,
        "Variance":       var_roll,
        "AR1":            ar1_roll,
        "PSD_LowFreq":    psd_low
    })


def safe_name(col: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", str(col)).strip("_") or "value"


def make_plotly_line(df: pd.DataFrame, y: str, title: str, yaxis_title: str) -> go.Figure:
    fig = px.line(df, x="datetime_start", y=y, title=title)
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title=yaxis_title,
        xaxis=dict(
            tickangle=-45,
            rangeslider=dict(visible=True)
        ),
        hovermode="x unified",  # nicer hover like your screenshot
        margin=dict(l=10, r=10, t=50, b=10)
    )
    return fig


def make_matplotlib_line(df: pd.DataFrame, y: str, title: str, ylabel: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df["datetime_start"], df[y])
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel(ylabel)

    locator = mdates.AutoDateLocator(minticks=6, maxticks=12)
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    return fig


def fig_to_png_bytes(fig: plt.Figure, dpi: int = 300) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="CSD Rolling Indicators", layout="wide")
st.title("CSD Rolling Indicators (Variance + PSD Low-Frequency)")

with st.sidebar:
    st.header("Inputs")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    tz = st.selectbox("Timezone for datetime parsing", ["Asia/Kuala_Lumpur", "UTC"], index=0)

    window_choices = [6, 12, 24, 36, 48, 72, 96, 120, 144, 168]
    windows = st.multiselect("Rolling windows (hours)", options=window_choices, default=window_choices)

    valid_frac = st.slider("Min valid data per window", min_value=0.50, max_value=1.00, value=0.80, step=0.05)

    run = st.button("Run CSD", type="primary")


if uploaded is None:
    st.info("Upload a CSV to begin.")
    st.stop()

raw_df = load_csv_bytes(uploaded.getvalue())
if raw_df.empty:
    st.error("CSV is empty.")
    st.stop()

cols = list(raw_df.columns)
c1, c2 = st.columns(2)
with c1:
    dt_col = st.selectbox("Datetime column", options=cols, index=0)
with c2:
    default_val_idx = 1 if len(cols) > 1 else 0
    val_col = st.selectbox("Value column (WL/SF)", options=cols, index=default_val_idx)

if not windows:
    st.warning("Select at least one rolling window.")
    st.stop()

# -----------------------------
# Run
# -----------------------------
if run:
    df = raw_df.copy()

    df[dt_col] = parse_dt_series(df[dt_col], tz=tz)
    df[val_col] = pd.to_numeric(df[val_col].replace("", np.nan), errors="coerce")

    df = df[df[dt_col].notna()].sort_values(dt_col)

    dt = df[dt_col]
    x = df[val_col].to_numpy(dtype=float)

    res_list = []
    for w in sorted(map(int, windows)):
        out = csd_roll(x=x, dt=dt, window_hours=w, valid_frac=float(valid_frac))
        if not out.empty:
            res_list.append(out)

    if not res_list:
        st.error("No results produced. Try smaller windows or lower valid fraction.")
        st.stop()

    res_all = pd.concat(res_list, ignore_index=True)

    # keep results in session
    st.session_state["res_all"] = res_all
    st.session_state["val_col"] = val_col

# show previous results if already computed
res_all = st.session_state.get("res_all", None)
if res_all is None:
    st.warning("Click **Run CSD** to compute indicators.")
    st.stop()

# -----------------------------
# Tabs
# -----------------------------
tab1, tab2, tab3 = st.tabs(["Preview", "Variance & PSD Plots", "Downloads"])

with tab1:
    st.subheader("CSD Results (preview)")
    st.dataframe(res_all, use_container_width=True, height=520)

with tab2:
    st.subheader("Variance & PSD Plots")

    available_windows = sorted(res_all["window_hours"].unique().astype(int).tolist())
    default_w = 24 if 24 in available_windows else available_windows[0]
    plot_w = st.selectbox("Window to plot (hours)", options=available_windows, index=available_windows.index(default_w))

    dfp = res_all[res_all["window_hours"] == int(plot_w)].copy()
    if dfp.empty:
        st.error("No data for this window.")
        st.stop()

    # Interactive plotly (hover tooltip like your screenshot)
    fig_var = make_plotly_line(dfp, "Variance", f"Variance ({plot_w}h window)", "Variance")
    fig_psd = make_plotly_line(dfp, "PSD_LowFreq", f"PSD Low-Frequency ({plot_w}h window)", "PSD_LowFreq")

    st.markdown("#### Interactive (Plotly)")
    st.plotly_chart(fig_var, use_container_width=True)
    st.plotly_chart(fig_psd, use_container_width=True)

    st.markdown("#### Static (Matplotlib) — used for PNG export")
    fig1 = make_matplotlib_line(dfp, "Variance", f"Variance ({plot_w}h window)", "Variance")
    fig2 = make_matplotlib_line(dfp, "PSD_LowFreq", f"PSD Low-Frequency ({plot_w}h window)", "PSD_LowFreq")
    st.pyplot(fig1, use_container_width=True)
    st.pyplot(fig2, use_container_width=True)

with tab3:
    st.subheader("Downloads")

    # combined CSV
    combined_csv = res_all.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download combined CSV",
        data=combined_csv,
        file_name=f"CSD_{safe_name(st.session_state.get('val_col','value'))}_6to168h.csv",
        mime="text/csv"
    )

    # per-window ZIP
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for w in sorted(res_all["window_hours"].unique()):
            dfw = res_all[res_all["window_hours"] == w]
            zf.writestr(f"CSD_{safe_name(st.session_state.get('val_col','value'))}_{int(w)}h.csv", dfw.to_csv(index=False))
    zip_buf.seek(0)

    st.download_button(
        "Download per-window ZIP (CSVs)",
        data=zip_buf.getvalue(),
        file_name=f"CSD_{safe_name(st.session_state.get('val_col','value'))}_per_window.zip",
        mime="application/zip"
    )

    # HTML export: BOTH plots in one HTML (with hover tooltips)
    # (This will show tooltips like your screenshot when you hover.)
    plot_w = int(st.session_state.get("last_plot_w", 24)) if "last_plot_w" in st.session_state else 24
    # Use the currently selected window if available in the session
    # If not, default to first available
    w_html = plot_w if plot_w in res_all["window_hours"].unique() else int(sorted(res_all["window_hours"].unique())[0])
    dfp_html = res_all[res_all["window_hours"] == w_html].copy()

    combo = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
        subplot_titles=(f"Variance ({w_html}h)", f"PSD Low-Frequency ({w_html}h)")
    )
    combo.add_trace(go.Scatter(x=dfp_html["datetime_start"], y=dfp_html["Variance"], mode="lines", name="Variance"), row=1, col=1)
    combo.add_trace(go.Scatter(x=dfp_html["datetime_start"], y=dfp_html["PSD_LowFreq"], mode="lines", name="PSD_LowFreq"), row=2, col=1)
    combo.update_layout(
        title=f"CSD Variance & PSD ({w_html}h window)",
        hovermode="x unified",
        xaxis=dict(rangeslider=dict(visible=True), tickangle=-45),
        xaxis2=dict(tickangle=-45),
        height=750,
        margin=dict(l=10, r=10, t=60, b=10)
    )

    # IMPORTANT: use "cdn" so file size is smaller; hover tooltips still work.
    html_bytes = combo.to_html(full_html=True, include_plotlyjs="cdn").encode("utf-8")
    st.download_button(
        "Download interactive HTML (Variance + PSD)",
        data=html_bytes,
        file_name=f"CSD_Variance_PSD_{w_html}h.html",
        mime="text/html"
    )

    # PNG exports (use current first window for export convenience)
    # Users can change window in "Variance & PSD Plots" tab and rerun download if needed.
    available_windows = sorted(res_all["window_hours"].unique().astype(int).tolist())
    w_png = 24 if 24 in available_windows else available_windows[0]
    dfp_png = res_all[res_all["window_hours"] == int(w_png)].copy()

    fig_var_png = make_matplotlib_line(dfp_png, "Variance", f"Variance ({w_png}h window)", "Variance")
    fig_psd_png = make_matplotlib_line(dfp_png, "PSD_LowFreq", f"PSD Low-Frequency ({w_png}h window)", "PSD_LowFreq")

    st.download_button(
        f"Download Variance PNG ({w_png}h)",
        data=fig_to_png_bytes(fig_var_png),
        file_name=f"Variance_{w_png}h.png",
        mime="image/png"
    )

    st.download_button(
        f"Download PSD PNG ({w_png}h)",
        data=fig_to_png_bytes(fig_psd_png),
        file_name=f"PSD_LowFreq_{w_png}h.png",
        mime="image/png"
    )

st.caption("CSD: Variance + PSD Low-Frequency (AR1 computed and exported in CSV).")
