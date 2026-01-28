# app.py
# ============================================================
# Streamlit App: CSD Rolling Indicators (Variance / PSD Low-Freq / AR1)
# - Upload CSV
# - Choose datetime & value column
# - Rolling windows (hours) + valid fraction filter
# - Tabs:
#     1) Preview (table + combined CSV + per-window ZIP)
#     2) Variance & PSD Plots (interactive Plotly + save HTML + PNG via Matplotlib)
#     3) AR1 Plot (interactive Plotly + save HTML + PNG via Matplotlib)
#     4) Messages
#
# Key fixes requested:
# - HTML hover tooltip shows FULL datetime (date + time)
# - Parsing is R-like (explicit formats first, dayfirst fallback)
# - Robust CSV read (avoids UnicodeDecodeError)
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
    Includes date-only formats too.
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

    # final fallback: inference but dayfirst=True (important to match R)
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
    Low-frequency PSD mean using rFFT power; excludes DC.
    (Stable and consistent; not identical to R spectrum() internals.)
    """
    w = w[np.isfinite(w)]
    if w.size < 4:
        return np.nan

    w = w - np.nanmean(w)
    spec = np.abs(np.fft.rfft(w)) ** 2
    if spec.size <= 1:
        return np.nan

    k = min(low_k, spec.size - 1)  # exclude DC
    if k < 1:
        return np.nan

    return float(np.nanmean(spec[1:k + 1]))


def csd_roll(x: np.ndarray, dt: pd.Series, window_hours: int, valid_frac: float = 0.8) -> pd.DataFrame:
    win = int(window_hours)
    low_k = min(8, max(2, win // 10))  # dynamic low_k (floor(win/10))

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
        # Sample variance to match R var()
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


# -----------------------------
# Plot helpers (FULL datetime hover)
# -----------------------------
HOVER_FMT = "%Y-%m-%d %H:%M:%S"

def make_plotly_line(df: pd.DataFrame, y: str, title: str, yaxis_title: str) -> go.Figure:
    fig = px.line(df, x="datetime_start", y=y, title=title)

    # Force hover to show full datetime + value
    fig.update_traces(
        hovertemplate=f"datetime: %{{x|{HOVER_FMT}}}<br>{yaxis_title}: %{{y:.6g}}<extra></extra>"
    )

    fig.update_layout(
        xaxis_title="DateTime",
        yaxis_title=yaxis_title,
        hovermode="x unified",
        xaxis=dict(
            tickangle=-45,
            rangeslider=dict(visible=True),
            hoverformat=HOVER_FMT,
        ),
        margin=dict(l=10, r=10, t=50, b=10)
    )
    return fig


def make_matplotlib_line(df: pd.DataFrame, y: str, title: str, ylabel: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df["datetime_start"], df[y])
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("DateTime")
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
# App
# -----------------------------
st.set_page_config(page_title="CSD Rolling Indicators", layout="wide")
st.title("CSD Rolling Indicators (Variance / PSD Low-Freq / AR1)")

with st.sidebar:
    st.header("Inputs")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    st.caption("Tip: Choose the datetime & value columns below.")

    tz = st.selectbox("Time zone for datetime parsing", ["Asia/Kuala_Lumpur", "UTC"], index=0)

    window_choices = [6, 12, 24, 36, 48, 72, 96, 120, 144, 168]
    windows = st.multiselect("Rolling windows (hours)", options=window_choices, default=window_choices)

    valid_frac = st.slider("Min valid data per window", min_value=0.50, max_value=1.00, value=0.80, step=0.05)

    run = st.button("Run CSD", type="primary")


def add_log(msg: str):
    st.session_state.setdefault("log", [])
    st.session_state["log"].append(msg)


if uploaded is None:
    st.info("Upload a CSV to begin.")
    st.stop()

raw_df = load_csv_bytes(uploaded.getvalue())
if raw_df.empty:
    st.error("Your CSV is empty.")
    st.stop()

cols = list(raw_df.columns)
c1, c2 = st.columns(2)
with c1:
    dt_col = st.selectbox("Datetime column", options=cols, index=0)
with c2:
    default_val_idx = 1 if len(cols) > 1 else 0
    val_col = st.selectbox("Value column (WL/SF)", options=cols, index=default_val_idx)

if not windows:
    st.warning("Please select at least one rolling window.")
    st.stop()

if run:
    st.session_state["log"] = []
    add_log("---- Run started ----")
    add_log(f"File: {uploaded.name}")
    add_log(f"Datetime col: {dt_col} | Value col: {val_col}")
    add_log(f"Timezone: {tz}")
    add_log(f"Windows: {', '.join(map(str, windows))} hours")
    add_log(f"Min valid fraction: {valid_frac}")

    df = raw_df.copy()

    # parse datetime + numeric values
    df[dt_col] = parse_dt_series(df[dt_col], tz=tz)
    df[val_col] = pd.to_numeric(df[val_col].replace("", np.nan), errors="coerce")

    # keep valid datetime rows & sort
    df = df[df[dt_col].notna()].sort_values(dt_col)

    dt = df[dt_col]
    x = df[val_col].to_numpy(dtype=float)

    # (Optional) log a quick sanity check
    add_log(f"Parsed datetime range: {dt.iloc[0]}  to  {dt.iloc[-1]}")
    add_log(f"Rows after datetime parse: {len(df)}")

    res_list = []
    for w in sorted(map(int, windows)):
        add_log(f"Processing window: {w} hours")
        out = csd_roll(x=x, dt=dt, window_hours=w, valid_frac=float(valid_frac))
        if not out.empty:
            res_list.append(out)

    if not res_list:
        add_log("No results produced (too few rows / too many missing).")
        st.error("No results produced. Try smaller windows or reduce 'Min valid data per window'.")
        st.stop()

    res_all = pd.concat(res_list, ignore_index=True)

    # Ensure datetime columns are datetime (keep timezone if present)
    res_all["datetime_start"] = pd.to_datetime(res_all["datetime_start"], errors="coerce")
    res_all["datetime_end"] = pd.to_datetime(res_all["datetime_end"], errors="coerce")

    add_log(f"Done. Rows: {len(res_all)}")
    add_log("---- Run finished ----")

    st.session_state["res_all"] = res_all
    st.session_state["val_col"] = val_col

# Use previous results if exist
res_all = st.session_state.get("res_all", None)
if res_all is None:
    st.warning("Click **Run CSD** to compute indicators.")
    st.stop()

# -----------------------------
# Tabs
# -----------------------------
tab1, tab2, tab3, tab4 = st.tabs(["Preview", "Variance & PSD Plots", "AR1 Plot", "Messages"])

with tab1:
    st.subheader("CSD Results (preview)")
    st.dataframe(res_all, use_container_width=True, height=520)

    # combined CSV download
    combined_csv = res_all.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download combined CSV",
        data=combined_csv,
        file_name=f"CSD_{safe_name(st.session_state.get('val_col','value'))}_combined.csv",
        mime="text/csv"
    )

    # per-window ZIP download
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for w in sorted(res_all["window_hours"].unique()):
            dfw = res_all[res_all["window_hours"] == w]
            zf.writestr(f"CSD_{int(w)}h.csv", dfw.to_csv(index=False))
    zip_buf.seek(0)

    st.download_button(
        "Download per-window ZIP (CSVs)",
        data=zip_buf.getvalue(),
        file_name=f"CSD_{safe_name(st.session_state.get('val_col','value'))}_per_window.zip",
        mime="application/zip"
    )

with tab2:
    st.subheader("Variance & PSD Plots")

    available_windows = sorted(res_all["window_hours"].unique().astype(int).tolist())
    default_w = 24 if 24 in available_windows else available_windows[0]

    plot_w = st.selectbox(
        "Window to plot (hours)",
        options=available_windows,
        index=available_windows.index(default_w),
        key="plot_w_varpsd"
    )

    dfp = res_all[res_all["window_hours"] == int(plot_w)].copy()
    if dfp.empty:
        st.error("No data for this window.")
        st.stop()

    # Interactive Plotly
    fig_var = make_plotly_line(dfp, "Variance", f"Variance ({plot_w}h window)", "Variance")
    fig_psd = make_plotly_line(dfp, "PSD_LowFreq", f"PSD Low-Frequency ({plot_w}h window)", "PSD_LowFreq")

    st.markdown("#### Interactive (Plotly) — hover shows full datetime + value")
    st.plotly_chart(fig_var, use_container_width=True)
    st.plotly_chart(fig_psd, use_container_width=True)

    # Single HTML with both plots (self-contained-ish: plotlyjs via CDN)
    combo = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
        subplot_titles=(f"Variance ({plot_w}h)", f"PSD Low-Frequency ({plot_w}h)")
    )
    combo.add_trace(go.Scatter(
        x=dfp["datetime_start"], y=dfp["Variance"], mode="lines", name="Variance",
        hovertemplate=f"datetime: %{{x|{HOVER_FMT}}}<br>Variance: %{{y:.6g}}<extra></extra>"
    ), row=1, col=1)

    combo.add_trace(go.Scatter(
        x=dfp["datetime_start"], y=dfp["PSD_LowFreq"], mode="lines", name="PSD_LowFreq",
        hovertemplate=f"datetime: %{{x|{HOVER_FMT}}}<br>PSD_LowFreq: %{{y:.6g}}<extra></extra>"
    ), row=2, col=1)

    combo.update_layout(
        title=f"CSD Variance & PSD ({plot_w}h window)",
        hovermode="x unified",
        xaxis=dict(rangeslider=dict(visible=True), tickangle=-45),
        xaxis2=dict(tickangle=-45),
        margin=dict(l=10, r=10, t=60, b=10),
        height=750
    )
    combo.update_xaxes(hoverformat=HOVER_FMT)

    html_bytes = combo.to_html(full_html=True, include_plotlyjs="cdn").encode("utf-8")
    st.download_button(
        "Download interactive HTML (Variance + PSD)",
        data=html_bytes,
        file_name=f"CSD_Variance_PSD_{plot_w}h.html",
        mime="text/html"
    )

    # Static plots (PNG)
    st.markdown("#### Static (PNG export uses these)")
    fig1 = make_matplotlib_line(dfp, "Variance", f"Variance ({plot_w}h window)", "Variance")
    fig2 = make_matplotlib_line(dfp, "PSD_LowFreq", f"PSD Low-Frequency ({plot_w}h window)", "PSD_LowFreq")
    st.pyplot(fig1, use_container_width=True)
    st.pyplot(fig2, use_container_width=True)

    png_var = fig_to_png_bytes(make_matplotlib_line(dfp, "Variance", f"Variance ({plot_w}h window)", "Variance"))
    png_psd = fig_to_png_bytes(make_matplotlib_line(dfp, "PSD_LowFreq", f"PSD Low-Frequency ({plot_w}h window)", "PSD_LowFreq"))

    cA, cB = st.columns(2)
    with cA:
        st.download_button("Download Variance PNG", data=png_var, file_name=f"Variance_{plot_w}h.png", mime="image/png")
    with cB:
        st.download_button("Download PSD PNG", data=png_psd, file_name=f"PSD_LowFreq_{plot_w}h.png", mime="image/png")

with tab3:
    st.subheader("AR1 Plot")

    available_windows_ar1 = sorted(res_all["window_hours"].unique().astype(int).tolist())
    default_w_ar1 = 24 if 24 in available_windows_ar1 else available_windows_ar1[0]

    plot_w_ar1 = st.selectbox(
        "Window to plot (hours)",
        options=available_windows_ar1,
        index=available_windows_ar1.index(default_w_ar1),
        key="plot_w_ar1"
    )

    dfp = res_all[res_all["window_hours"] == int(plot_w_ar1)].copy()
    if dfp.empty:
        st.error("No data for this window.")
        st.stop()

    # Interactive Plotly
    fig_ar1 = make_plotly_line(dfp, "AR1", f"AR1 (Lag-1 Autocorrelation) ({plot_w_ar1}h window)", "AR1")
    fig_ar1.add_hline(y=0, line_width=1, line_dash="dash")

    st.markdown("#### Interactive (Plotly) — hover shows full datetime + value")
    st.plotly_chart(fig_ar1, use_container_width=True)

    # HTML export AR1
    html_ar1 = fig_ar1.to_html(full_html=True, include_plotlyjs="cdn").encode("utf-8")
    st.download_button(
        "Download interactive HTML (AR1)",
        data=html_ar1,
        file_name=f"CSD_AR1_{plot_w_ar1}h.html",
        mime="text/html"
    )

    # Static PNG
    st.markdown("#### Static (PNG export uses this)")
    fig_ar1_png = make_matplotlib_line(dfp, "AR1", f"AR1 ({plot_w_ar1}h window)", "AR1")
    st.pyplot(fig_ar1_png, use_container_width=True)

    png_ar1 = fig_to_png_bytes(make_matplotlib_line(dfp, "AR1", f"AR1 ({plot_w_ar1}h window)", "AR1"))
    st.download_button("Download AR1 PNG", data=png_ar1, file_name=f"AR1_{plot_w_ar1}h.png", mime="image/png")

with tab4:
    st.subheader("Messages")
    logs = st.session_state.get("log", ["Upload a CSV to begin."])
    st.code("\n".join(logs), language="text")

st.caption("CSD rolling indicators: Variance, PSD Low-Frequency, AR1 (lag-1 autocorrelation).")
