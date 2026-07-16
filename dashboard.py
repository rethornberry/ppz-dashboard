import io
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

# ── Default pkl path (works when running locally) ────────────────────────────
DEFAULT_PKL = Path(__file__).parent / "rates_maxPPZ3000mm.pkl"

ERATIO_LABELS = {None: "No ERatio", "TomasOptimal": "Tomas Optimal"}
TRG_LABELS = {
    "single_egamma": "Single e/γ",
    "di_egamma":     "Di e/γ",
    "delta_ppz":     "ΔPPZ",
    "combo_and":     "Di e/γ AND ΔPPZ",
    "combo_or":      "Di e/γ OR ΔPPZ",
}
TRG_COLORS = {
    "single_egamma": "#1f77b4",
    "di_egamma":     "#ff7f0e",
    "delta_ppz":     "#2ca02c",
    "combo_and":     "#9467bd",
    "combo_or":      "#d62728",
}

PT_BINS    = np.array([15, 20, 30, 40, 50, 75, 100, 150, 200])
PT_CENTERS = 0.5 * (PT_BINS[:-1] + PT_BINS[1:])


@st.cache_data
def load_pkl(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


# ── Page setup ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="PPZ Trigger Dashboard", layout="wide")
st.title("PPZ Photon Pointing — Trigger Rate Dashboard")

# ── Load results ─────────────────────────────────────────────────────────────
results = None

if DEFAULT_PKL.exists():
    results = load_pkl(str(DEFAULT_PKL))
else:
    st.info("Local results file not found. Upload a .pkl file to continue.")
    uploaded = st.file_uploader("Upload results .pkl", type=["pkl"])
    if uploaded:
        results = pickle.load(io.BytesIO(uploaded.read()))

if results is None:
    st.stop()

# ── Introspect available keys ─────────────────────────────────────────────────
eratio_options = list(results.keys())
et_options     = list(results[eratio_options[0]].keys())
trg_options = list(results[eratio_options[0]][et_options[0]].keys())
sig_options = list(
    results[eratio_options[0]][et_options[0]][trg_options[0]]["efficiency"].keys()
)

# ── Sidebar controls ──────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")

    eratio_sel = st.selectbox(
        "ERatio condition",
        eratio_options,
        format_func=lambda x: ERATIO_LABELS.get(x, str(x)),
    )
    et_sel = st.selectbox("Min TOB ET [GeV]", et_options)
    plot_type = st.radio(
        "Plot type",
        ["Efficiency vs Rate", "Turn-on Curves", "Rate vs Threshold"],
    )

    st.divider()

    if plot_type == "Efficiency vs Rate":
        trg_sel_multi = st.multiselect(
            "Trigger types",
            trg_options,
            default=trg_options,
            format_func=lambda x: TRG_LABELS.get(x, x),
        )
        sig_sel = st.selectbox("Signal sample", sig_options)
        rate_budget        = st.number_input("Di/ΔPPZ rate budget [kHz]", value=60.0, step=10.0)
        rate_budget_single = st.number_input("Single e/γ budget [kHz]",   value=200.0, step=10.0)
        log_x = st.checkbox("Log x-axis (rate)", value=False)

    elif plot_type == "Turn-on Curves":
        trg_sel = st.selectbox(
            "Trigger type",
            trg_options,
            format_func=lambda x: TRG_LABELS.get(x, x),
        )
        sig_sel = st.selectbox("Signal sample", sig_options)
        eff_vs_pt_data = results[eratio_sel][et_sel][trg_sel].get("eff_vs_pt", {}).get(sig_sel, {})
        available_rate_points = sorted(eff_vs_pt_data.keys())
        rate_points_sel = st.multiselect(
            "Rate operating points",
            available_rate_points,
            default=available_rate_points,
            format_func=lambda x: f"{x/1e3:.0f} kHz",
        )

    elif plot_type == "Rate vs Threshold":
        trg_sel_multi = st.multiselect(
            "Trigger types",
            trg_options,
            default=trg_options,
            format_func=lambda x: TRG_LABELS.get(x, x),
        )
        log_y = st.checkbox("Log y-axis (rate)", value=False)


# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))

if plot_type == "Efficiency vs Rate":
    for trg in trg_sel_multi:
        rate = results[eratio_sel][et_sel][trg]["rate"] / 1e3
        eff  = results[eratio_sel][et_sel][trg]["efficiency"][sig_sel]
        ax.plot(rate, eff, label=TRG_LABELS.get(trg, trg), color=TRG_COLORS.get(trg), lw=2)

    ax.axvline(rate_budget,        color="red",    ls="--", lw=1.5, label=f"{rate_budget:.0f} kHz (di/ΔPPZ)")
    ax.axvline(rate_budget_single, color="orange", ls="--", lw=1.5, label=f"{rate_budget_single:.0f} kHz (single)")

    ax.set_xlabel("Trigger rate [kHz]")
    ax.set_ylabel("Signal efficiency")
    ax.set_title(
        f"{sig_sel}   |   ERatio: {ERATIO_LABELS.get(eratio_sel)}   |   Min ET: {et_sel} GeV"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    if log_x:
        ax.set_xscale("log")
    else:
        ax.set_xlim(left=0)

elif plot_type == "Turn-on Curves":
    if not available_rate_points:
        st.warning("No turn-on curve data in this pkl. Re-run trigger_rate.py to generate it.")
        st.stop()
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(rate_points_sel)))
    eff_vs_pt_dict = eff_vs_pt_data
    for color, rp in zip(colors, rate_points_sel):
        eff = eff_vs_pt_dict[rp]
        ax.plot(PT_CENTERS, eff, "o-", label=f"{rp/1e3:.0f} kHz", color=color, lw=2)

    ax.set_xlabel(r"Leading truth photon $p_T$ [GeV]")
    ax.set_ylabel("Signal efficiency")
    ax.set_title(
        f"{sig_sel}   |   {TRG_LABELS.get(trg_sel)}   |   "
        f"ERatio: {ERATIO_LABELS.get(eratio_sel)}   |   Min ET: {et_sel} GeV"
    )
    ax.legend(title="Rate operating point")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    ax.set_xlim(PT_BINS[0], PT_BINS[-1])

elif plot_type == "Rate vs Threshold":
    thresholds = results[eratio_sel][et_sel][trg_options[0]]["thresholds"]
    for trg in trg_sel_multi:
        rate = results[eratio_sel][et_sel][trg]["rate"] / 1e3
        ax.plot(thresholds, rate, label=TRG_LABELS.get(trg, trg), color=TRG_COLORS.get(trg), lw=2)

    ax.set_xlabel("PPZ threshold [mm]")
    ax.set_ylabel("Trigger rate [kHz]")
    ax.set_title(
        f"ERatio: {ERATIO_LABELS.get(eratio_sel)}   |   Min ET: {et_sel} GeV"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    if log_y:
        ax.set_yscale("log")

st.pyplot(fig)
plt.close(fig)
