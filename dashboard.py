import io
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
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
JZ_COLORS  = {"JZ0": "#4e79a7", "JZ1": "#f28e2b", "JZ2": "#59a14f"}
SIG_COLORS = {"HHbbyy": "darkorange", "Zee": "royalblue", "Hyy": "maroon"}

PT_BINS = np.array([15, 16, 17, 18, 20, 22, 25, 27, 30, 40, 50, 75, 100, 150, 200])

COMBO_TRGS = {"combo_and", "combo_or"}
PLOT_TYPES = [
    "Efficiency vs Rate",
    "Signal Comparison",
    "Rate by JZ Slice",
    "Turn-on Curves",
    "Rate vs Threshold",
    "3D Surface (Combo)",
]


@st.cache_data
def load_pkl(path: str, _mtime: float):
    with open(path, "rb") as f:
        return pickle.load(f)


# ── Page setup ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="PPZ Trigger Dashboard", layout="wide")
st.title("PPZ Photon Pointing — Trigger Rate Dashboard")

# ── Load results ─────────────────────────────────────────────────────────────
results = None

if DEFAULT_PKL.exists():
    results = load_pkl(str(DEFAULT_PKL), DEFAULT_PKL.stat().st_mtime)
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
trg_options    = list(results[eratio_options[0]][et_options[0]].keys())
sig_options    = list(results[eratio_options[0]][et_options[0]][trg_options[0]]["efficiency"].keys())
jz_options     = list(results[eratio_options[0]][et_options[0]][trg_options[0]]["rate_per_slice"].keys())
combo_options  = [t for t in trg_options if t in COMBO_TRGS]

# ── Sidebar controls ──────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")

    eratio_sel = st.selectbox(
        "ERatio condition", eratio_options,
        format_func=lambda x: ERATIO_LABELS.get(x, str(x)),
    )
    et_sel    = st.selectbox("Min TOB ET [GeV]", et_options)
    plot_type = st.radio("Plot type", PLOT_TYPES)

    st.divider()

    if plot_type == "Efficiency vs Rate":
        trg_sel_multi = st.multiselect(
            "Trigger types", trg_options, default=trg_options,
            format_func=lambda x: TRG_LABELS.get(x, x),
        )
        sig_sel            = st.selectbox("Signal sample", sig_options)
        rate_budget        = st.number_input("Di/ΔPPZ rate budget [kHz]", value=60.0,  step=10.0)
        rate_budget_single = st.number_input("Single e/γ budget [kHz]",   value=200.0, step=10.0)
        log_x              = st.checkbox("Log x-axis (rate)", value=False)

    elif plot_type == "Signal Comparison":
        trg_sel = st.selectbox(
            "Trigger type", trg_options,
            format_func=lambda x: TRG_LABELS.get(x, x),
        )
        sig_sel_multi      = st.multiselect("Signal samples", sig_options, default=sig_options)
        rate_budget        = st.number_input("Di/ΔPPZ rate budget [kHz]", value=60.0,  step=10.0)
        rate_budget_single = st.number_input("Single e/γ budget [kHz]",   value=200.0, step=10.0)
        log_x              = st.checkbox("Log x-axis (rate)", value=False)

    elif plot_type == "Rate by JZ Slice":
        trg_sel = st.selectbox(
            "Trigger type", trg_options,
            format_func=lambda x: TRG_LABELS.get(x, x),
        )
        stack = st.checkbox("Stacked area", value=True)
        log_y = st.checkbox("Log y-axis", value=False)

    elif plot_type == "Turn-on Curves":
        trg_sel = st.selectbox(
            "Trigger type", trg_options,
            format_func=lambda x: TRG_LABELS.get(x, x),
        )
        sig_sel = st.selectbox("Signal sample", sig_options)
        # combo triggers store turn-on in eff_vs_pt_2d; others use eff_vs_pt
        pt_key = "eff_vs_pt_2d" if trg_sel in COMBO_TRGS else "eff_vs_pt"
        eff_vs_pt_data      = results[eratio_sel][et_sel][trg_sel].get(pt_key, {}).get(sig_sel, {})
        available_rate_points = sorted(eff_vs_pt_data.keys())
        rate_points_sel     = st.multiselect(
            "Rate operating points", available_rate_points, default=available_rate_points,
            format_func=lambda x: f"{x/1e3:.0f} kHz",
        )

    elif plot_type == "Rate vs Threshold":
        trg_sel_multi = st.multiselect(
            "Trigger types", [t for t in trg_options if t not in COMBO_TRGS],
            default=[t for t in trg_options if t not in COMBO_TRGS],
            format_func=lambda x: TRG_LABELS.get(x, x),
        )
        log_y = st.checkbox("Log y-axis (rate)", value=False)

    elif plot_type == "3D Surface (Combo)":
        trg_sel  = st.selectbox(
            "Combo trigger", combo_options,
            format_func=lambda x: TRG_LABELS.get(x, x),
        )
        z_sel    = st.radio("Z axis", ["Rate [kHz]", "Signal efficiency"])
        if z_sel == "Signal efficiency":
            sig_sel = st.selectbox("Signal sample", sig_options)
        log_z    = st.checkbox("Log z-axis", value=False)


# ── Plot ──────────────────────────────────────────────────────────────────────

if plot_type == "3D Surface (Combo)":
    trg_data = results[eratio_sel][et_sel][trg_sel]
    ppz_t    = trg_data["ppz_thresholds"]
    delta_t  = trg_data["delta_thresholds"]

    if ppz_t is None:
        st.warning("No 2D data in this pkl. Re-run trigger_rate.py with the 2D grid code.")
        st.stop()

    if z_sel == "Rate [kHz]":
        z = trg_data["rate_2d"] / 1e3
        ztitle = "Rate [kHz]"
    else:
        z = trg_data["efficiency_2d"][sig_sel]
        ztitle = f"Signal efficiency ({sig_sel})"

    if log_z:
        z = np.log10(np.where(z > 0, z, np.nan))
        ztitle = f"log₁₀({ztitle})"

    fig3d = go.Figure(go.Surface(
        x=ppz_t,
        y=delta_t,
        z=z.T,           # plotly: z[i,j] corresponds to x[i], y[j]
        colorscale="Viridis",
        colorbar=dict(title=ztitle),
    ))
    fig3d.update_layout(
        scene=dict(
            xaxis_title="PPZ threshold t [mm]",
            yaxis_title="ΔPPZ threshold t' [mm]",
            zaxis_title=ztitle,
        ),
        title=f"{TRG_LABELS.get(trg_sel)}   |   ERatio: {ERATIO_LABELS.get(eratio_sel)}   |   Min ET: {et_sel} GeV",
        height=650,
        margin=dict(l=0, r=0, t=50, b=0),
    )
    st.plotly_chart(fig3d, use_container_width=True)

else:
    fig, ax = plt.subplots(figsize=(8, 5))

    if plot_type == "Efficiency vs Rate":
        fig_plotly = go.Figure()
        thresholds = results[eratio_sel][et_sel][trg_options[0]]["thresholds"]
        for trg in trg_sel_multi:
            if trg in COMBO_TRGS:
                st.info(f"{TRG_LABELS[trg]}: use '3D Surface (Combo)' — no 1D curve with independent thresholds.")
                continue
            rate = results[eratio_sel][et_sel][trg]["rate"] / 1e3
            eff  = results[eratio_sel][et_sel][trg]["efficiency"][sig_sel]
            fig_plotly.add_trace(go.Scatter(
                x=rate, y=eff,
                mode="lines",
                name=TRG_LABELS.get(trg, trg),
                line=dict(color=TRG_COLORS.get(trg), width=2),
                customdata=thresholds,
                hovertemplate="Rate: %{x:.1f} kHz<br>Efficiency: %{y:.3f}<br>PPZ threshold: %{customdata:.0f} mm<extra></extra>",
            ))
        fig_plotly.add_vline(x=rate_budget,        line=dict(color="red",    dash="dash", width=1.5), annotation_text=f"{rate_budget:.0f} kHz")
        fig_plotly.add_vline(x=rate_budget_single, line=dict(color="orange", dash="dash", width=1.5), annotation_text=f"{rate_budget_single:.0f} kHz")
        fig_plotly.update_layout(
            xaxis_title="Trigger rate [kHz]",
            yaxis_title="Signal efficiency",
            yaxis_range=[0, 1],
            xaxis_type="log" if log_x else "linear",
            title=f"{sig_sel}   |   ERatio: {ERATIO_LABELS.get(eratio_sel)}   |   Min ET: {et_sel} GeV",
            hovermode="x unified",
            height=500,
        )
        st.plotly_chart(fig_plotly, use_container_width=True)
        plt.close(fig)
        st.stop()

    elif plot_type == "Signal Comparison":
        for sig in sig_sel_multi:
            rate = results[eratio_sel][et_sel][trg_sel]["rate"] / 1e3
            eff  = results[eratio_sel][et_sel][trg_sel]["efficiency"][sig]
            ax.plot(rate, eff, label=sig, color=SIG_COLORS.get(sig), lw=2)
        budget = rate_budget_single if trg_sel == "single_egamma" else rate_budget
        ax.axvline(budget, color="red", ls="--", lw=1.5, label=f"{budget:.0f} kHz budget")
        ax.set_xlabel("Trigger rate [kHz]")
        ax.set_ylabel("Signal efficiency")
        ax.set_title(f"{TRG_LABELS.get(trg_sel)}   |   ERatio: {ERATIO_LABELS.get(eratio_sel)}   |   Min ET: {et_sel} GeV")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
        ax.set_xscale("log") if log_x else ax.set_xlim(left=0)

    elif plot_type == "Rate by JZ Slice":
        thresholds = results[eratio_sel][et_sel][trg_sel]["thresholds"]
        slices = {jz: results[eratio_sel][et_sel][trg_sel]["rate_per_slice"][jz] / 1e3
                  for jz in jz_options}
        if stack:
            ax.stackplot(thresholds,
                         [slices[jz] for jz in jz_options],
                         labels=jz_options,
                         colors=[JZ_COLORS.get(jz, f"C{i}") for i, jz in enumerate(jz_options)],
                         alpha=0.8)
        else:
            for jz in jz_options:
                ax.plot(thresholds, slices[jz], label=jz, color=JZ_COLORS.get(jz), lw=2)
            total = results[eratio_sel][et_sel][trg_sel]["rate"] / 1e3
            ax.plot(thresholds, total, "k--", lw=1.5, label="Total")
        ax.set_xlabel("PPZ threshold [mm]")
        ax.set_ylabel("Trigger rate [kHz]")
        ax.set_title(f"{TRG_LABELS.get(trg_sel)}   |   ERatio: {ERATIO_LABELS.get(eratio_sel)}   |   Min ET: {et_sel} GeV")
        ax.legend()
        ax.grid(True, alpha=0.3)
        if log_y:
            ax.set_yscale("log")

    elif plot_type == "Turn-on Curves":
        if not available_rate_points:
            st.warning("No turn-on curve data in this pkl. Re-run trigger_rate.py to generate it.")
            st.stop()
        colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(rate_points_sel)))
        for color, rp in zip(colors, rate_points_sel):
            eff = eff_vs_pt_data[rp]
            n   = len(eff)
            pt_centers = 0.5 * (PT_BINS[:n] + PT_BINS[1:n+1])
            ax.plot(pt_centers, eff, "o-", label=f"{rp/1e3:.0f} kHz", color=color, lw=2)
        ax.set_xlabel(r"Subleading truth photon $p_T$ [GeV]")
        ax.set_ylabel("Signal efficiency")
        ax.set_title(
            f"{sig_sel}   |   {TRG_LABELS.get(trg_sel)}   |   "
            f"ERatio: {ERATIO_LABELS.get(eratio_sel)}   |   Min ET: {et_sel} GeV"
        )
        ax.legend(title="Rate operating point")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
        ax.set_xlim(PT_BINS[0], PT_BINS[n])

    elif plot_type == "Rate vs Threshold":
        thresholds = results[eratio_sel][et_sel][trg_options[0]]["thresholds"]
        for trg in trg_sel_multi:
            rate = results[eratio_sel][et_sel][trg]["rate"] / 1e3
            ax.plot(thresholds, rate, label=TRG_LABELS.get(trg, trg), color=TRG_COLORS.get(trg), lw=2)
        ax.set_xlabel("PPZ threshold [mm]")
        ax.set_ylabel("Trigger rate [kHz]")
        ax.set_title(f"ERatio: {ERATIO_LABELS.get(eratio_sel)}   |   Min ET: {et_sel} GeV")
        ax.legend()
        ax.grid(True, alpha=0.3)
        if log_y:
            ax.set_yscale("log")

    st.pyplot(fig)
    plt.close(fig)
