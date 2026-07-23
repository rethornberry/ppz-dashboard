import io
import pickle
from pathlib import Path

import matplotlib.pyplot as plt  # still needed for viridis colormap
import numpy as np
import plotly.graph_objects as go
import streamlit as st

# ── Default pkl path (works when running locally) ────────────────────────────
DEFAULT_PKL = Path(__file__).parent / "rates_maxPPZ3000mm.pkl"

ERATIO_LABELS = {None: "No ERatio", "TomasOptimal": "Tomas Optimal"}
ALG_LABELS = {
    "PPZL1L2_MaxCell":           "Max Cell",
    "PPZL1L2_MaxTower_MaxCell":  "Max Tower - Max Cell",
    "PPZL1L2_AvgMaxCell":        "Avg Max Cell",
    "PPZL1L2BE_5Cells_Multi":    "5 Cells Multi",
    "Avg_PPZL1L2BE_5Cells":      "Avg 5 Cells",
    "PPZL1L2_Multi5L1_GeoL2":    "Multi5 L1 Geo L2",
}
TRG_LABELS = {
    "single_egamma": "Single e/γ",
    "di_egamma":     "Di e/γ",
    "delta_ppz":     "ΔPPZ",
    "combo_and":     "Di-e/γ AND ΔPPZ",
    "combo_or":      "Di-e/γ OR ΔPPZ",
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
QUANTITIES = ["Threshold [mm]", "Rate [kHz]", "Signal efficiency"]
SPLIT_DIMS = ["Trigger", "Signal", "Algorithm", "ERatio", "Min ET [GeV]"]
PREMADE_PLOTS = [
    "Efficiency vs Rate",
    "ROC Curve",
    "Rate by JZ Slice",
    "Turn-on Curves",
    "Rate vs Threshold",
    "3D Surface (Combo)",
    "2D Heatmap (Combo)",
]

TOTAL_INPUT_RATE = 30.92e6 + 30.91e6 + 1.84e6  # Hz
JZ_INPUT_RATE = {"JZ0": 30.92e6, "JZ1": 30.91e6, "JZ2": 1.84e6}


@st.cache_data
def load_pkl(path: str, _mtime: float):
    with open(path, "rb") as f:
        return pickle.load(f)


def get_array(results, eratio, min_et, alg, trg, sig, quantity):
    d = results[eratio][min_et][alg][trg]
    if quantity == "Threshold [mm]":
        return d["thresholds"]
    elif quantity == "Rate [kHz]":
        return d["rate"] / 1e3
    elif quantity == "Signal efficiency":
        return d["efficiency"][sig]


# ── Page setup ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="PPZ Trigger Dashboard", layout="wide")
st.title("PPZ Photon Pointing — Trigger Rate Dashboard")

# ── Load results ─────────────────────────────────────────────────────────────
results = None
data    = None

if DEFAULT_PKL.exists():
    data = load_pkl(str(DEFAULT_PKL), DEFAULT_PKL.stat().st_mtime)
else:
    st.info("Local results file not found. Upload a .pkl file to continue.")
    uploaded = st.file_uploader("Upload results .pkl", type=["pkl"])
    if uploaded:
        data = pickle.load(io.BytesIO(uploaded.read()))

if data is not None:
    results    = data["results"]
    yields_bkg = data.get("yields_bkg", {})
    yields_sig = data.get("yields_sig", {})
    # backward compat: old pkls only stored n_raw scalars
    if not yields_bkg and "n_raw_bkg" in data:
        yields_bkg = {jz: {"n_raw": n} for jz, n in data["n_raw_bkg"].items()}
    if not yields_sig and "n_raw_sig" in data:
        yields_sig = {sig: {"n_raw": n} for sig, n in data["n_raw_sig"].items()}

if data is None:
    st.stop()

# ── Introspect available keys ─────────────────────────────────────────────────
eratio_options = list(results.keys())
et_options     = list(results[eratio_options[0]].keys())
alg_options    = list(results[eratio_options[0]][et_options[0]].keys())
trg_options    = list(results[eratio_options[0]][et_options[0]][alg_options[0]].keys())
sig_options    = list(results[eratio_options[0]][et_options[0]][alg_options[0]][trg_options[0]]["efficiency"].keys())
jz_options     = list(results[eratio_options[0]][et_options[0]][alg_options[0]][trg_options[0]]["rate_per_slice"].keys())
combo_options  = [t for t in trg_options if t in COMBO_TRGS]

# ── Sidebar controls ──────────────────────────────────────────────────────────
with st.sidebar:
    page = st.radio("", ["Dashboard", "Cutflow", "Documentation"], horizontal=True)
    st.divider()

    if page == "Dashboard":
        mode = st.radio("", ["Explorer", "Pre-made plots"], horizontal=True)
        if mode == "Pre-made plots":
            plot_type = st.selectbox("Plot", PREMADE_PLOTS)
        else:
            plot_type = "Explorer"

        st.divider()
        st.header("Settings")
        eratio_sel = st.selectbox(
            "ERatio condition", eratio_options,
            format_func=lambda x: ERATIO_LABELS.get(x, str(x)),
        )
        et_sel  = st.selectbox("Min TOB ET [GeV]", et_options)
        default_alg = "Avg_PPZL1L2BE_5Cells"
        alg_sel = st.selectbox("PPZ algorithm", alg_options,
                               index=alg_options.index(default_alg) if default_alg in alg_options else 0,
                               format_func=lambda x: ALG_LABELS.get(x, x))
        st.divider()

        if plot_type == "Efficiency vs Rate":
            trg_sel_multi = st.multiselect(
                "Trigger types", trg_options, default=trg_options,
                format_func=lambda x: TRG_LABELS.get(x, x),
            )
            sig_sel = st.selectbox("Signal sample", sig_options)
            log_x   = st.checkbox("Log x-axis (rate)", value=False)
            show_budget        = st.checkbox("Show Di/ΔPPZ rate budget", value=True)
            rate_budget        = st.number_input("Di/ΔPPZ budget [kHz]", value=60.0,  step=10.0) if show_budget else None
            show_budget_single = "single_egamma" in trg_sel_multi and st.checkbox("Show Single e/γ rate budget", value=True)
            rate_budget_single = st.number_input("Single e/γ budget [kHz]", value=200.0, step=10.0) if show_budget_single else None
            show_baselines     = st.checkbox("Show no-PPZ baselines", value=True)

        elif plot_type == "ROC Curve":
            trg_sel_multi = st.multiselect(
                "Trigger types", trg_options, default=trg_options,
                format_func=lambda x: TRG_LABELS.get(x, x),
            )
            sig_sel  = st.selectbox("Signal sample", sig_options)
            x_mode   = st.radio("X axis", ["Background efficiency", "Background rejection (1−ε)", "Rejection factor (1/ε)"])
            log_x_roc = st.checkbox("Log x-axis", value=(x_mode == "Rejection factor (1/ε)"))
            show_auc  = st.checkbox("Show AUC in legend", value=True)

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
            pt_key = "eff_vs_pt_2d" if trg_sel in COMBO_TRGS else "eff_vs_pt"
            eff_vs_pt_data        = results[eratio_sel][et_sel][alg_sel][trg_sel].get(pt_key, {}).get(sig_sel, {})
            available_rate_points = sorted(eff_vs_pt_data.keys())
            rate_points_sel       = st.multiselect(
                "Rate operating points", available_rate_points, default=available_rate_points,
                format_func=lambda x: f"{x/1e3:.0f} kHz",
            )

        elif plot_type == "Rate vs Threshold":
            trg_sel_multi = st.multiselect(
                "Trigger types", trg_options, default=trg_options,
                format_func=lambda x: TRG_LABELS.get(x, x),
            )
            log_y          = st.checkbox("Log y-axis (rate)", value=False)
            show_baselines = st.checkbox("Show no-PPZ baselines", value=True)

        elif plot_type == "Explorer":
            x_qty     = st.selectbox("X axis", QUANTITIES, index=2)
            y_qty     = st.selectbox("Y axis", QUANTITIES, index=1)
            split_by  = st.selectbox("Split lines by", SPLIT_DIMS)
            log_x_exp = st.checkbox("Log X", value=False)
            log_y_exp = st.checkbox("Log Y", value=False)
            st.divider()

            if split_by == "Trigger":
                split_vals = st.multiselect("Triggers", trg_options, default=trg_options,
                                            format_func=lambda x: TRG_LABELS.get(x, x))
            elif split_by == "Signal":
                split_vals = st.multiselect("Signals", sig_options, default=sig_options)
            elif split_by == "Algorithm":
                split_vals = st.multiselect("Algorithms", alg_options, default=alg_options,
                                            format_func=lambda x: ALG_LABELS.get(x, x))
            elif split_by == "ERatio":
                split_vals = st.multiselect("ERatio", eratio_options, default=eratio_options,
                                            format_func=lambda x: ERATIO_LABELS.get(x, str(x)))
            elif split_by == "Min ET [GeV]":
                split_vals = st.multiselect("Min ET", et_options, default=et_options)

            if split_by != "ERatio":
                exp_eratio = st.selectbox("ERatio (fixed)", eratio_options,
                                          format_func=lambda x: ERATIO_LABELS.get(x, str(x)))
            else:
                exp_eratio = None
            if split_by != "Min ET [GeV]":
                exp_et = st.selectbox("Min ET (fixed)", et_options)
            else:
                exp_et = None
            if split_by != "Algorithm":
                exp_alg = st.selectbox("Algorithm (fixed)", alg_options,
                                       format_func=lambda x: ALG_LABELS.get(x, x))
            else:
                exp_alg = None
            if split_by != "Trigger":
                exp_trg = st.selectbox("Trigger (fixed)", trg_options,
                                       format_func=lambda x: TRG_LABELS.get(x, x))
            else:
                exp_trg = None
            needs_sig = ("Signal efficiency" in (x_qty, y_qty))
            if split_by != "Signal" and needs_sig:
                exp_sig = st.selectbox("Signal (fixed)", sig_options)
            else:
                exp_sig = None

        elif plot_type == "3D Surface (Combo)":
            trg_sel = st.selectbox(
                "Combo trigger", combo_options,
                format_func=lambda x: TRG_LABELS.get(x, x),
            )
            z_sel = st.radio("Z axis", ["Rate [kHz]", "Signal efficiency"])
            if z_sel == "Signal efficiency":
                sig_sel = st.selectbox("Signal sample", sig_options)
            log_z = st.checkbox("Log z-axis", value=False)

        elif plot_type == "2D Heatmap (Combo)":
            trg_sel = st.selectbox(
                "Combo trigger", combo_options,
                format_func=lambda x: TRG_LABELS.get(x, x),
            )
            sig_sel = st.selectbox("Signal sample", sig_options)
            rate_contours = st.multiselect(
                "Rate contours [kHz]", [10, 20, 30, 60, 100, 200],
                default=[30, 60, 100],
            )
            highlight_contour = st.number_input("Highlight contour [kHz]", value=60, step=10)

    elif page == "Cutflow":
        st.header("Settings")
        cf_eratio_sel = st.selectbox(
            "ERatio condition", eratio_options,
            format_func=lambda x: ERATIO_LABELS.get(x, str(x)),
            key="cf_eratio",
        )
        cf_et_sel = st.selectbox("Min TOB ET [GeV]", et_options, key="cf_et")
        default_alg = "Avg_PPZL1L2BE_5Cells"
        cf_alg_sel = st.selectbox("PPZ algorithm", alg_options,
                                  index=alg_options.index(default_alg) if default_alg in alg_options else 0,
                                  format_func=lambda x: ALG_LABELS.get(x, x), key="cf_alg")
        cf_trg_sel = st.selectbox("Trigger type", trg_options,
                                  format_func=lambda x: TRG_LABELS.get(x, x), key="cf_trg")
        cf_rate_budget = st.number_input("PPZ trigger rate budget [kHz]", value=60.0, step=10.0, key="cf_budget")


# ── Documentation page ───────────────────────────────────────────────────────

if page == "Documentation":
    st.markdown("""
## PPZ Trigger Rate Study

This dashboard shows trigger rate and signal efficiency results for photon pointing (PPZ)
triggers at the ATLAS HL-LHC L1 trigger level, using eFex TOBs.

---

### Samples

| Sample | Type | Description |
|--------|------|-------------|
| HHbbyy | Signal | HH → bbyy, μ=200 |
| Zee | Signal | Z → ee, μ=200 |
| Hyy | Signal | H → yy |
| JZ0 | Background | QCD dijets, pT slice 0 (max pT 20 GeV) |
| JZ1 | Background | QCD dijets, pT slice 1 (max pT 60 GeV) |
| JZ2 | Background | QCD dijets, pT slice 2 (max pT 160 GeV) |

Input rates: JZ0 = 30.92 MHz, JZ1 = 30.91 MHz, JZ2 = 1.84 MHz.

---

### Pre-selections

Pre-selection to TOBs
- **|η| < 2.4** and outside crack region (1.37 < |η| < 1.52)
- TOB matched to at least one tower (min 10 GeV) with ΔR < 0.1
- TOB ET > min (scan: 10, 15, 20 GeV)
- Loose shower shape cuts
- ERatio selection (None, Tomas Optimal, or 95 Flat working point)
- **|PPZ| < 3000 mm** applied to all TOBs before any trigger logic
    - To remove outliers

Events must have at least one surviving TOB after pre-selections

Signal Truth Matching

- Signal samples require truth-matched leading and subleading photon TOBs (exactly 2 matched)
    - [photon matching algorithm]
- Subleading truth photon pT > 15 GeV (applied at ntuple production)


---

### PPZ Algorithms

| Algorithm | Short name |
|-----------|-----------|
| PPZL1L2_MaxCell | Max Cell |
| PPZL1L2_MaxTower_MaxCell | Max Tower - Max Cell |
| PPZL1L2_AvgMaxCell | Avg Max Cell |
| PPZL1L2BE_5Cells_Multi | 5 Cells Multi |
| Avg_PPZL1L2BE_5Cells | Avg 5 Cells |
| PPZL1L2_Multi5L1_GeoL2 | Multi5 L1 Geo L2 |

Each algorithm estimates the z-position of the photon origin using L1 and L2 calorimeter
cell positions. Signal photons from H/HH decays point to the primary vertex (z ~ 0 mm).

---

### Trigger Types

| Trigger | Logic | Description |
|---------|-------|-------------|
| Single e/γ | min\|PPZ\| < t | At least one TOB points to z ~ 0 |
| Di e/γ | 2nd-min\|PPZ\| < t | Both leading and subleading point to z ~ 0 |
| ΔPPZ | min\|ΔPPZ\| < t | Two TOBs consistent with same vertex (vertex-agnostic) |
| Di e/γ AND ΔPPZ | both < t | Both conditions simultaneously (t = t') |
| Di e/γ OR ΔPPZ | either < t | Either condition (t = t') |

The **3D Surface** plot uses independent thresholds t (PPZ) and t' (ΔPPZ) for the combo triggers.

---

### Rate Formula

Rate = Σ_JZ [ JZ_input_rate × (n_events_passing / n_raw) ]

where n_raw is the total number of events before any selection.

---

### Signal Efficiency

ε = n_events_passing_trigger / n_raw

Denominator is the raw event count before any TOB or PPZ selection, so efficiency
reflects the full acceptance × trigger performance.

**Turn-on curves** show efficiency as a function of subleading truth photon pT
at fixed rate operating points.

---

### ERatio

Shower shape discriminant against hadronic fakes. Applied per-TOB in the barrel (|η| < 1.37).
Endcap TOBs always pass. Working points:

- **None**: no ERatio cut
- **Tomas Optimal**: optimised cut as a function of ET and η

---
""")
    st.stop()

# ── Cutflow page ──────────────────────────────────────────────────────────────

if page == "Cutflow":
    import pandas as pd

    if not yields_bkg or not yields_sig:
        st.warning("yields not found in pkl — regenerate with updated trigger_rate.py.")
        st.stop()
    has_tob          = any("n_tob"         in v for v in yields_bkg.values())
    has_eta          = any("n_eta"         in v for v in yields_bkg.values())
    has_tower_match  = any("n_tower_match" in v for v in yields_bkg.values())

    eratio_label = ERATIO_LABELS.get(cf_eratio_sel, str(cf_eratio_sel))
    st.header(f"Cutflow — {TRG_LABELS.get(cf_trg_sel)}  |  {ALG_LABELS.get(cf_alg_sel, cf_alg_sel)}  |  ET > {cf_et_sel} GeV")

    # find threshold index at rate budget
    rate_curve = results[cf_eratio_sel][cf_et_sel][cf_alg_sel][cf_trg_sel]["rate"]
    t_idx = int(np.argmin(np.abs(rate_curve - cf_rate_budget * 1e3)))

    # steps: (label, eratio_key, threshold_idx)
    # threshold_idx == None → Raw, "eta" → η mask, -1 → pre-PPZ, int → PPZ scan point
    tob_step         = [("≥1 TOB",         None, "tob")]          if has_tob         else []
    eta_step         = [("After η mask",   None, "eta")]          if has_eta         else []
    tower_match_step = [("Tower match",    None, "tower_match")]  if has_tower_match else []
    if cf_eratio_sel is None:
        steps = (
            [("Raw", None, None)]
            + tob_step
            + eta_step
            + tower_match_step
            + [
                (f"Min ET > {cf_et_sel} GeV",               None, -1),
                (f"PPZ trigger ({cf_rate_budget:.0f} kHz)",  None, t_idx),
            ]
        )
    else:
        steps = (
            [("Raw", None, None)]
            + tob_step
            + eta_step
            + tower_match_step
            + [
                (f"Min ET > {cf_et_sel} GeV",               None,          -1),
                (f"+ {eratio_label}",                        cf_eratio_sel, -1),
                (f"PPZ trigger ({cf_rate_budget:.0f} kHz)",  cf_eratio_sel, t_idx),
            ]
        )

    step_labels = [s[0] for s in steps]

    # ── Compute background values ─────────────────────────────────────────────
    def bkg_rate_at(eratio, tidx, jz):
        n_raw = yields_bkg[jz]["n_raw"]
        if tidx is None:
            return JZ_INPUT_RATE[jz] / 1e3
        if tidx in ("tob", "eta", "tower_match"):
            key = {"tob": "n_tob", "eta": "n_eta", "tower_match": "n_tower_match"}[tidx]
            n = yields_bkg[jz].get(key, n_raw)
            return JZ_INPUT_RATE[jz] / 1e3 * (n / n_raw) if n_raw > 0 else 0.0
        return results[eratio][cf_et_sel][cf_alg_sel][cf_trg_sel]["rate_per_slice"][jz][tidx] / 1e3

    def bkg_n_at(eratio, tidx, jz):
        n_raw = yields_bkg[jz]["n_raw"]
        if tidx is None:
            return n_raw
        if tidx in ("tob", "eta", "tower_match"):
            key = {"tob": "n_tob", "eta": "n_eta", "tower_match": "n_tower_match"}[tidx]
            return yields_bkg[jz].get(key, n_raw)
        return int(bkg_rate_at(eratio, tidx, jz) * 1e3 / JZ_INPUT_RATE[jz] * n_raw)

    bkg_data = {jz: [{"label": lbl, "n": bkg_n_at(era, ti, jz), "rate": bkg_rate_at(era, ti, jz)}
                      for lbl, era, ti in steps] for jz in jz_options}
    bkg_total = [{"label": lbl,
                  "n":    sum(bkg_n_at(era, ti, jz) for jz in jz_options),
                  "rate": (TOTAL_INPUT_RATE / 1e3 if ti is None else
                           sum(bkg_rate_at(era, ti, jz) for jz in jz_options) if ti in ("tob", "eta", "tower_match") else
                           results[era][cf_et_sel][cf_alg_sel][cf_trg_sel]["rate"][ti] / 1e3)}
                 for lbl, era, ti in steps]

    # ── Compute signal values ─────────────────────────────────────────────────
    def sig_eff_at(eratio, tidx, sig):
        n_raw = yields_sig[sig]["n_raw"]
        if tidx is None:
            return 1.0
        if tidx in ("tob", "eta", "tower_match"):
            key = {"tob": "n_tob", "eta": "n_eta", "tower_match": "n_tower_match"}[tidx]
            n = yields_sig[sig].get(key, n_raw)
            return n / n_raw if n_raw > 0 else 1.0
        return float(results[eratio][cf_et_sel][cf_alg_sel][cf_trg_sel]["efficiency"][sig][tidx])

    sig_data = {sig: [{"label": lbl, "eff": sig_eff_at(era, ti, sig),
                        "n": int(sig_eff_at(era, ti, sig) * yields_sig[sig]["n_raw"])}
                       for lbl, era, ti in steps]
                for sig in sig_options if sig in yields_sig}

    # ── Tabs ─────────────────────────────────────────────────────────────────
    tab1, tab2 = st.tabs(["Table", "Waterfall"])

    # ── Tab 1: Table ─────────────────────────────────────────────────────────
    with tab1:
        st.subheader("Background")
        bkg_cols = {("", "Cut"): [lbl for lbl, _, _ in steps]}
        for jz in jz_options:
            bkg_cols[(jz, "N")]         = []
            bkg_cols[(jz, "ε (%)")]     = []
            bkg_cols[(jz, "Rate (kHz)")] = []
        bkg_cols[("Total", "N")]            = []
        bkg_cols[("Total", "Rate (kHz)")]  = []
        bkg_cols[("Total", "Rejection")]   = []
        for i in range(len(steps)):
            for jz in jz_options:
                n  = bkg_data[jz][i]["n"]
                n0 = bkg_data[jz][0]["n"]
                bkg_cols[(jz, "N")].append(f"{n:,}")
                bkg_cols[(jz, "ε (%)")].append(f"{100*n/n0:.1f}" if n0 > 0 else "—")
                bkg_cols[(jz, "Rate (kHz)")].append(f"{bkg_data[jz][i]['rate']:.1f}")
            bkg_cols[("Total", "N")].append(f"{bkg_total[i]['n']:,}")
            bkg_cols[("Total", "Rate (kHz)")].append(f"{bkg_total[i]['rate']:.1f}")
            if i == 0 or bkg_total[i]["rate"] == 0:
                bkg_cols[("Total", "Rejection")].append("—")
            else:
                bkg_cols[("Total", "Rejection")].append(f"÷{bkg_total[i-1]['rate'] / bkg_total[i]['rate']:.1f}")
        bkg_df = pd.DataFrame(bkg_cols)
        bkg_df.columns = pd.MultiIndex.from_tuples(bkg_df.columns)
        st.dataframe(bkg_df, hide_index=True, use_container_width=True)

        st.subheader("Signal")
        sig_cols = {("", "Cut"): [lbl for lbl, _, _ in steps]}
        for sig in sig_data:
            sig_cols[(sig, "N")]     = []
            sig_cols[(sig, "ε (%)")]  = []
        for i in range(len(steps)):
            for sig, sdata in sig_data.items():
                n0 = sdata[0]["n"]
                sig_cols[(sig, "N")].append(f"{sdata[i]['n']:,}")
                sig_cols[(sig, "ε (%)")].append(f"{100*sdata[i]['eff']:.1f}")
        sig_df = pd.DataFrame(sig_cols)
        sig_df.columns = pd.MultiIndex.from_tuples(sig_df.columns)
        st.dataframe(sig_df, hide_index=True, use_container_width=True)

    # ── Tab 2: Waterfall ─────────────────────────────────────────────────────
    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Background — rate by JZ slice (kHz)")
            fig = go.Figure()
            for jz in jz_options:
                fig.add_trace(go.Bar(
                    name=jz,
                    x=step_labels,
                    y=[d["rate"] for d in bkg_data[jz]],
                    marker_color=JZ_COLORS.get(jz),
                    text=[f"{d['rate']:.0f}" for d in bkg_data[jz]],
                    textposition="inside",
                ))
            # overlay total rate as a line
            fig.add_trace(go.Scatter(
                x=step_labels,
                y=[d["rate"] for d in bkg_total],
                mode="lines+markers+text",
                name="Total",
                line=dict(color="white", width=2, dash="dot"),
                marker=dict(size=7),
                text=[f"{d['rate']:.0f} kHz" for d in bkg_total],
                textposition="top center",
                textfont=dict(size=11),
            ))
            fig.update_layout(
                barmode="stack", height=420,
                yaxis_title="Rate (kHz)", yaxis_type="log",
                margin=dict(l=0, r=0, t=10, b=0),
                legend=dict(orientation="h", y=1.05),
            )
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.subheader("Signal — efficiency (%)")
            fig = go.Figure()
            for sig, sdata in sig_data.items():
                fig.add_trace(go.Bar(
                    name=sig,
                    x=step_labels,
                    y=[d["eff"] * 100 for d in sdata],
                    marker_color=SIG_COLORS.get(sig),
                    text=[f"{d['eff']*100:.1f}%" for d in sdata],
                    textposition="inside",
                ))
            fig.update_layout(
                barmode="group", height=420,
                yaxis_title="Efficiency (%)", yaxis_range=[0, 105],
                margin=dict(l=0, r=0, t=10, b=0),
                legend=dict(orientation="h", y=1.05),
            )
            st.plotly_chart(fig, use_container_width=True)


    st.stop()

# ── Plot ──────────────────────────────────────────────────────────────────────

if plot_type == "3D Surface (Combo)":
    trg_data = results[eratio_sel][et_sel][alg_sel][trg_sel]
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
        x=ppz_t, y=delta_t, z=z.T,
        colorscale="Viridis",
        colorbar=dict(title=ztitle),
    ))
    fig3d.update_layout(
        scene=dict(
            xaxis_title="PPZ threshold t [mm]",
            yaxis_title="ΔPPZ threshold t' [mm]",
            zaxis_title=ztitle,
        ),
        title=f"{TRG_LABELS.get(trg_sel)}   |   {ALG_LABELS.get(alg_sel, alg_sel)}   |   ERatio: {ERATIO_LABELS.get(eratio_sel)}   |   Min ET: {et_sel} GeV",
        height=650,
        margin=dict(l=0, r=0, t=50, b=0),
    )
    st.plotly_chart(fig3d, use_container_width=True)

elif plot_type == "2D Heatmap (Combo)":
    trg_data = results[eratio_sel][et_sel][alg_sel][trg_sel]
    ppz_t    = trg_data["ppz_thresholds"]
    delta_t  = trg_data["delta_thresholds"]

    if ppz_t is None:
        st.warning("No 2D data in this pkl. Re-run trigger_rate.py with the 2D grid code.")
        st.stop()

    eff_2d  = trg_data["efficiency_2d"][sig_sel]
    rate_2d = trg_data["rate_2d"] / 1e3  # kHz

    # find optimal point inside highlight contour
    feasible = rate_2d <= highlight_contour
    if np.any(feasible):
        masked = np.where(feasible, eff_2d, -1)
        oi, oj = np.unravel_index(np.argmax(masked), masked.shape)
        opt_t   = ppz_t[oi]
        opt_dt  = delta_t[oj]
        opt_eff = eff_2d[oi, oj]
        opt_rate = rate_2d[oi, oj]
    else:
        oi, oj = None, None

    fig2d = go.Figure()

    # efficiency heatmap
    fig2d.add_trace(go.Heatmap(
        x=ppz_t, y=delta_t, z=eff_2d.T,
        colorscale="Viridis",
        colorbar=dict(title=f"Signal efficiency ({sig_sel})"),
        hovertemplate="t: %{x:.0f} mm<br>t': %{y:.0f} mm<br>Efficiency: %{z:.3f}<extra></extra>",
        zmin=0, zmax=1,
    ))

    # rate contour lines — no legend entries, labels drawn on the lines
    contour_colors = {c: ("red" if c == highlight_contour else "white") for c in rate_contours}
    contour_widths = {c: (3   if c == highlight_contour else 1)         for c in rate_contours}
    for level in sorted(rate_contours):
        fig2d.add_trace(go.Contour(
            x=ppz_t, y=delta_t, z=rate_2d.T,
            contours=dict(
                type="constraint", operation="=", value=level,
                showlabels=True,
                labelfont=dict(size=11, color=contour_colors[level]),
            ),
            line=dict(color=contour_colors[level], width=contour_widths[level]),
            showscale=False,
            showlegend=False,
            hoverinfo="skip",
        ))

    # optimal point marker
    if oi is not None:
        fig2d.add_trace(go.Scatter(
            x=[opt_t], y=[opt_dt], mode="markers",
            marker=dict(symbol="star", size=16, color="red", line=dict(color="white", width=1)),
            showlegend=False,
            hovertemplate=f"<b>Optimum</b><br>t={opt_t:.0f} mm<br>t'={opt_dt:.0f} mm<br>Efficiency: {opt_eff:.3f}<br>Rate: {opt_rate:.1f} kHz<extra></extra>",
        ))

    fig2d.update_layout(
        title=f"{TRG_LABELS.get(trg_sel)}   |   {ALG_LABELS.get(alg_sel, alg_sel)}   |   ERatio: {ERATIO_LABELS.get(eratio_sel)}   |   Min ET: {et_sel} GeV",
        xaxis_title="PPZ threshold t [mm]",
        yaxis_title="ΔPPZ threshold t' [mm]",
        height=600,
        showlegend=False,
    )
    st.plotly_chart(fig2d, use_container_width=True)
    if oi is not None:
        st.info(f"★ Optimal point within {highlight_contour} kHz budget: t = {opt_t:.0f} mm, t' = {opt_dt:.0f} mm → efficiency = {opt_eff:.3f}, rate = {opt_rate:.1f} kHz")

else:
    fig   = go.Figure()
    r     = results[eratio_sel][et_sel][alg_sel]
    title = f"{ALG_LABELS.get(alg_sel, alg_sel)}   |   ERatio: {ERATIO_LABELS.get(eratio_sel)}   |   Min ET: {et_sel} GeV"
    xaxis = dict(); yaxis = dict()

    if plot_type == "Efficiency vs Rate":
        thresholds = r[trg_options[0]]["thresholds"]
        for trg in trg_sel_multi:
            rate = r[trg]["rate"] / 1e3
            eff  = r[trg]["efficiency"][sig_sel]
            fig.add_trace(go.Scatter(
                x=rate, y=eff, mode="lines",
                name=TRG_LABELS.get(trg, trg),
                line=dict(color=TRG_COLORS.get(trg), width=2),
                customdata=thresholds,
                hovertemplate="Rate: %{x:.1f} kHz<br>Efficiency: %{y:.3f}<br>Threshold: %{customdata:.0f} mm<extra></extra>",
            ))
        if rate_budget is not None:
            fig.add_vline(x=rate_budget,        line=dict(color="red",    dash="dash", width=1.5), annotation_text=f"{rate_budget:.0f} kHz")
        if rate_budget_single is not None:
            fig.add_vline(x=rate_budget_single, line=dict(color="orange", dash="dash", width=1.5), annotation_text=f"{rate_budget_single:.0f} kHz")
        if show_baselines:
            eratio_label = ERATIO_LABELS.get(eratio_sel, str(eratio_sel))
            baselines = [
                (eratio_sel, et_sel, "single_egamma", f"{eratio_label}, ≥1 TOB",  "rgba(160,160,160,0.8)", 0.97),
                (eratio_sel, et_sel, "di_egamma",     f"{eratio_label}, ≥2 TOBs", "rgba(160,160,160,0.8)", 0.89),
            ]
            for eratio_bl, et_bl, trg_bl, label_bl, color_bl, ann_y in baselines:
                try:
                    ref = results[eratio_bl][et_bl][alg_sel][trg_bl]["rate"][-1] / 1e3
                    fig.add_vline(
                        x=ref,
                        line=dict(color=color_bl, dash="dot", width=1.5),
                        annotation=dict(
                            text=f"{label_bl} ({ref:.0f} kHz)",
                            font=dict(size=10, color="rgba(160,160,160,0.9)"),
                            y=ann_y, yref="paper",
                            yanchor="middle",
                            xanchor="left",
                        ),
                    )
                except KeyError:
                    pass
        xaxis = dict(title="Trigger rate [kHz]", type="log" if log_x else "linear")
        yaxis = dict(title="Signal efficiency", range=[0, 1])
        title = f"{sig_sel}   |   {title}"

    elif plot_type == "Rate by JZ Slice":
        thresholds = r[trg_sel]["thresholds"]
        slices = {jz: r[trg_sel]["rate_per_slice"][jz] / 1e3 for jz in jz_options}
        if stack:
            for jz in jz_options:
                fig.add_trace(go.Scatter(
                    x=thresholds, y=slices[jz], mode="lines", name=jz,
                    line=dict(color=JZ_COLORS.get(jz), width=2),
                    stackgroup="one", fillcolor=JZ_COLORS.get(jz),
                    hovertemplate="Threshold: %{x:.0f} mm<br>Rate: %{y:.2f} kHz<extra></extra>",
                ))
        else:
            for jz in jz_options:
                fig.add_trace(go.Scatter(
                    x=thresholds, y=slices[jz], mode="lines", name=jz,
                    line=dict(color=JZ_COLORS.get(jz), width=2),
                    hovertemplate="Threshold: %{x:.0f} mm<br>Rate: %{y:.2f} kHz<extra></extra>",
                ))
            fig.add_trace(go.Scatter(
                x=thresholds, y=r[trg_sel]["rate"] / 1e3, mode="lines", name="Total",
                line=dict(color="black", dash="dash", width=1.5),
                hovertemplate="Threshold: %{x:.0f} mm<br>Total rate: %{y:.2f} kHz<extra></extra>",
            ))
        xaxis = dict(title="PPZ threshold [mm]")
        yaxis = dict(title="Trigger rate [kHz]", type="log" if log_y else "linear")
        title = f"{TRG_LABELS.get(trg_sel)}   |   {title}"

    elif plot_type == "Turn-on Curves":
        if not available_rate_points:
            st.warning("No turn-on curve data in this pkl. Re-run trigger_rate.py to generate it.")
            st.stop()
        colors_viridis = [
            f"rgb({int(rv*255)},{int(gv*255)},{int(bv*255)})"
            for rv, gv, bv, _ in plt.cm.viridis(np.linspace(0.1, 0.9, len(rate_points_sel)))
        ]
        for color, rp in zip(colors_viridis, rate_points_sel):
            eff = eff_vs_pt_data[rp]
            n   = len(eff)
            pt_centers = 0.5 * (PT_BINS[:n] + PT_BINS[1:n+1])
            fig.add_trace(go.Scatter(
                x=pt_centers, y=eff, mode="lines+markers",
                name=f"{rp/1e3:.0f} kHz",
                line=dict(color=color, width=2),
                hovertemplate="pT: %{x:.0f} GeV<br>Efficiency: %{y:.3f}<extra></extra>",
            ))
        xaxis = dict(title="Subleading truth photon pT [GeV]", range=[PT_BINS[0], PT_BINS[n]])
        yaxis = dict(title="Signal efficiency", range=[0, 1])
        title = f"{sig_sel}   |   {TRG_LABELS.get(trg_sel)}   |   {title}"

    elif plot_type == "ROC Curve":
        for trg in trg_sel_multi:
            rate = r[trg]["rate"]
            eff  = r[trg]["efficiency"][sig_sel]
            bkg_eff = rate / TOTAL_INPUT_RATE
            if x_mode == "Background rejection (1−ε)":
                x_vals = 1.0 - bkg_eff
                xlabel = "Background rejection (1−ε_bkg)"
                hover_x = "Bkg rejection: %{x:.4f}"
            elif x_mode == "Rejection factor (1/ε)":
                x_vals = np.where(bkg_eff > 0, 1.0 / bkg_eff, np.nan)
                xlabel = "Rejection factor (1/ε_bkg)"
                hover_x = "Rejection factor: %{x:.0f}"
            else:
                x_vals = bkg_eff
                xlabel = "Background efficiency"
                hover_x = "Bkg eff: %{x:.4f}"
            auc = np.trapz(eff, bkg_eff)
            name = TRG_LABELS.get(trg, trg)
            if show_auc:
                name += f"  (AUC={auc:.4f})"
            fig.add_trace(go.Scatter(
                x=x_vals, y=eff, mode="lines",
                name=name,
                line=dict(color=TRG_COLORS.get(trg), width=2),
                customdata=np.stack([rate / 1e3, r[trg]["thresholds"], bkg_eff], axis=1),
                hovertemplate=hover_x + "<br>Sig eff: %{y:.3f}<br>Rate: %{customdata[0]:.1f} kHz<br>Threshold: %{customdata[1]:.0f} mm<extra></extra>",
            ))
        xaxis = dict(title=xlabel, type="log" if log_x_roc else "linear")
        yaxis = dict(title="Signal efficiency", range=[0, 1])
        title = f"{sig_sel}   |   {title}"

    elif plot_type == "Rate vs Threshold":
        thresholds = r[trg_options[0]]["thresholds"]
        for trg in trg_sel_multi:
            fig.add_trace(go.Scatter(
                x=thresholds, y=r[trg]["rate"] / 1e3, mode="lines",
                name=TRG_LABELS.get(trg, trg),
                line=dict(color=TRG_COLORS.get(trg), width=2),
                hovertemplate="Threshold: %{x:.0f} mm<br>Rate: %{y:.2f} kHz<extra></extra>",
            ))
        if show_baselines:
            eratio_label = ERATIO_LABELS.get(eratio_sel, str(eratio_sel))
            baselines = [
                (eratio_sel, et_sel, "single_egamma", f"{eratio_label}, ≥1 TOB",  "rgba(160,160,160,0.8)", 0.97),
                (eratio_sel, et_sel, "di_egamma",     f"{eratio_label}, ≥2 TOBs", "rgba(160,160,160,0.8)", 0.89),
            ]
            for eratio_bl, et_bl, trg_bl, label_bl, color_bl, ann_x in baselines:
                try:
                    ref = results[eratio_bl][et_bl][alg_sel][trg_bl]["rate"][-1] / 1e3
                    fig.add_hline(
                        y=ref,
                        line=dict(color=color_bl, dash="dot", width=1.5),
                        annotation_text=f"{label_bl} ({ref:.0f} kHz)",
                        annotation_font_size=10,
                        annotation_position="right",
                    )
                except KeyError:
                    pass
        xaxis = dict(title="PPZ threshold [mm]")
        yaxis = dict(title="Trigger rate [kHz]", type="log" if log_y else "linear")

    elif plot_type == "Explorer":
        def resolve(split_val):
            eratio = split_val if split_by == "ERatio"       else exp_eratio
            et     = split_val if split_by == "Min ET [GeV]" else exp_et
            alg    = split_val if split_by == "Algorithm"    else exp_alg
            trg    = split_val if split_by == "Trigger"      else exp_trg
            sig    = split_val if split_by == "Signal"       else exp_sig
            return eratio, et, alg, trg, sig

        def label(split_val):
            if split_by == "Trigger":      return TRG_LABELS.get(split_val, split_val)
            if split_by == "Signal":       return split_val
            if split_by == "Algorithm":    return ALG_LABELS.get(split_val, split_val)
            if split_by == "ERatio":       return ERATIO_LABELS.get(split_val, str(split_val))
            if split_by == "Min ET [GeV]": return f"{split_val} GeV"

        for val in split_vals:
            eratio, et, alg, trg, sig = resolve(val)
            try:
                x = get_array(results, eratio, et, alg, trg, sig, x_qty)
                y = get_array(results, eratio, et, alg, trg, sig, y_qty)
                color = (TRG_COLORS.get(val) if split_by == "Trigger"
                         else SIG_COLORS.get(val) if split_by == "Signal"
                         else None)
                fig.add_trace(go.Scatter(
                    x=x, y=y, mode="lines", name=label(val),
                    line=dict(color=color, width=2),
                    hovertemplate=f"{x_qty}: %{{x:.2f}}<br>{y_qty}: %{{y:.3f}}<extra></extra>",
                ))
            except Exception as e:
                st.warning(f"Could not plot {label(val)}: {e}")
        xaxis = dict(title=x_qty, type="log" if log_x_exp else "linear")
        yaxis = dict(title=y_qty, type="log" if log_y_exp else "linear")

    fig.update_layout(
        title=title,
        xaxis=xaxis,
        yaxis=yaxis,
        hovermode="x unified",
        height=520,
        legend=dict(bgcolor="rgba(255,255,255,0.8)"),
    )
    st.plotly_chart(fig, use_container_width=True)
