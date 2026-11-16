from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from rdkit import Chem
from rdkit.Chem import Draw
from scipy.optimize import least_squares


DATA_FILE = Path(__file__).with_name("imine_condensation_dataset_for_students.csv")
SIGMA_PARA = {"NH2": -0.66, "OMe": -0.27, "CH3": -0.17,
              "H": 0.00, "Cl": 0.23, "CF3": 0.54, "NO2": 0.78}
# Literature sigma_para for substituents NOT in the dataset (prediction only)
EXTRA_SIGMA = {"NMe2": -0.83, "OH": -0.37, "F": 0.06, "Br": 0.23, "COMe": 0.50, "CN": 0.66}
R_ORDER = list(SIGMA_PARA)
AMINE_ORDER = ["n-butylamine", "sec-butylamine", "tert-butylamine"]
SOLVENTS = ["DCM", "MTBE"]
REQUIRED = {"aldehyde_name", "aldehyde_R_group", "aldehyde_SMILES",
            "amine_name", "amine_SMILES", "imine_SMILES", "solvent",
            "temperature_C", "time_min", "conversion_percent"}
PAIR_KEYS = ["aldehyde_R_group", "amine_name", "solvent"]
ANCHOR_T, ANCHOR_TIME = 25, 60
R_GAS = 8.314
ORDERS = {"aldehyde_R_group": R_ORDER, "amine_name": AMINE_ORDER, "solvent": SOLVENTS}
LABELS = {
    "time_min": "Time (min)", "temperature_C": "Temperature (°C)",
    "conversion_percent": "Conversion (%)", "aldehyde_R_group": "R group",
    "amine_name": "Amine", "solvent": "Solvent", "sigma_para": "Hammett σₚ",
    "anchor": "Conversion at 25 °C, 60 min (%)",
}

st.set_page_config(page_title="Imine condensation explorer", page_icon="🧪", layout="wide")


# ============================== Data ==============================
@st.cache_data
def load_data(path):
    data = pd.read_csv(path)
    missing = REQUIRED - set(data.columns)
    if missing:
        raise ValueError("Missing CSV columns: " + ", ".join(sorted(missing)))
    for column in data.select_dtypes(include=["object", "string"]).columns:
        data[column] = data[column].str.strip()
    for column in ("temperature_C", "time_min", "conversion_percent"):
        data[column] = pd.to_numeric(data[column], errors="coerce")
    if data[list(REQUIRED)].isna().any().any():
        raise ValueError("The CSV has missing or invalid values in required columns.")
    if not data["conversion_percent"].between(0, 100).all():
        raise ValueError("Conversion percentages must be between 0 and 100.")
    unknown = set(data["aldehyde_R_group"]) - set(SIGMA_PARA)
    if unknown:
        raise ValueError("No sigma_para value for: " + ", ".join(sorted(unknown)))
    data["sigma_para"] = data["aldehyde_R_group"].map(SIGMA_PARA)
    data["is_anchor"] = (data["temperature_C"] == ANCHOR_T) & (data["time_min"] == ANCHOR_TIME)
    anchors = data.loc[data["is_anchor"], PAIR_KEYS + ["conversion_percent"]].rename(
        columns={"conversion_percent": "anchor"})
    data = data.merge(anchors, on=PAIR_KEYS, how="left")
    # Values reported with at most one decimal place (measured data are reported this way)
    data["one_decimal"] = np.isclose(data["conversion_percent"], data["conversion_percent"].round(1))
    return data


@st.cache_data
def mol_image(smiles, size=(320, 220)):
    molecule = Chem.MolFromSmiles(smiles)
    return None if molecule is None else Draw.MolToImage(molecule, size=size)


def k_eff(conversion, t=ANCHOR_TIME):
    """Effective first-order rate constant (min⁻¹) from one conversion value."""
    return -np.log(1 - np.clip(conversion, 0, 99.9) / 100) / t  # keeps Series as Series


def conv_from_k(k, t=ANCHOR_TIME):
    return 100 * (1 - np.exp(-k * t))


@st.cache_data
def provenance_checks(data):
    """Evidence that only the 25 °C / 60 min values are independent measurements."""
    decimals = data.pivot_table(index="temperature_C", columns="time_min",
                                values="one_decimal", aggfunc="mean") * 100
    spread = (data[~data["is_anchor"]]
              .groupby(["anchor", "temperature_C", "time_min"])["conversion_percent"].std()
              .fillna(0))
    curves = data.pivot_table(index=PAIR_KEYS + ["temperature_C"], columns="time_min",
                              values="conversion_percent").round(2)
    return decimals, float(spread.max()), int(curves.drop_duplicates().shape[0]), len(curves)


@st.cache_data
def kinetic_form_comparison(data):
    """Fit three rate laws to each of the 126 time courses; report mean absolute error."""
    curves = data.pivot_table(index=PAIR_KEYS + ["temperature_C"], columns="time_min",
                              values="conversion_percent")
    t = curves.columns.to_numpy(float)
    forms = {
        "First order: 100(1 − e^(−kt))": (lambda p: 100 * (1 - np.exp(-np.exp(p[0]) * t)), [np.log(0.01)]),
        "Second order: 100·kt / (1 + kt)": (lambda p: 100 * np.exp(p[0]) * t / (1 + np.exp(p[0]) * t), [np.log(0.01)]),
        "First order to a plateau: P(1 − e^(−kt))": (lambda p: p[1] * (1 - np.exp(-np.exp(p[0]) * t)), [np.log(0.01), 100]),
    }
    rows = []
    for name, (model, p0) in forms.items():
        errors = [np.abs(least_squares(lambda p: model(p) - y, p0).fun).mean() for y in curves.to_numpy()]
        rows.append({"Rate law": name, "Parameters": len(p0),
                     "Mean error (pp)": np.mean(errors), "Worst curve (pp)": np.max(errors)})
    return pd.DataFrame(rows)


def interpolate_curves(data, anchor_value):
    """Map a 25 °C / 60 min conversion onto every (T, t) using the dataset's own anchor→curve relation."""
    rows = []
    for (temp, time), g in data.groupby(["temperature_C", "time_min"]):
        g = g.groupby("anchor")["conversion_percent"].mean().sort_index()
        rows.append({"temperature_C": temp, "time_min": time,
                     "conversion_percent": float(np.interp(anchor_value, g.index, g.to_numpy()))})
    return pd.DataFrame(rows)


# ============================== Model ==============================
MODEL_SPECS = {
    "A · Amine + solvent only": dict(sigma=False, sigma2=False, inter=False),
    "B · + σₚ (linear Hammett)": dict(sigma=True, sigma2=False, inter=False),
    "C · + σₚ + σₚ² (curved Hammett)": dict(sigma=True, sigma2=True, inter=False),
    "D · C + σₚ×amine + σₚ×solvent": dict(sigma=True, sigma2=True, inter=True),
}


def design_matrix(frame, spec):
    mtbe = (frame["solvent"] == "MTBE").to_numpy(float)
    s = frame["sigma_para"].to_numpy(float)
    cols = {"intercept": np.ones(len(frame)), "MTBE": mtbe}
    for amine in AMINE_ORDER[1:]:
        d = (frame["amine_name"] == amine).to_numpy(float)
        cols[amine] = d
        cols[f"{amine} × MTBE"] = d * mtbe
        if spec["inter"]:
            cols[f"σₚ × {amine}"] = d * s
    if spec["sigma"]:
        cols["σₚ"] = s
    if spec["sigma2"]:
        cols["σₚ²"] = s ** 2
    if spec["inter"]:
        cols["σₚ × MTBE"] = s * mtbe
    return pd.DataFrame(cols)


def fit_model(train, spec):
    """ln k = X·β, fitted in conversion space at 25 °C / 60 min."""
    X = design_matrix(train, spec)
    y = train["conversion_percent"].to_numpy(float)
    beta0 = np.zeros(X.shape[1])
    beta0[0] = np.log(0.02)
    result = least_squares(lambda b: conv_from_k(np.exp(np.clip(X.to_numpy() @ b, -30, 5))) - y,
                           beta0, x_scale="jac")
    return pd.Series(result.x, index=X.columns)


def predict_conversion(beta, frame, spec):
    lnk = design_matrix(frame, spec).to_numpy() @ beta.to_numpy()
    return conv_from_k(np.exp(np.clip(lnk, -30, 5)))


@st.cache_data
def compare_models(anchors):
    """Leave-one-aldehyde-out CV for every candidate model on the 42 measured points."""
    summary, predictions, betas = [], {}, {}
    for name, spec in MODEL_SPECS.items():
        beta = fit_model(anchors, spec)
        betas[name] = beta
        folds = []
        for r_group in R_ORDER:
            test = anchors[anchors["aldehyde_R_group"] == r_group].copy()
            fold_beta = fit_model(anchors[anchors["aldehyde_R_group"] != r_group], spec)
            test["predicted"] = predict_conversion(fold_beta, test, spec)
            folds.append(test)
        cv = pd.concat(folds, ignore_index=True)
        cv["abs_error"] = (cv["predicted"] - cv["conversion_percent"]).abs()
        predictions[name] = cv
        train_mae = np.abs(predict_conversion(beta, anchors, spec) - anchors["conversion_percent"]).mean()
        summary.append({"Model": name, "Parameters": len(beta),
                        "Training MAE (pp)": train_mae, "Held-out MAE (pp)": cv["abs_error"].mean()})
    return pd.DataFrame(summary), predictions, betas


# ============================== Load ==============================
try:
    df = load_data(DATA_FILE)
except (FileNotFoundError, ValueError, pd.errors.ParserError) as exc:
    st.error(f"Could not load {DATA_FILE.name}: {exc}")
    st.info("Save the CSV beside app.py, then reload the page.")
    st.stop()

anchors = df[df["is_anchor"]].copy()
anchors["k"] = k_eff(anchors["conversion_percent"])
decimals, max_spread, n_unique_curves, n_curves = provenance_checks(df)

# ---------- Shared results (used by several pages) ----------
paired = anchors.pivot_table(index=["aldehyde_R_group", "amine_name"], columns="solvent",
                             values="conversion_percent").reset_index()
diff = paired["MTBE"] - paired["DCM"]
ratio = k_eff(paired["MTBE"]) / k_eff(paired["DCM"])
tf = df[df["time_min"] == ANCHOR_TIME].pivot_table(index=PAIR_KEYS, columns="temperature_C",
                                                   values="conversion_percent")
tf = tf[(tf[15] > 2) & (tf[35] < 98)]
tf_ratio = k_eff(tf[35]) / k_eff(tf[15])
ea_implied = R_GAS * np.log(tf_ratio) / (1 / 288.15 - 1 / 308.15) / 1000
summary, cv_preds, betas = compare_models(anchors)
best_name = summary.loc[summary["Held-out MAE (pp)"].idxmin(), "Model"]

PAGES = {
    "🔍 Data check": "What is really measured?",
    "🗺️ Reactivity map": "The measured results",
    "🔬 Effects": "Mechanism, sterics, electronics, solvent",
    "⏱️ Time & temperature": "Derived curves",
    "🔮 Predict": "Model + validation",
    "🧭 Explore freely": "Build your own chart",
    "📋 Design rules & data": "Summary + downloads",
}

with st.sidebar:
    st.header("🧪 Imine explorer")
    page = st.radio("Go to", list(PAGES), key="page",
                    captions=list(PAGES.values()), label_visibility="collapsed")
    st.divider()
    st.markdown(
        "**About.** Imine condensation of 7 para-substituted benzaldehydes with 3 butylamines in DCM or MTBE "
        "(J. Chem. Educ. 2023, 100, 4456).\n\n"
        "**Key finding about the data:** only the 42 values at **25 °C / 60 min** behave like independent "
        "measurements. Every other value is a fixed function of that one number (see *Data check*). "
        "Charts mark derived data clearly."
    )
    with st.expander("📖 Glossary (plain language)"):
        st.markdown(
            "- **Imine condensation**: aldehyde + amine → imine (C=N bond) + water.\n"
            "- **Conversion (%)**: share of the aldehyde that has turned into imine.\n"
            "- **pp (percentage points)**: difference between two percentages, e.g. 60% → 70% is +10 pp.\n"
            "- **R group / para substituent**: the group opposite the CHO on the benzene ring.\n"
            "- **Electron-donating / -withdrawing**: pushes electron density into the ring (NH₂, OMe) "
            "or pulls it out (Cl, CF₃, NO₂).\n"
            "- **Hammett σₚ**: a number for how strongly a para group withdraws electrons. "
            "Negative = donor, 0 = H, positive = acceptor.\n"
            "- **Hammett plot / ρ**: log of relative rate vs σₚ. The slope ρ says how sensitive the reaction "
            "is to electronics; ρ > 0 means acceptors speed it up.\n"
            "- **Steric hindrance**: bulky groups physically blocking the reacting atom "
            "(n-butyl < sec-butyl < tert-butyl).\n"
            "- **Rate constant k**: how fast a reaction goes; here estimated from one conversion value, k = −ln(1 − X)/t.\n"
            "- **Activation energy Ea**: how strongly the rate increases with temperature.\n"
            "- **Measured vs derived**: measured = the 42 values at 25 °C / 60 min; derived = all other rows, "
            "calculated from those 42.\n"
            "- **DCM / MTBE**: dichloromethane (chlorinated, more toxic) / methyl tert-butyl ether (greener alternative).\n"
            "- **Held-out (cross-validation) error**: how wrong the model is on an aldehyde it was not trained on, "
            "the honest estimate of prediction quality."
        )

st.title("🧪 Imine condensation explorer")
st.caption("Aldehyde + amine → imine + H₂O. Goal: find design rules for reactivity and predict untested reactions.")

st.header(page)
page_names = list(PAGES)

MECHANISM_SVG = """
<div style="overflow-x:auto">
<svg viewBox="0 0 940 300" width="100%" style="max-width:940px;min-width:640px;font-family:sans-serif;color:inherit">
  <defs>
    <marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">
      <path d="M0,0 L10,5 L0,10 z" fill="currentColor"/>
    </marker>
  </defs>
  <g fill="none" stroke="currentColor" stroke-width="1.5">
    <rect x="10" y="40" width="200" height="70" rx="10"/>
    <rect x="370" y="40" width="200" height="70" rx="10"/>
    <rect x="730" y="40" width="200" height="70" rx="10"/>
    <path d="M215 65 L365 65" marker-end="url(#arr)"/>
    <path d="M365 85 L215 85" marker-end="url(#arr)" stroke-dasharray="4 3"/>
    <path d="M575 75 L725 75" marker-end="url(#arr)"/>
  </g>
  <g fill="currentColor" font-size="15" text-anchor="middle">
    <text x="110" y="70" font-weight="bold">Ar–CHO + R′–NH₂</text>
    <text x="110" y="93" font-size="13">aldehyde + amine</text>
    <text x="470" y="70" font-weight="bold">Ar–CH(OH)–NH–R′</text>
    <text x="470" y="93" font-size="13">hemiaminal (intermediate)</text>
    <text x="830" y="70" font-weight="bold">Ar–CH=N–R′</text>
    <text x="830" y="93" font-size="13">imine (+ H₂O, removed by MgSO₄)</text>
    <text x="290" y="30" font-size="13" font-weight="bold" fill="#2f6fde">Step 1 · amine adds to C=O</text>
    <text x="650" y="30" font-size="13" font-weight="bold" fill="#d9822b">Step 2 · loses water</text>
  </g>
  <g fill="currentColor" font-size="13">
    <text x="140" y="150" font-weight="bold" fill="#2f6fde">Step 1 is helped by…</text>
    <text x="140" y="172">• an electron-poor C=O carbon: acceptors (Cl, CF₃, NO₂) speed it up</text>
    <text x="140" y="192">• an unhindered amine: n-butyl ≫ sec-butyl ≫ tert-butyl</text>
    <text x="440" y="230" font-weight="bold" fill="#d9822b">Step 2 is helped by…</text>
    <text x="440" y="252">• an electron-rich ring that stabilises the C=N⁺ character</text>
    <text x="440" y="272">  as water leaves: donors help, acceptors slow it down</text>
  </g>
  <g fill="currentColor" font-size="12.5" font-style="italic">
    <text x="10" y="232">Opposite demands ⇒ for strong acceptors step 2</text>
    <text x="10" y="250">becomes rate-limiting, so the rate gain levels off:</text>
    <text x="10" y="268">the Hammett plot bends downward (concave).</text>
  </g>
</svg>
</div>
"""

# ============================== 1. Data check ==============================
if page == page_names[0]:
    st.subheader("How much independent information does the dataset contain?")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{len(df)}", help="7 aldehydes × 3 amines × 2 solvents × 3 temperatures × 4 times")
    c2.metric("Reaction conditions", f"{n_curves}", help="Aldehyde × amine × solvent × temperature")
    c3.metric("Distinct time courses", f"{n_unique_curves}",
              help="Several different conditions have exactly the same 4-point curve.")
    c4.metric("Independent measurements", f"{len(anchors)}", help="Values at 25 °C / 60 min")

    left, right = st.columns([2, 3])
    with left:
        st.markdown("**Evidence 1 · Number format**")
        dec_fig = px.imshow(decimals, text_auto=".0f", color_continuous_scale="Blues", zmin=0, zmax=100,
                            labels={"x": "Time (min)", "y": "Temperature (°C)", "color": "% of values"},
                            aspect="auto")
        dec_fig.update_yaxes(type="category")
        dec_fig.update_xaxes(type="category")
        st.plotly_chart(dec_fig, width="stretch")
        st.caption("Share of values reported with ≤ 1 decimal place. Only 25 °C / 60 min is 100% "
                   "(e.g. 12, 65, 0.5), like rounded experimental values. The rest look computed (e.g. 55.28).")
    with right:
        st.markdown("**Evidence 2 · Every curve is a function of one number**")
        collapse = df.copy()
        collapse["Temperature"] = collapse["temperature_C"].map(lambda x: f"{x:g} °C")
        col_fig = px.scatter(collapse, x="anchor", y="conversion_percent", color="Temperature",
                             facet_col="time_min", facet_col_wrap=2, opacity=0.8, height=560,
                             facet_col_spacing=0.08, facet_row_spacing=0.12,
                             category_orders={"time_min": sorted(df["time_min"].unique()),
                                              "Temperature": ["15 °C", "25 °C", "35 °C"]},
                             hover_data=PAIR_KEYS, labels={**LABELS, "anchor": "Value at 25 °C/60 min (%)"})
        col_fig.for_each_annotation(lambda a: a.update(text=f"t = {a.text.split('=')[-1]} min"))
        st.plotly_chart(col_fig, width="stretch")
        st.caption(f"Plotting each value against the same reaction's 25 °C / 60 min value collapses all 504 points "
                   f"onto 12 smooth curves. Reactions with the same 25 °C / 60 min value have identical curves at every "
                   f"temperature and time (largest spread: {max_spread:.2f} pp).")

    st.info(
        "**What this means for the analysis**\n\n"
        f"- The dataset has **{len(anchors)} independent measurements**, not 504. Means over all rows "
        "mostly repeat the same information 12 times.\n"
        "- Time and temperature trends come from the rule used to build the dataset, not from chemistry. "
        "Activation energies or rate laws fitted to them describe that rule, so this dashboard labels them *derived*.\n"
        "- The real chemical comparisons (aldehyde, amine, solvent) and the predictive model therefore use the 42 measured values."
    )

# ============================== 2. Reactivity map ==============================
if page == page_names[1]:
    st.subheader("Measured conversion: aldehyde × amine × solvent")
    m1, m2 = st.columns([3, 2])
    with m1:
        view = st.radio("Show", ["Measured (25 °C, 60 min)", "Derived condition"], horizontal=True, key="map_view")
    if view.startswith("Derived"):
        with m2:
            d1, d2 = st.columns(2)
            map_temp = d1.selectbox("Temperature (°C)", sorted(df["temperature_C"].unique()), key="map_temp")
            map_time = d2.selectbox("Time (min)", sorted(df["time_min"].unique()), index=2, key="map_time")
    else:
        map_temp, map_time = ANCHOR_T, ANCHOR_TIME
    shown = df[(df["temperature_C"] == map_temp) & (df["time_min"] == map_time)]
    heat_cols = st.columns(2)
    for column, solvent in zip(heat_cols, SOLVENTS):
        heat = (shown[shown["solvent"] == solvent]
                .pivot_table(index="aldehyde_R_group", columns="amine_name", values="conversion_percent")
                .reindex(index=R_ORDER, columns=AMINE_ORDER))
        heat.index = [f"{r} ({SIGMA_PARA[r]:+.2f})" for r in heat.index]
        fig = px.imshow(heat, text_auto=".0f", zmin=0, zmax=100, color_continuous_scale="Blues", aspect="auto",
                        labels={"x": "Amine", "y": "R group (σₚ)", "color": "Conv. (%)"})
        fig.update_layout(title=f"{solvent}", coloraxis_showscale=(solvent == "MTBE"))
        column.plotly_chart(fig, width="stretch")
    st.caption(f"{map_temp:g} °C, {map_time:g} min. Rows go from electron-donating (top) to electron-withdrawing (bottom); "
               "columns from least to most hindered amine." +
               ("" if view.startswith("Measured") else " ⚠️ Derived values, not independent measurements."))

    st.subheader("Draw a reaction")
    p1, p2, p3 = st.columns(3)
    rx_r = p1.selectbox("Aldehyde R group", R_ORDER, index=3, key="rx_r")
    rx_amine = p2.selectbox("Amine", AMINE_ORDER, key="rx_amine")
    rx_solv = p3.selectbox("Solvent", SOLVENTS, key="rx_solv")
    row = anchors[(anchors["aldehyde_R_group"] == rx_r) & (anchors["amine_name"] == rx_amine)
                  & (anchors["solvent"] == rx_solv)].iloc[0]
    s1, s2, s3, s4, s5 = st.columns([4, 1, 4, 1, 4])
    for column, title, smiles in [(s1, row["aldehyde_name"], row["aldehyde_SMILES"]),
                                  (s3, row["amine_name"], row["amine_SMILES"]),
                                  (s5, "Imine (+ H₂O)", row["imine_SMILES"])]:
        with column:
            st.markdown(f"**{title}**")
            image = mol_image(smiles)
            if image is None:
                st.warning(f"Cannot draw: {smiles}")
            else:
                st.image(image, width="stretch")
    s2.markdown("<h2 style='text-align:center;margin-top:90px'>+</h2>", unsafe_allow_html=True)
    s4.markdown(f"<h4 style='text-align:center;margin-top:80px'>→<br><small>{row['conversion_percent']:g}%</small></h4>",
                unsafe_allow_html=True)
    st.caption(f"Measured conversion after 60 min at 25 °C in {rx_solv}: **{row['conversion_percent']:g}%**.")

# ============================== 3. Effects ==============================
if page == page_names[2]:
    st.caption("All comparisons on this tab use the 42 measured values (25 °C, 60 min) and compare reactions that differ in only one factor.")

    st.subheader("0 · Mechanism: why each factor matters")
    st.html(MECHANISM_SVG)
    st.caption("Textbook two-step mechanism of imine formation. The plots below test each of these predictions against the data.")

    st.subheader("1 · Amine: steric hindrance")
    amine_fig = px.line(anchors, x="amine_name", y="conversion_percent", color="aldehyde_R_group",
                        facet_col="solvent", facet_col_spacing=0.08, markers=True,
                        category_orders=ORDERS, labels=LABELS)
    amine_fig.update_yaxes(range=[0, 105])
    amine_fig.update_xaxes(tickvals=AMINE_ORDER, ticktext=["n-Bu", "sec-Bu", "tert-Bu"])
    amine_fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    st.plotly_chart(amine_fig, width="stretch")
    drops = anchors.pivot_table(index=["aldehyde_R_group", "solvent"], columns="amine_name", values="conversion_percent")
    n_mono = int(((drops["n-butylamine"] >= drops["sec-butylamine"]) & (drops["sec-butylamine"] >= drops["tert-butylamine"])).sum())
    st.caption(f"Each line is one aldehyde in one solvent. Conversion falls from n- to sec- to tert-butyl in "
               f"{n_mono} of {len(drops)} series: more branching at the carbon next to N blocks attack on the carbonyl.")

    st.subheader("2 · Aldehyde: electronic effect (Hammett plot)")
    e1, e2, e3 = st.columns(3)
    h_amine = e1.selectbox("Amine", AMINE_ORDER, key="h_amine")
    h_solv = e2.selectbox("Solvent", SOLVENTS, key="h_solv")
    h_fit = e3.radio("Fit", ["Linear", "Quadratic"], index=1, horizontal=True, key="h_fit",
                     help="A downward-curved Hammett plot often signals a change in the rate-limiting step.")
    hx = anchors[(anchors["amine_name"] == h_amine) & (anchors["solvent"] == h_solv)].copy()
    k_h = hx.loc[hx["aldehyde_R_group"] == "H", "k"].iloc[0]
    hx["log_rel"] = np.log10(hx["k"] / k_h)
    deg = 1 if h_fit == "Linear" else 2
    coef = np.polyfit(hx["sigma_para"], hx["log_rel"], deg)
    fitted = np.polyval(coef, hx["sigma_para"])
    r2 = 1 - ((hx["log_rel"] - fitted) ** 2).sum() / ((hx["log_rel"] - hx["log_rel"].mean()) ** 2).sum()
    hm = px.scatter(hx, x="sigma_para", y="log_rel", text="aldehyde_R_group",
                    hover_data=["conversion_percent"],
                    labels={"sigma_para": "Hammett σₚ", "log_rel": "log(k_X / k_H)"})
    hm.update_traces(textposition="top center", marker=dict(size=11))
    xs = np.linspace(-0.7, 0.8, 60)
    hm.add_scatter(x=xs, y=np.polyval(coef, xs), mode="lines", line=dict(dash="dot"), name=f"{h_fit} fit")
    st.plotly_chart(hm, width="stretch")
    rho_lin = np.polyfit(hx["sigma_para"], hx["log_rel"], 1)[0]
    q1, q2, q3 = st.columns(3)
    q1.metric("ρ (linear slope)", f"{rho_lin:+.2f}")
    q2.metric(f"R² ({h_fit.lower()})", f"{r2:.2f}")
    if deg == 2:
        q3.metric("Curvature", "concave ∩" if coef[0] < 0 else "convex ∪", f"{coef[0]:+.2f}")
    st.caption("k is the effective first-order rate constant from the 60 min conversion, k = −ln(1 − X)/t. "
               "Positive ρ: electron-withdrawing groups make the carbonyl carbon more electrophilic and speed up amine attack. "
               "Very low conversions (tert-butylamine, < 2%) are reported only to 0.1–1%, so their k is imprecise.")

    st.subheader("3 · Solvent: can MTBE replace DCM?")
    sol_fig = px.scatter(paired, x="DCM", y="MTBE", color="amine_name", text="aldehyde_R_group",
                         category_orders=ORDERS, labels={"DCM": "Conversion in DCM (%)",
                                                         "MTBE": "Conversion in MTBE (%)", **LABELS})
    sol_fig.update_traces(textposition="top center", marker=dict(size=10))
    sol_fig.add_scatter(x=[0, 100], y=[0, 100], mode="lines", line=dict(dash="dash", color="grey"), name="equal")
    sol_fig.update_layout(xaxis_range=[-3, 103], yaxis_range=[-3, 103])
    st.plotly_chart(sol_fig, width="stretch")
    w1, w2, w3 = st.columns(3)
    w1.metric("MTBE ≥ DCM", f"{(diff >= 0).sum()} of {len(diff)} pairs")
    w2.metric("Median difference", f"{diff.median():+.0f} pp")
    w3.metric("Median rate ratio k(MTBE)/k(DCM)", f"{ratio.median():.2f}×")
    st.caption("Points above the dashed line react faster in MTBE. The solvent effect is not uniform: "
               "it depends on the aldehyde–amine pair.")

# ============================== 4. Time & temperature ==============================
if page == page_names[3]:
    st.warning("Everything on this tab uses **derived** values (see *Data check*). It shows the shape of the curves the "
               "dataset assumes; it is not independent kinetic evidence.")
    t1, t2, t3 = st.columns(3)
    tc_r = t1.selectbox("Aldehyde R group", R_ORDER, index=3, key="tc_r")
    tc_amine = t2.selectbox("Amine", AMINE_ORDER, key="tc_amine")
    tc_solv = t3.selectbox("Solvent", SOLVENTS, key="tc_solv")
    tc = df[(df["aldehyde_R_group"] == tc_r) & (df["amine_name"] == tc_amine) & (df["solvent"] == tc_solv)].copy()
    tc["Temperature"] = tc["temperature_C"].map(lambda x: f"{x:g} °C")
    tc_fig = px.line(tc.sort_values("time_min"), x="time_min", y="conversion_percent", color="Temperature",
                     markers=True, labels=LABELS)
    tc_fig.add_scatter(x=[ANCHOR_TIME], y=tc.loc[tc["is_anchor"], "conversion_percent"], mode="markers",
                       marker=dict(size=16, symbol="star", color="gold", line=dict(width=1, color="black")),
                       name="measured value")
    tc_fig.update_yaxes(range=[0, 105])
    st.plotly_chart(tc_fig, width="stretch")
    st.caption("The ★ is the only measured point; the rest of the curves are generated from it.")

    st.subheader("Which rate law describes the curves?")
    st.dataframe(kinetic_form_comparison(df).round(2), width="stretch", hide_index=True)
    st.caption("A simple first-order law does not fit as well as a law whose rate slows down over time "
               "(second order, or first order approaching an incomplete plateau). Fitting a single k per curve is therefore an approximation.")

    st.subheader("Temperature factor implied by the dataset")
    z1, z2 = st.columns(2)
    z1.metric("Median k(35 °C) / k(15 °C)", f"{tf_ratio.median():.2f}×")
    z2.metric("Implied apparent Ea", f"{ea_implied.median():.0f} kJ/mol",
              help=f"Interquartile range {ea_implied.quantile(.25):.0f}–{ea_implied.quantile(.75):.0f} kJ/mol over {len(tf)} conditions")
    st.caption("Higher temperature speeds up every reaction. The Ea describes the rule used to build the dataset, "
               "so it should be confirmed by measuring at other temperatures before it is used.")

# ============================== 5. Predict ==============================
if page == page_names[4]:
    st.markdown(
        "A **physics-informed model** for the 42 measured values: "
        "**ln k = amine effect + solvent effect + amine×solvent + f(σₚ)**, with conversion = 100(1 − e^(−k·60)). "
        "Four versions of f(σₚ) are compared by **leave-one-aldehyde-out** cross-validation: the model is refitted "
        "without one aldehyde and asked to predict it, which tests the real use case of a substituent never run before."
    )
    st.dataframe(summary.round(1).style.apply(
        lambda r: ["font-weight: bold" if r["Model"] == best_name else "" for _ in r], axis=1),
        width="stretch", hide_index=True)
    st.caption(f"Best held-out error: **{best_name}**. Training error always falls as parameters are added; "
               "held-out error shows which extra terms actually help prediction.")

    model_name = st.selectbox("Model used below", list(MODEL_SPECS), index=list(MODEL_SPECS).index(best_name), key="pr_model")
    spec, beta, cv = MODEL_SPECS[model_name], betas[model_name], cv_preds[model_name]

    v1, v2 = st.columns([3, 2])
    with v1:
        parity = px.scatter(cv, x="conversion_percent", y="predicted", color="aldehyde_R_group",
                            symbol="amine_name", category_orders=ORDERS,
                            hover_data=["solvent"], labels={"conversion_percent": "Measured (%)",
                                                            "predicted": "Predicted, held out (%)", **LABELS})
        parity.add_scatter(x=[0, 100], y=[0, 100], mode="lines", line=dict(dash="dash", color="grey"), name="perfect")
        st.plotly_chart(parity, width="stretch")
    with v2:
        per_r = cv.groupby("aldehyde_R_group")["abs_error"].mean().reindex(R_ORDER).rename("Held-out MAE (pp)")
        st.dataframe(per_r.round(1).reset_index(), width="stretch", hide_index=True)
        st.caption("Aldehydes with large errors do not follow the σₚ trend set by the others.")

    st.subheader("Predict a new reaction")
    n1, n2, n3 = st.columns(3)
    sub_options = R_ORDER + [f"{k} (not in dataset)" for k in EXTRA_SIGMA] + ["Custom σₚ"]
    with n1:
        sub = st.selectbox("Aldehyde para substituent", sub_options, index=sub_options.index("CN (not in dataset)"), key="pr_sub")
        if sub == "Custom σₚ":
            sigma_val = st.number_input("σₚ", -1.0, 1.0, 0.0, 0.01, key="pr_sigma")
        else:
            sub_name = sub.split(" ")[0]
            sigma_val = SIGMA_PARA.get(sub_name, EXTRA_SIGMA.get(sub_name))
            st.caption(f"σₚ = {sigma_val:+.2f}")
    pr_amine = n2.selectbox("Amine", AMINE_ORDER, key="pr_amine")
    pr_solv = n3.selectbox("Solvent", SOLVENTS, key="pr_solv")
    if not min(SIGMA_PARA.values()) <= sigma_val <= max(SIGMA_PARA.values()):
        st.warning("σₚ is outside the measured range (−0.66 to +0.78): the prediction is an extrapolation.")
    query = pd.DataFrame({"sigma_para": [sigma_val], "amine_name": [pr_amine], "solvent": [pr_solv]})
    pred60 = float(predict_conversion(beta, query, spec)[0])
    measured = anchors[(anchors["aldehyde_R_group"] == sub.split(" ")[0]) & (anchors["amine_name"] == pr_amine)
                       & (anchors["solvent"] == pr_solv)]
    o1, o2, o3 = st.columns(3)
    o1.metric("Predicted conversion (25 °C, 60 min)", f"{pred60:.0f}%")
    o2.metric("Predicted k", f"{k_eff(pred60):.4f} min⁻¹")
    if not measured.empty:
        held = cv[(cv["aldehyde_R_group"] == sub) & (cv["amine_name"] == pr_amine) & (cv["solvent"] == pr_solv)]
        o3.metric("Measured", f"{measured['conversion_percent'].iloc[0]:g}%",
                  help=f"Held-out prediction for this point: {held['predicted'].iloc[0]:.0f}%")

    curves = interpolate_curves(df, pred60)
    curves["Temperature"] = curves["temperature_C"].map(lambda x: f"{x:g} °C")
    pc = px.line(curves, x="time_min", y="conversion_percent", color="Temperature", markers=True, labels=LABELS)
    pc.update_yaxes(range=[0, 105])
    st.plotly_chart(pc, width="stretch")
    st.caption("Curves at other times and temperatures are read off the dataset's own anchor → curve relation "
               "(Data check, evidence 2), so they inherit its assumptions.")

    with st.expander("Model coefficients"):
        st.dataframe(beta.rename("β (ln k units)").round(3).to_frame(), width="stretch")
        if "σₚ" in beta:
            st.markdown(f"- Hammett slope at σₚ = 0: **ρ ≈ {beta['σₚ'] / np.log(10):+.2f}** (log₁₀ units)")
        if "σₚ²" in beta:
            st.markdown(f"- Curvature β(σₚ²) = **{beta['σₚ²']:+.2f}**: "
                        + ("negative, so the rate gain from stronger acceptors levels off." if beta["σₚ²"] < 0
                           else "positive, so stronger acceptors accelerate more than linearly."))
        st.markdown(f"- tert-butylamine reacts **{np.exp(beta['tert-butylamine']):.3f}×** as fast as n-butylamine (in DCM).")

# ============================== 6. Explore freely ==============================
FREE_COLUMNS = {
    "time_min": "Time (min)", "temperature_C": "Temperature (°C)", "sigma_para": "Hammett σₚ",
    "aldehyde_R_group": "R group", "amine_name": "Amine", "solvent": "Solvent",
    "anchor": "Conversion at 25 °C, 60 min (%)", "Data type": "Data type",
}
CATEGORICAL = {"temperature_C", "aldehyde_R_group", "amine_name", "solvent", "Data type"}

if page == page_names[5]:
    st.markdown("Filter the full dataset and build any chart. **★ = measured** (25 °C, 60 min), "
                "**● = derived** from the measured value (see *Data check*).")
    times = sorted(df["time_min"].unique().tolist())
    temps = sorted(df["temperature_C"].unique().tolist())
    f1, f2, f3, f4, f5 = st.columns(5)
    fr_ald = f1.multiselect("R group", R_ORDER, default=R_ORDER, key="fr_ald")
    fr_am = f2.multiselect("Amine", AMINE_ORDER, default=AMINE_ORDER, key="fr_am")
    fr_sol = f3.multiselect("Solvent", SOLVENTS, default=SOLVENTS, key="fr_sol")
    fr_temp = f4.multiselect("Temperature (°C)", temps, default=temps, key="fr_temp")
    fr_time = f5.multiselect("Time (min)", times, default=times, key="fr_time")
    fr_src = st.radio("Rows", ["All", "Measured only", "Derived only"], horizontal=True, key="fr_src")

    sel = df[df["aldehyde_R_group"].isin(fr_ald) & df["amine_name"].isin(fr_am) & df["solvent"].isin(fr_sol)
             & df["temperature_C"].isin(fr_temp) & df["time_min"].isin(fr_time)].copy()
    if fr_src == "Measured only":
        sel = sel[sel["is_anchor"]]
    elif fr_src == "Derived only":
        sel = sel[~sel["is_anchor"]]
    sel["Data type"] = np.where(sel["is_anchor"], "Measured", "Derived")
    sel["k_eff"] = k_eff(sel["conversion_percent"], sel["time_min"])

    if sel.empty:
        st.info("No rows match these filters. Select at least one option in each filter.")
    else:
        g1, g2, g3, g4, g5 = st.columns(5)
        x_col = g1.selectbox("X axis", list(FREE_COLUMNS), index=0, format_func=FREE_COLUMNS.get, key="fr_x")
        y_col = g2.selectbox("Y axis", ["conversion_percent", "k_eff"], key="fr_y",
                             format_func={"conversion_percent": "Conversion (%)",
                                          "k_eff": "Effective k (min⁻¹, log)"}.get)
        group_options = [None] + [c for c in ["aldehyde_R_group", "amine_name", "solvent", "temperature_C", "Data type"]
                                  if c != x_col]
        colour_col = g3.selectbox("Colour", group_options, index=1, key="fr_c",
                                  format_func=lambda c: "None" if c is None else FREE_COLUMNS[c])
        facet_col = g4.selectbox("Split into panels", [None, "amine_name", "solvent", "temperature_C"], key="fr_f",
                                 format_func=lambda c: "None" if c is None else FREE_COLUMNS[c])
        chart = g5.radio("Chart", ["Points", "Mean line", "Box"], key="fr_chart")

        plot_df = sel.copy()
        for col in {colour_col, facet_col} - {None, x_col}:
            if col == "temperature_C":
                plot_df[col] = plot_df[col].map(lambda v: f"{v:g} °C")
        labels = {**FREE_COLUMNS, "conversion_percent": "Conversion (%)", "k_eff": "Effective k (min⁻¹)"}
        orders = {**ORDERS, "temperature_C": [f"{t:g} °C" for t in temps], "Data type": ["Measured", "Derived"]}
        common = dict(x=x_col, y=y_col, color=colour_col, facet_col=facet_col, facet_col_spacing=0.06,
                      category_orders=orders, labels=labels)
        if chart == "Box":
            free_fig = px.box(plot_df, points="all", **common)
        elif chart == "Mean line":
            keys = [k for k in (x_col, colour_col, facet_col) if k]
            mean_df = plot_df.groupby(keys, as_index=False)[y_col].mean().sort_values(x_col)
            free_fig = px.line(mean_df, markers=True, **common)
        else:
            free_fig = px.scatter(plot_df, symbol="Data type", symbol_map={"Measured": "star", "Derived": "circle"},
                                  hover_data=PAIR_KEYS + ["temperature_C", "time_min", "conversion_percent"],
                                  **common)
            free_fig.update_traces(marker=dict(size=9))
        if y_col == "k_eff":
            free_fig.update_yaxes(type="log")
        free_fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
        st.plotly_chart(free_fig, width="stretch")
        n_meas = int(sel["is_anchor"].sum())
        st.caption(f"{len(sel)} rows shown: {n_meas} measured, {len(sel) - n_meas} derived. "
                   "Mean lines average whatever is selected, so check the row types before drawing conclusions.")
        with st.expander("Show the selected rows"):
            st.dataframe(sel.drop(columns=["one_decimal", "is_anchor"]), width="stretch", hide_index=True)
            st.download_button("Download selection (CSV)",
                               sel.drop(columns=["one_decimal", "is_anchor"]).to_csv(index=False).encode("utf-8"),
                               file_name="imine_selection.csv", mime="text/csv")

# ============================== 7. Design rules ==============================
if page == page_names[6]:
    st.caption("Rules 1–3 come from the 42 measured values; rule 4 comes from derived data and is a hypothesis to test.")
    amine_med = anchors.groupby("amine_name")["k"].median().reindex(AMINE_ORDER)
    best_beta = betas[best_name]
    hx_all = anchors.copy()
    hx_all["log_k"] = np.log10(hx_all["k"])
    sol_better = int((diff >= 0).sum())
    best_row = summary.set_index("Model").loc[best_name]

    r1, r2, r3, r4 = st.columns(4)
    r1.metric("k: n-Bu / tert-Bu", f"{amine_med['n-butylamine'] / amine_med['tert-butylamine']:.0f}×",
              help="Ratio of median effective rate constants")
    r2.metric("Hammett ρ (best model)", f"{best_beta.get('σₚ', np.nan) / np.log(10):+.2f}")
    r3.metric("MTBE ≥ DCM", f"{sol_better} / {len(diff)}")
    r4.metric("Prediction error (held out)", f"{best_row['Held-out MAE (pp)']:.0f} pp")

    curvature = best_beta.get("σₚ²", 0)
    st.markdown(f"""
1. **Use an unhindered primary amine.** The median rate constant drops about
   {amine_med['n-butylamine'] / amine_med['sec-butylamine']:.1f}× from n- to sec-butylamine and
   {amine_med['n-butylamine'] / amine_med['tert-butylamine']:.0f}× from n- to tert-butylamine. Steric bulk next to N is the largest single effect in the data.
2. **Electron-withdrawing para groups speed up the reaction, up to a point.** Donors (NH₂, OMe) are slow because they make the
   carbonyl carbon less electrophilic. The best model includes a {'negative σₚ² term: the gain levels off for strong acceptors (Cl, CF₃, NO₂)' if curvature < 0 else 'σₚ² term'},
   which is typical when the rate-limiting step changes (from amine addition to dehydration of the hemiaminal). σₚ alone predicts a new aldehyde to about ±{best_row['Held-out MAE (pp)']:.0f} pp.
3. **DCM is not required.** MTBE gives equal or higher conversion in {sol_better} of {len(diff)} matched pairs (median {diff.median():+.0f} pp),
   so the greener, less toxic solvent is a reasonable default. Check the specific pair, because the effect varies.
4. **Heat helps (hypothesis).** The derived data imply an apparent Ea of about {ea_implied.median():.0f} kJ/mol (k roughly ×{tf_ratio.median():.1f} from 15 to 35 °C).
   This should be confirmed experimentally, because the time and temperature data are not independent measurements.

**Data quality note:** only 42 of the 504 rows are independent. Any analysis that treats all rows as separate experiments overstates the evidence.
""")

    with st.expander("Measured data (42 rows)"):
        out = anchors[["aldehyde_name", "aldehyde_R_group", "sigma_para", "amine_name", "solvent",
                       "conversion_percent", "k"]]
        st.dataframe(out, width="stretch", hide_index=True)
        st.download_button("Download measured data (CSV)", out.to_csv(index=False).encode("utf-8"),
                           file_name="imine_measured_25C_60min.csv", mime="text/csv")
    with st.expander("Full dataset (504 rows, with anchor value)"):
        full = df.drop(columns=["one_decimal"])
        st.dataframe(full, width="stretch", hide_index=True)
        st.download_button("Download full dataset (CSV)", full.to_csv(index=False).encode("utf-8"),
                           file_name="imine_full_with_anchor.csv", mime="text/csv")
