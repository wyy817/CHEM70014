# 🧪 Imine Condensation Explorer

An interactive Streamlit dashboard for exploring how aldehyde electronics, amine sterics, solvent, temperature and time affect the rate of imine condensation, and for predicting the reactivity of untested aldehydes.

Built for CHEM70014 *Automation and Enabling Technologies in Chemistry* (Imperial College London, Workshop 2: "Exploring Chemical Data using Dashboards").

```
Ar–CHO  +  R′–NH₂   ──(DCM or MTBE, MgSO₄)──▶   Ar–CH=N–R′  +  H₂O
```

## Quick start

```bash
# 1. create an environment (conda shown; venv works too)
conda create -n imine-dashboard python=3.12
conda activate imine-dashboard

# 2. install dependencies
pip install -r requirements.txt

# 3. run (the CSV must sit next to app.py)
streamlit run app.py
```

The app opens at <http://localhost:8501>.

## The dataset

`imine_condensation_dataset_for_students.csv`: 504 rows covering 7 para-substituted benzaldehydes (NH₂, OMe, CH₃, H, Cl, CF₃, NO₂) × 3 amines (n-, sec-, tert-butylamine) × 2 solvents (DCM, MTBE) × 3 temperatures (15, 25, 35 °C) × 4 times (20–80 min). The target is `conversion_percent`. Setting based on *J. Chem. Educ.* 2023, 100, 4456–4461.

> **Important:** only the **42 values at 25 °C / 60 min** behave like independent measurements. Every other row is a deterministic function of the same reaction's 25 °C / 60 min value: reactions with equal anchor values have identical curves at every time and temperature. The *Data check* tab shows the evidence. The dashboard therefore draws chemical conclusions from the 42 measured values and labels everything else as *derived*.

## What each page answers

Pages are selected from the sidebar.

| Page | Question it answers |
|---|---|
| 🔍 Data check | How much independent information is in the data? |
| 🗺️ Reactivity map | Which aldehyde–amine–solvent combinations react well? |
| 🔬 Effects | Why? Mechanism, amine sterics, Hammett plot, DCM vs MTBE (paired) |
| ⏱️ Time & temperature | What do the (derived) kinetic curves look like, and which rate law fits them? |
| 🔮 Predict | Can we predict an aldehyde we have never run? Model comparison with leave-one-aldehyde-out validation |
| 🧭 Explore freely | Free chart builder with filters; measured vs derived points are marked |
| 📋 Design rules & data | Summary of the rules, plus CSV downloads |

## Main findings

1. **Amine sterics dominate:** the effective rate constant drops ~73× from n-butyl to tert-butylamine.
2. **Electron-withdrawing para groups accelerate the reaction, but the gain levels off:** the Hammett plot is concave (ρ > 0 at σₚ ≈ 0). This is consistent with a change in the rate-limiting step from amine addition to hemiaminal dehydration. Adding a σₚ² term lowers held-out prediction error from ~17 to ~12 percentage points.
3. **DCM is not required:** in matched pairs, MTBE gives equal or higher conversion in 15 of 21 cases (median difference 0 pp). The greener solvent is a reasonable default, but the effect depends on the pair.
4. **Temperature (hypothesis):** the derived data imply an apparent Ea of ~34 kJ/mol. This should be confirmed experimentally.

## Does the dashboard need documentation?

**For everyday use, no:** a *How to use* panel, a plain-language glossary, tooltips on every control and captions under every chart are built into the app.
**For interpretation, yes:** two things cannot be seen from the charts alone and are documented here and in the app: (1) only 42 rows are independent measurements, and (2) the predictive model's assumptions (first-order kinetics at 60 min, additive effects on ln k, σₚ as the only aldehyde descriptor).

## Method notes

- **Effective rate constant:** k = −ln(1 − X)/t from a single conversion X at time t.
- **Predictive model:** ln k = amine + solvent + amine×solvent + f(σₚ), fitted by non-linear least squares in conversion space (SciPy). Four forms of f(σₚ) are compared by leave-one-aldehyde-out cross-validation, and the best held-out model is used by default.
- **Curves for new predictions** at other times and temperatures are interpolated from the dataset's own anchor → curve relation, so they inherit its assumptions.

## Files

| File | Purpose |
|---|---|
| `app.py` | The Streamlit dashboard |
| `imine_condensation_dataset_for_students.csv` | Course dataset |
| `requirements.txt` | Python dependencies |
| `handover.md` | Development log: how the dashboard was built with an AI assistant |

## Acknowledgements

Dataset and task from the CHEM70014 teaching team. Dashboard developed with the help of an AI coding assistant (Claude Code); see `handover.md`.
