# Handover: building the Imine Condensation Explorer with an AI assistant

This file follows the workshop tip *"save your progress in a Markdown handover file and start a new conversation when needed"*. It records what was built, why, and what to tell an LLM to continue the work.

## Goal

Build a user-friendly Streamlit dashboard for `imine_condensation_dataset_for_students.csv` that helps a chemist (with no coding background) explore reactivity trends, infer design rules and predict future reactivity.

## Development log

| Version | What happened | Lesson |
|---|---|---|
| **v1** (LLM via dAIsy) | First prompt described the goal, the columns and one example row, and asked for: sidebar filters, kinetics plot, aldehyde×amine heatmap, DCM vs MTBE bar chart, conversion vs Hammett σₚ plot and RDKit structures. The result ran first time and had good data validation and a "matched conditions" comparison. | A clear prompt with the data schema plus one example row was enough for a working app. |
| **v2** (review) | Review found: Hammett plot averaged over amines (hid trends), no numbers on the heatmap, solvent comparison only as one mean, no design-rules summary, deprecated `use_container_width`. All fixed; added a computed *Design rules* section and CSV download. | Ask the LLM to *critique* its own output against the task brief, not only to write code. |
| **v3** (more analysis) | Added tabs, KPI cards, ± SD error bars, a custom chart builder, first-order fits (k per condition), Arrhenius and Hammett plots, and a physics-informed predictive model with leave-one-aldehyde-out validation. | The held-out error (~16 pp) was much worse than the training error, a first sign that something about the data was unusual. |
| **v4** (rebuild) | Checking the data showed that only the 42 values at 25 °C / 60 min are independent: they are the only ones reported as rounded numbers, and every other value is a deterministic function of them (identical anchor → identical curve). The app was rebuilt around this: a *Data check* tab, chemistry conclusions from the 42 measured points only, derived data clearly labelled, and a model comparison (linear vs curved Hammett) on the real data. One earlier conclusion was corrected: MTBE is *as good as* DCM (median 0 pp), not better (the +5 pp came from derived rows). | Validate the data before trusting the analysis. More rows is not more evidence. |
| **v5** (usability + sharing) | Added a plain-language glossary, a mechanism diagram linking each factor to a reaction step, a free "Explore" tab with measured/derived markers, `README.md`, `requirements.txt` and this handover file. | A dashboard for other people needs a glossary and a README; one for yourself does not. |

## Current state

- `app.py` runs with `streamlit run app.py` (tested with Streamlit 1.67, Python 3.14; see `requirements.txt`).
- Key numbers: n-Bu/tert-Bu rate ratio ≈ 73×; concave Hammett plot (quadratic R² ≈ 0.89 for n-BuNH₂/DCM); MTBE ≥ DCM in 15/21 pairs; best model held-out MAE ≈ 12 pp.

## Prompt to continue in a new conversation

```
I have a Streamlit app (app.py, attached) analysing an imine condensation dataset
(7 para-benzaldehydes x 3 butylamines x DCM/MTBE x 15/25/35 °C x 20-80 min, target conversion_percent).
Important: only the 42 rows at 25 °C/60 min are independent measurements; all other rows are
derived from them. The app already has tabs: Data check, Reactivity map, Effects (mechanism,
sterics, Hammett, solvent), Time & temperature (derived), Predict (LOAO-validated model),
Explore freely, Design rules. Please <describe the next change>. Keep derived data labelled.
```

## Ideas not yet done

- Add more aldehyde descriptors (e.g. σ⁺, steric parameters, computed LUMO energy) and test if they beat σₚ in cross-validation.
- Bootstrap confidence intervals for ρ and the solvent effect (only 7 and 21 points).
- Suggest the next most informative experiment (active learning / largest model uncertainty), closing the loop as in a self-driving lab.
