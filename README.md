# Predicting Hospital Readmission Risk for Diabetic Patients

> An end-to-end machine learning project that scores diabetic patients' readmission risk at discharge, so a hospital can direct limited follow-up resources (nurse calls, medication reviews, outpatient bookings) to the patients who need them most.

**Stack:** Python · pandas · scikit-learn · matplotlib · seaborn · Google Colab
**Data:** 25,000 real inpatient encounters from 130 US hospitals (UCI Diabetes 130-US Hospitals, 1999–2008)
**Notebook:** [Open in Colab](https://colab.research.google.com/drive/1Rv8uqrLg_Bo1QuAzZxMeKdSylysWVq_L?usp=sharing)

---

## TL;DR

| | Logistic Regression | Random Forest |
|---|---|---|
| **Test AUC-ROC** | 0.643 | **0.653** |
| **Test Recall** (readmissions caught) | 0.406 | **0.488** |
| **Test Accuracy** | 0.607 | **0.616** |
| 5-fold CV AUC (mean ± std) | 0.642 ± 0.005 | 0.656 ± 0.004 |

- Both models beat the majority-class baseline (53% accuracy), with Random Forest consistently ahead on ranking (AUC gap ≈ 3–4× the fold-to-fold std).
- On the validation set, Random Forest caught **128 more readmitted patients** than Logistic Regression (835 vs 707), at the cost of 125 extra false alarms.
- Lowering the decision threshold from 0.5 to 0.4 lifts Logistic Regression recall from **0.40 to 0.76** — the single biggest lever for a screening use case.
- **Key insight:** a far more flexible model gave only a marginal gain, which points to the *features*, not the algorithm, as the bottleneck. The next win is better data, not a bigger model.

---

## The problem

Unplanned readmissions are expensive, strain beds and staff, and often signal that discharge planning or follow-up care fell short. Diabetic patients are especially exposed: the Endocrine Society (2019) reported they are around 40% more likely to be readmitted within 30 days than patients without diabetes.

Most hospitals screen with simple rules ("older patients with long stays are high risk"). Those rules miss combinations of factors, e.g. a patient with a moderate stay but several recent emergency and inpatient visits. This project tests how far a data-driven model can go using only the information available at discharge.

**Framing the decision:** this is a screening problem, so errors are not equal.
- **False negative** (high-risk patient sent home without follow-up) → missed intervention, possible readmission.
- **False positive** (low-risk patient gets an extra call) → small, recoverable cost.

That asymmetry drives the metric choices and the threshold analysis below.

---

## Data

Source: [Hospital Readmissions (Kaggle, dubradave)](https://www.kaggle.com/datasets/dubradave/hospital-readmissions), a cleaned subset of the [UCI Diabetes 130-US Hospitals dataset](https://doi.org/10.24432/C5230J) (Strack et al., 2014).

- **25,000 encounters**, 16 input features, binary target `readmitted` (yes/no)
- **Close to balanced:** 53% not readmitted / 47% readmitted, so accuracy is not automatically misleading, but recall still matters most
- **Features:**
  - Utilisation history: prior-year `n_inpatient`, `n_outpatient`, `n_emergency`
  - Encounter: `time_in_hospital`, `n_lab_procedures`, `n_procedures`, `n_medications`
  - Clinical: `medical_specialty`, `diag_1`–`diag_3` (grouped diagnoses), `glucose_test`, `A1Ctest`
  - Treatment: `change` (medication regimen changed), `diabetes_med`
  - Demographic: `age` (10-year bands)

**Data-quality finding:** `isnull()` reports zero missing values, but `medical_specialty` and `diag_1`–`diag_3` hide missingness as an explicit `"Missing"` category. I kept it as its own category instead of dropping rows, because *whether* a record is incomplete can itself be informative.

---

## Approach

```
raw CSV ──► encode categoricals + target ──► stratified split 70 / 15 / 15
                                                    │
                     ┌──────────────────────────────┴───────────────┐
                     ▼                                              ▼
          Logistic Regression                              Random Forest
       (interpretable baseline)                  (200 trees, max_depth=10)
                     └──────────────► validation: metrics, confusion
                                      matrices, threshold sweep, 5-fold CV
                                                    │
                                                    ▼
                                  test set — evaluated exactly once
```

**Why these two models**
- **Logistic Regression**: every coefficient has a direction and size a clinician can read. In decision support, being able to explain *why* a patient was flagged matters as much as a small accuracy gain.
- **Random Forest**: captures non-linear interactions (e.g. utilisation × diagnosis) to test whether extra model capacity buys real performance.

**Evaluation discipline**
- Stratified splits preserve the 53/47 ratio in every set.
- All iteration (model comparison, threshold sweep) used the validation set only; the test set was touched once at the end.
- 5-fold stratified cross-validation on train + validation (21,250 rows) to check results don't depend on one lucky split.
- Five metrics reported side by side (Accuracy, Precision, Recall, F1, AUC-ROC) because accuracy alone hides the error type that matters.

---

## Results

### Validation vs test (no sign of overfitting to the validation set)

| Model | Set | Accuracy | Precision | Recall | F1 | AUC-ROC |
|---|---|---|---|---|---|---|
| Logistic Regression | Validation | 0.607 | 0.630 | 0.401 | 0.490 | 0.642 |
| Logistic Regression | **Test** | 0.607 | 0.627 | 0.406 | 0.493 | 0.643 |
| Random Forest | Validation | 0.608 | 0.607 | 0.473 | 0.532 | 0.651 |
| Random Forest | **Test** | 0.616 | 0.615 | 0.488 | 0.545 | 0.653 |

Validation and test differ by at most ~0.015 on every metric.

### Where the errors are (validation, n = 3,751)

| | Caught readmissions (TP) | Missed readmissions (FN) | False alarms (FP) |
|---|---|---|---|
| Logistic Regression | 707 | 1,057 | 416 |
| Random Forest | **835** | **929** | 541 |

At the default 0.5 threshold, Logistic Regression misses 60% of patients who are actually readmitted. That is the number that motivated the threshold analysis.

### Threshold = the operating knob (Logistic Regression, validation)

| Threshold | Precision | Recall | F1 |
|---|---|---|---|
| 0.5 (default) | 0.630 | 0.401 | 0.490 |
| **0.4** | **0.530** | **0.759** | **0.624** |
| 0.3 | 0.472 | 0.988 | 0.639 |
| 0.2 | 0.470 | 1.000 | 0.640 |

- **0.4 is the practical sweet spot:** recall nearly doubles while precision stays above the base rate.
- At **0.3 and below**, precision collapses to the base rate (0.47), meaning the model flags almost everyone. High F1, but useless as a filter.
- In production, the threshold should be set from the hospital's **follow-up capacity** (how many patients the team can actually call per day), not fixed at 0.5.

### What drives risk

Random Forest feature importance (top 3):
1. `n_inpatient` — prior-year inpatient admissions (0.22, by a clear margin)
2. `n_lab_procedures` (0.115)
3. `n_medications` (0.11)

Prior inpatient history is the strongest single signal, which matches clinical intuition: patients who have been admitted before are more likely to come back.

---

## What I learned

1. **The data sets the ceiling, not the model.** A 200-tree ensemble improved AUC by only ~0.014 over a linear model. If strong non-linear signal existed in these 16 features, the gap would be much larger. More algorithm won't fix missing information.
2. **The training objective ≠ the business objective.** Log loss treats a missed readmission and a false alarm as equally bad; the hospital does not. Without explicitly handling that (threshold tuning or class weights), a "good" model quietly optimises for the wrong thing.
3. **"No nulls" doesn't mean "no missing data."** Always check what values like `"Missing"`, `"?"` or `"Unknown"` are hiding.
4. **Honest framing beats inflated metrics.** An AUC of ~0.65 is weak-to-moderate discrimination. Useful for prioritising a follow-up list, not for making clinical decisions on its own.

---

## Limitations

- **Feature set is thin:** no lab values, vitals, dosages or clinical notes.
- **No patient ID** in this subset, so repeat encounters for the same patient could fall in both train and test.
- **Single population** (US hospitals, 1999–2008); would need re-validation before use in another health system.
- **No systematic hyperparameter search** — configurations are sensible defaults, not tuned.

## Next iteration

- [ ] Replace `LabelEncoder` with one-hot encoding + `StandardScaler` inside a scikit-learn `Pipeline` (label-encoding nominal features imposes a false order on Logistic Regression)
- [ ] Cost-sensitive training (`class_weight="balanced"` or custom FN penalty) instead of only post-hoc thresholding
- [ ] Gradient boosting (LightGBM / XGBoost) + `GridSearchCV` / Optuna
- [ ] Permutation importance or SHAP to confirm drivers (impurity-based importance favours high-cardinality numeric features)
- [ ] Probability calibration, then a capacity-based threshold ("top N% riskiest patients per day")
- [ ] Wrap `predict_readmission(record) -> probability` in a small FastAPI / Streamlit demo

---

## Run it yourself

```bash
pip install pandas scikit-learn matplotlib seaborn kagglehub
```

```python
import kagglehub, pandas as pd
path = kagglehub.dataset_download("dubradave/hospital-readmissions")
df = pd.read_csv(path + "/hospital_readmissions.csv")
```

Then run the notebook top to bottom. All splits and models use `random_state=42` for reproducibility.

---

## References

- Strack, B. et al. (2014). *Impact of HbA1c measurement on hospital readmission rates: Analysis of 70,000 clinical database patient records.* BioMed Research International. https://doi.org/10.1155/2014/781670
- Strack, B. et al. (2014). *Diabetes 130-US Hospitals for Years 1999–2008* [Data set]. UCI Machine Learning Repository. https://doi.org/10.24432/C5230J
- Endocrine Society (2019). *Patients with diabetes are 40 percent more likely to be readmitted to the hospital* [Press release].
- Pedregosa, F. et al. (2011). *Scikit-learn: Machine learning in Python.* JMLR 12, 2825–2830.

---

*Built by Bao Anh Nguyen · [LinkedIn](#) · [GitHub](#)*
