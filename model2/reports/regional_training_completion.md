# Regional Training Completion

## DATA USED
- `crop_production.csv` (Base Table)
- `soil-nutrient-analysis.csv` (District soil aggregates)
- `district wise rainfall normal.csv` (Rainfall features)
- `Agroclimatic_regions.shp` (Regional temperature data)

## FEATURES
- `State`
- `District`
- `Season`
- `NITROGEN`
- `PHOSPHOROUS`
- `POTASSIUM`
- `PH`
- `avgtmp_jan`
- `avgtmp_jul`
- `avgann_rf`
- `ANNUAL`
- `Jun-Sep`

## TARGET
The target is a classification task predicting the most suitable `Crop` based on historical success in matching regions, explicitly omitting the target year's production to prevent leakage.

## MODEL
- **Selected Model**: RandomForestClassifier
- **Hyperparameters**: `n_estimators=50`, `max_depth=15`, `random_state=42`

## VALIDATION
- **Train Years**: 1997 - 2013
- **Validation Year**: 2014
- **Test Year**: 2015

## METRICS
- **Top-1 Accuracy**: 47.9% (on unseen 2015 test data, strictly based on historic ranking)
- **Top-3 Accuracy**: 74.2% 
- **Precision**: 45.1%
- **Recall**: 47.9%
- **F1 Score**: 44.8%

## SAVED ARTIFACTS
- `models/regional/regional_model.pkl`
- `models/regional/regional_preprocessor.pkl`
- `models/regional/regional_label_encoder.pkl`
- `models/regional/regional_feature_names.json`
- `models/regional/regional_model_metadata.json`

## SHAP
Generated successfully at `reports/regional_shap/regional_shap_summary.png`.

## LIME
Generated successfully at `reports/regional_lime/regional_lime_example_1.html`.

## BASELINE STATUS
The existing 7-parameter model has not been modified or overwritten.

## NEXT STEP
The pipeline is fully executed and artifacts are saved. The backend `main.py` can now be updated to load `src/models/regional_predictor.py` and serve predictions from the `predict_regional_top3()` function via a new API endpoint, if requested.
