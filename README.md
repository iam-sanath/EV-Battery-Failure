# EV Battery Failure Predictor (Streamlit app)

Web app for the best model from **ML Mini Project **: an **SVM (RBF kernel)** inside a scikit-learn pipeline
(imputation + scaling + one-hot encoding), trained on 16,000 vehicles and tested on 4,000.


## Deployed Link : https://ev-battery-failure.streamlit.app/

## Files
| File | What it is |
|---|---|
| `app.py` | The Streamlit app |
| `ev_battery_model.joblib` | Saved pipeline + defaults + test metrics (created by the last step of the Phase 2 notebook) |
| `sample_vehicles.csv` | 30 test-set vehicles (with the actual outcome) to try the batch feature |
| `requirements.txt` | Libraries needed (scikit-learn version must match the one that saved the model) |

## Features
- **Single vehicle**: enter readings (top 12 features shown first, rest pre-filled), or load a random real test vehicle, then predict.
- **Batch prediction**: upload a CSV in the dataset format and download predictions for every vehicle.
- **Model performance**: test-set metrics for both classes (recall, precision, specificity, NPV, balanced accuracy, G-mean, MCC) and the confusion matrix.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```



If you re-run the notebook on your own computer, it creates a new `ev_battery_model.joblib` and `requirements.txt`.
Upload **both** together, because the saved model must be loaded with the same scikit-learn version.
