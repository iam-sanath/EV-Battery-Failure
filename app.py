"""EV Battery Failure Predictor - Streamlit app for the ML Mini Project (Phase 2 best model).

Run locally:   streamlit run app.py
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).parent

st.set_page_config(page_title="EV Battery Failure Predictor", page_icon="", layout="wide")


@st.cache_resource
def load_bundle():
    return joblib.load(APP_DIR / "ev_battery_model.joblib")


@st.cache_data
def load_sample():
    path = APP_DIR / "sample_vehicles.csv"
    return pd.read_csv(path) if path.exists() else None


bundle = load_bundle()
model = bundle["model"]
FEATURES = bundle["feature_columns"]
NUM_COLS = bundle["numeric_columns"]
CAT_COLS = bundle["categorical_columns"]
INT_COLS = set(bundle["integer_columns"])
DEFAULTS = bundle["defaults"]
RANGES = bundle["ranges"]
CATEGORIES = bundle["categories"]
TOP = bundle["top_features"]
METRICS = bundle["test_metrics"]
sample_df = load_sample()


def pretty(col):
    return col.replace("_", " ").replace("BMS", "BMS").strip().capitalize()


def key(col):
    return f"in_{col}"


def init_inputs(row=None):
    """Put default values (or one sample vehicle's values) into the input widgets."""
    for c in FEATURES:
        value = DEFAULTS[c] if row is None else row.get(c, DEFAULTS[c])
        if c in NUM_COLS:
            value = DEFAULTS[c] if pd.isna(value) else float(value)
            st.session_state[key(c)] = int(round(value)) if c in INT_COLS else float(value)
        else:
            value = DEFAULTS[c] if pd.isna(value) or str(value) not in CATEGORIES[c] else str(value)
            st.session_state[key(c)] = value


def load_random_sample():
    row = sample_df.sample(1).iloc[0]
    init_inputs(row.to_dict())
    st.session_state["sample_actual"] = int(row["actual_battery_failure"]) if "actual_battery_failure" in row else None


def reset_inputs():
    init_inputs()
    st.session_state["sample_actual"] = None


if key(FEATURES[0]) not in st.session_state:
    init_inputs()
    st.session_state["sample_actual"] = None


def input_widget(c, container):
    lo, hi = RANGES.get(c, (None, None))
    help_txt = f"Typical range in the training data: {lo:,.2f} to {hi:,.2f}" if c in NUM_COLS else None
    if c in CAT_COLS:
        container.selectbox(pretty(c), CATEGORIES[c], key=key(c))
    elif c in INT_COLS:
        container.number_input(pretty(c), step=1, key=key(c), help=help_txt)
    else:
        container.number_input(pretty(c), step=float(max((hi - lo) / 100, 0.01)), format="%.3f", key=key(c), help=help_txt)


def predict(df):
    df = df.reindex(columns=FEATURES)
    proba = model.predict_proba(df)[:, 1]
    label = model.predict(df)
    return label, proba


# ------------------------------------------------------------------ header
st.title("🔋 EV Battery Failure Predictor")
st.caption(
    f"ML Mini Project | Best model: **{bundle['model_name']} (RBF kernel)** with preprocessing pipeline | "
    f"trained on {bundle['train_rows']:,} vehicles, tested on {bundle['test_rows']:,}"
)

tab_single, tab_batch, tab_perf = st.tabs(["Single vehicle", "Batch prediction (CSV)", "Model performance"])

# ------------------------------------------------------------------ single vehicle
with tab_single:
    st.subheader("Predict failure for one vehicle")
    st.write(
        "Enter the battery readings below. The **most important features** are shown first; all other "
        "features are pre-filled with typical values (the training median, or the most common category) and can be changed under *More features*."
    )

    b1, b2, _ = st.columns([1.2, 1, 3])
    b1.button("Load a random test vehicle", on_click=load_random_sample, disabled=sample_df is None,
              help="Fills every input with a real vehicle from the test set")
    b2.button("↺ Reset to typical values", on_click=reset_inputs)

    st.markdown("##### Most important features")
    cols = st.columns(3)
    for i, c in enumerate(TOP):
        input_widget(c, cols[i % 3])

    others = [c for c in FEATURES if c not in TOP]
    with st.expander(f"More features ({len(others)}) - pre-filled with typical values"):
        cols2 = st.columns(3)
        for i, c in enumerate(others):
            input_widget(c, cols2[i % 3])

    st.divider()
    if st.button("Predict", type="primary", width="stretch"):
        row = pd.DataFrame([{c: st.session_state[key(c)] for c in FEATURES}])
        label, proba = predict(row)
        label, p = int(label[0]), float(proba[0])

        r1, r2 = st.columns([1, 2])
        with r1:
            st.metric("Estimated failure probability", f"{p:.1%}")
            st.progress(min(max(p, 0.0), 1.0))
        with r2:
            if label == 1:
                st.error("**Prediction: BATTERY FAILURE likely.** Schedule an inspection of this battery.")
            else:
                st.success("**Prediction: No failure expected.** The battery looks healthy.")
            actual = st.session_state.get("sample_actual")
            if actual is not None:
                st.info(f"This was a real test-set vehicle. Actual outcome: **{'Failure' if actual == 1 else 'No failure'}**"
                        f" - the prediction is **{'correct ✔' if actual == label else 'wrong ✘'}**.")
        st.caption("The yes/no prediction comes from the SVM's decision boundary. The probability is the SVM's "
                   "calibrated estimate (Platt scaling) and is shown as a guide to how confident the model is.")

# ------------------------------------------------------------------ batch
with tab_batch:
    st.subheader("Predict for many vehicles at once")
    st.write("Upload a CSV with the same columns as the original dataset. Missing columns are filled with typical values; "
             "extra columns (such as IDs) are ignored.")

    uploaded = st.file_uploader("Upload CSV", type="csv")
    use_sample = st.checkbox("Or use the included sample file (30 test vehicles)", value=uploaded is None and sample_df is not None)

    data = None
    if uploaded is not None:
        data = pd.read_csv(uploaded)
    elif use_sample and sample_df is not None:
        data = sample_df.copy()

    if data is not None:
        missing = [c for c in FEATURES if c not in data.columns]
        if len(missing) == len(FEATURES):
            st.error("None of the expected columns were found in this file. Please upload the EV battery dataset format.")
        else:
            if missing:
                st.warning(f"{len(missing)} column(s) missing, filled with typical values: {', '.join(missing[:10])}"
                           + (" ..." if len(missing) > 10 else ""))
            X_in = data.reindex(columns=FEATURES)
            for c in missing:
                X_in[c] = DEFAULTS[c]
            label, proba = predict(X_in)

            out = data.copy()
            out.insert(0, "failure_probability", np.round(proba, 3))
            out.insert(0, "prediction", np.where(label == 1, "FAILURE", "No failure"))

            n_fail = int(label.sum())
            m1, m2, m3 = st.columns(3)
            m1.metric("Vehicles", len(out))
            m2.metric("Predicted failures", n_fail)
            m3.metric("Predicted healthy", len(out) - n_fail)

            if "actual_battery_failure" in out.columns:
                correct = int((label == out["actual_battery_failure"].values).sum())
                st.info(f"This file includes the actual outcome: the model is correct for **{correct} of {len(out)}** vehicles.")

            def highlight(row):
                color = "background-color: rgba(220, 53, 69, 0.15)" if row["prediction"] == "FAILURE" else ""
                return [color] * len(row)

            show_cols = ["prediction", "failure_probability"] + (["actual_battery_failure"] if "actual_battery_failure" in out else []) \
                        + [c for c in ["vehicle_id"] + TOP[:6] if c in out.columns]
            st.dataframe(out[show_cols].style.apply(highlight, axis=1).format(precision=3, na_rep="missing"), width="stretch", height=420)
            st.download_button("Download all predictions (CSV)", out.to_csv(index=False).encode(),
                               "ev_battery_predictions.csv", "text/csv")

# ------------------------------------------------------------------ performance
with tab_perf:
    st.subheader(f"How good is the model? ({bundle['model_name']}, test set of {bundle['test_rows']:,} vehicles)")

    st.markdown("**Failure class (class 1, the minority - 7% of vehicles)**")
    a, b, c, d = st.columns(4)
    a.metric("Recall / Sensitivity", f"{METRICS['Recall']:.3f}", help="Share of real failures that the model catches")
    b.metric("Precision", f"{METRICS['Precision']:.3f}", help="Share of predicted failures that really fail")
    c.metric("F1-score", f"{METRICS['F1-score']:.3f}")
    d.metric("PR-AUC", f"{METRICS['PR-AUC']:.3f}")

    st.markdown("**No-failure class (class 0, the majority - 93% of vehicles)**")
    a, b, c, d = st.columns(4)
    a.metric("Specificity", f"{METRICS['Specificity (class 0)']:.3f}", help="Share of healthy batteries correctly cleared")
    b.metric("NPV", f"{METRICS['NPV (class 0)']:.3f}", help="When the model says 'no failure', how often it is right")
    c.metric("Accuracy", f"{METRICS['Accuracy']:.3f}")
    d.metric("ROC-AUC", f"{METRICS['ROC-AUC']:.3f}")

    st.markdown("**Both classes together (imbalance-aware)**")
    a, b, c, _ = st.columns(4)
    a.metric("Balanced accuracy", f"{METRICS['Balanced accuracy']:.3f}")
    b.metric("G-mean", f"{METRICS['G-mean']:.3f}")
    c.metric("MCC", f"{METRICS['MCC']:.3f}")

    st.markdown("**Confusion matrix (test set)**")
    cm = np.array(bundle["confusion_matrix"])
    cm_df = pd.DataFrame(cm, index=["Actual: No failure", "Actual: Failure"], columns=["Predicted: No failure", "Predicted: Failure"])
    left, right = st.columns([1.3, 1])
    left.dataframe(cm_df.style.background_gradient(cmap="Blues", axis=None), width="stretch")
    tn, fp, fn, tp = cm.ravel()
    right.markdown(f"- **{tp}** failures caught\n-  **{fn}** failures missed\n-  **{fp}** false alarms\n-  **{tn}** healthy batteries correctly cleared")

    st.markdown("**Most important features** (Random Forest ranking)")
    st.write(", ".join(f"`{f}`" for f in TOP))
