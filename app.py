import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# -------------------------------------------------------
# PAGE SETUP
# -------------------------------------------------------

st.set_page_config(page_title="Obesity Risk Checker", layout="wide")

# -------------------------------------------------------
# LOAD DATA
# -------------------------------------------------------

@st.cache_data
@st.cache_data
def load_data():
    df = pd.read_csv("brfss_obesity_clean.csv")

    # This dataset is already filtered for obesity, but we add this as a safety check
    df["Data_Value"] = pd.to_numeric(df["Data_Value"], errors="coerce")
    df = df.dropna(subset=["Data_Value"])

    # Create binary target
    df["high_obesity"] = (df["Data_Value"] >= 35).astype(int)

    return df



@st.cache_resource
def train_model(obesity_df: pd.DataFrame):
    feature_cols = ["YearStart", "LocationAbbr",
                    "StratificationCategory1", "Stratification1"]

    X = obesity_df[feature_cols]
    y = obesity_df["high_obesity"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    numeric_features = ["YearStart"]
    categorical_features = ["LocationAbbr",
                            "StratificationCategory1", "Stratification1"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    rf_clf = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", RandomForestClassifier(n_estimators=300, random_state=42)),
    ])

    rf_clf.fit(X_train, y_train)
    return rf_clf


# -------------------------------------------------------
# MAIN APP (YEAR REMOVED)
# -------------------------------------------------------

st.title("Obesity Risk Checker")

st.write(
    """
Select a **state**, then choose demographic groups for  
**Race/Ethnicity, Sex, Age, Education, and Income**.

The model predicts whether each demographic group is likely to have  
**high obesity prevalence (≥ 35%)** based on CDC BRFSS data.
"""
)

with st.spinner("Loading data and training the model..."):
    obesity_df = load_data()
    model = train_model(obesity_df)

# -------------------------------------------------------
# STATE ONLY (NO YEAR)
# -------------------------------------------------------

st.subheader("1. Choose state")

state_options = sorted(obesity_df["LocationAbbr"].unique())
state_choice = st.selectbox("State (two-letter code)", state_options)

# filter by state only (all years included)
subset = obesity_df[obesity_df["LocationAbbr"] == state_choice].copy()

# -------------------------------------------------------
# DEMOGRAPHIC GROUP SECTIONS
# -------------------------------------------------------

st.subheader("2. Choose demographic groups")

GROUP_TYPES = [
    ("Race/Ethnicity", "Race/Ethnicity group", "race_group"),
    ("Sex", "Sex group", "sex_group"),
    ("Age (years)", "Age group", "age_group"),
    ("Education", "Education group", "edu_group"),
    ("Income", "Income group", "inc_group"),
]

selected_groups = {}
actual_values = {}

for group_type, label, key in GROUP_TYPES:
    st.markdown(f"### {group_type}")

    g_sub = subset[subset["StratificationCategory1"] == group_type].copy()

    if g_sub.empty:
        st.write(f"_No {group_type.lower()} data available for this state._")
        continue

    options = sorted(g_sub["Stratification1"].unique())
    choice = st.selectbox(label, options, key=key)
    selected_groups[group_type] = choice

    match_row = g_sub[g_sub["Stratification1"] == choice]
    actual_values[group_type] = (
        match_row["Data_Value"].iloc[0] if not match_row.empty else None
    )

st.markdown("---")

predict_clicked = st.button("🔍 Predict obesity risk for all selected groups")

# -------------------------------------------------------
# RESULTS
# -------------------------------------------------------

if predict_clicked and selected_groups:
    rows_to_predict = []
    meta_info = []

    for g_type, g_choice in selected_groups.items():
        # we use latest available year for that group in that state
        latest_year = subset[
            subset["StratificationCategory1"] == g_type
        ]["YearStart"].max()

        rows_to_predict.append({
            "YearStart": latest_year,
            "LocationAbbr": state_choice,
            "StratificationCategory1": g_type,
            "Stratification1": g_choice,
        })
        meta_info.append((g_type, g_choice))

    X_input = pd.DataFrame(rows_to_predict)

    probs = model.predict_proba(X_input)[:, 1]
    preds = model.predict(X_input)

    st.subheader("3. Estimated obesity risk by demographic group")

    for i, (g_type, g_choice) in enumerate(meta_info):
        proba = probs[i]
        pred_class = preds[i]
        actual_val = actual_values.get(g_type)

        st.markdown(f"#### {g_type}")

        c1, c2 = st.columns(2)

        with c1:
            st.write("**Group selected:**", g_choice)

            if actual_val is not None:
                st.write(f"Observed BRFSS prevalence: **{actual_val:.1f}%**")

        with c2:
            st.metric(
                "Predicted probability (≥ 35% obesity)",
                f"{proba * 100:.1f}%"
            )
            if pred_class == 1:
                st.write("🟥 **High obesity risk group**")
            else:
                st.write("🟩 **Not a high obesity risk group**")

    st.info(
        "Each prediction is based on BRFSS survey data for the selected state. "
        "Because BRFSS reports each demographic breakdown separately, predictions "
        "are shown independently for Race, Sex, Age, Education, and Income groups."
    )
