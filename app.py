import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from pathlib import Path

# Basic streamlit setup
st.set_page_config(
    page_title="Healthcare Prediction Playground",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Some basic CSS to make it look cleaner
st.markdown("""
    <style>
    .main {
        padding-top: 2rem;
    }
    
    h2 {
        margin-top: 2rem;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid #e0e0e0;
    }
    
    .block-container {
        padding-top: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# Loading the model and SHAP explainer - using cache so it doesn't reload every time
@st.cache_resource
def load_model():
    """
    Load the trained LightGBM model from disk.
    
    Uses Streamlit's cache_resource decorator to ensure the model is only loaded once
    and reused across app reruns for better performance.
    
    Returns:
        Trained LightGBM classifier model
    """
    model_path = Path("fairness_artifacts/final_lightgbm_model.pkl")
    model = joblib.load(model_path)
    return model

@st.cache_resource
def load_explainer():
    """
    Load the pre-computed SHAP explainer from disk.
    
    The explainer was trained on the training data and is used to generate
    SHAP values for individual predictions in real-time.
    
    Returns:
        SHAP explainer object
    """
    explainer_path = Path("fairness_results/shap_explainer.pkl")
    return joblib.load(explainer_path)

@st.cache_data
def load_feature_template():
    """
    Load the feature column names from training data.
    
    This ensures we know exactly what features the model expects and in what order.
    Critical for creating feature vectors that match the training format.
    
    Returns:
        List of feature column names
    """
    X_train = pd.read_parquet("fairness_artifacts/X_train.parquet")
    return X_train.columns.tolist()

@st.cache_resource
def get_model_feature_order():
    """
    Get the exact feature order expected by the trained model.
    
    LightGBM models store feature names in a specific order. This function
    retrieves that order to ensure feature vectors match exactly.
    
    Returns:
        List of feature names in the order expected by the model
    """
    model = load_model()
    if hasattr(model, 'feature_name_') and model.feature_name_ is not None:
        return list(model.feature_name_)
    else:
        # Fallback if model doesn't have feature names
        return load_feature_template()

@st.cache_data
def get_default_values():
    """
    Get default (median) values for optional features.
    
    Some features like years_in_us, education_years, and family_income
    may not be set by the user. This function provides median values from
    the training data as defaults.
    
    Returns:
        Dictionary with default values for optional features
    """
    X_train = pd.read_parquet("fairness_artifacts/X_train.parquet")
    return {
        'years_in_us': float(X_train['years_in_us'].median()),
        'education_years': float(X_train['education_years'].median()),
        'family_income': float(X_train['family_income'].median())
    }

# Load everything we need
model = load_model()
explainer = load_explainer()
feature_columns = load_feature_template()
model_feature_order = get_model_feature_order()
default_values = get_default_values()

# Maps for converting user inputs to model codes (from MEPS data)
SEX_MAP = {"Male": 1, "Female": 2}
RACE_ETH_MAP = {
    "Hispanic": 1,
    "White": 2,
    "Black": 3,
    "Asian": 4,
    "Other/multiple": 5
}
POVERTY_MAP = {
    "Poor": 1,
    "Low income": 2,
    "Middle income": 3,
    "High income": 4,
    "Unclassifiable": 5
}
INSURANCE_MAP = {
    "Any private": 1,
    "Public only": 2,
    "Uninsured": 3
}
SMOKER_MAP = {
    "Never smoked": 1,
    "Former smoker": 2,
    "Current smoker": 3
}
REGION_MAP = {
    "Northeast": 1,
    "Midwest": 2,
    "South": 3,
    "West": 4
}

def create_feature_vector(age, sex, race_ethnicity, poverty_category, 
                         insurance_coverage, region, hypertension, diabetes, 
                         asthma, smoker, years_in_us=None, education_years=None, 
                         family_income=None):
    """
    Convert user inputs into a feature vector matching the model's expected format.
    
    This function takes all the user inputs from the Streamlit sidebar and converts
    them into a pandas DataFrame with the exact same structure as the training data.
    This includes one-hot encoding categorical variables and matching the exact
    feature order expected by the LightGBM model.
    
    Args:
        age: Patient age (0-90)
        sex: "Male" or "Female"
        race_ethnicity: One of ["Hispanic", "White", "Black", "Asian", "Other/multiple"]
        poverty_category: Poverty level category
        insurance_coverage: Insurance type
        region: Geographic region
        hypertension: Boolean for hypertension diagnosis
        diabetes: Boolean for diabetes diagnosis
        asthma: Boolean for asthma diagnosis
        smoker: Smoking status
        years_in_us: Optional, defaults to median if not provided
        education_years: Optional, defaults to median if not provided
        family_income: Optional, defaults to median if not provided
    
    Returns:
        pandas DataFrame with one row containing the feature vector
    """
    if years_in_us is None:
        years_in_us = default_values['years_in_us']
    if education_years is None:
        education_years = default_values['education_years']
    if family_income is None:
        family_income = default_values['family_income']
    
    # Start with all zeros, then fill in the values
    # Had issues with feature order mismatch before, so being careful here
    features_dict = {col: 0.0 for col in model_feature_order}
    features = pd.DataFrame([features_dict], columns=model_feature_order, dtype=float)
    
    # Set numeric features
    features['age'] = float(age)
    features['years_in_us'] = float(years_in_us)
    features['education_years'] = float(education_years)
    features['family_income'] = float(family_income)
    
    # One-hot encode categorical features
    sex_code = SEX_MAP[sex]
    features[f'sex_{sex_code}'] = 1.0
    
    race_code = RACE_ETH_MAP[race_ethnicity]
    features[f'race_ethnicity_{race_code}'] = 1.0
    
    # Hispanic is separate from race_ethnicity in MEPS
    if race_ethnicity == "Hispanic":
        features['hispanic_1'] = 1.0
    else:
        features['hispanic_2'] = 1.0
    
    pov_code = POVERTY_MAP[poverty_category]
    features[f'poverty_category_{pov_code}'] = 1.0
    
    ins_code = INSURANCE_MAP[insurance_coverage]
    features[f'insurance_coverage_{ins_code}'] = 1.0
    
    # Chronic conditions: 1 = Yes, 2 = No (MEPS coding)
    features[f'hypertension_dx_{1 if hypertension else 2}'] = 1.0
    features[f'diabetes_dx_{1 if diabetes else 2}'] = 1.0
    features[f'asthma_dx_{1 if asthma else 2}'] = 1.0
    
    smoker_code = SMOKER_MAP[smoker]
    features[f'smoker_{smoker_code}'] = 1.0
    
    region_code = REGION_MAP[region]
    features[f'region_{region_code}'] = 1.0
    
    # keeping some features not in the sidebar
    # These don't seem to matter much but model expects them
    features['race_simple_2'] = 1.0
    features['insurance_category_1'] = 1.0
    features['born_in_usa_1'] = 1.0
    features['coronary_hd_dx_2'] = 1.0  # No coronary HD
    
    # float type
    features = features.astype(float)
    
    return features

def get_risk_category(probability):
    """
    Categorize predicted probability into risk levels.
    
    Args:
        probability: Predicted probability of hospitalization (0-1)
    
    Returns:
        Tuple of (risk_category_string, emoji) for display
    """
    if probability < 0.1:
        return "Low", "🟢"
    elif probability < 0.25:
        return "Medium", "🟡"
    else:
        return "High", "🔴"

def generate_shap_summary(shap_values, feature_names, top_n=8):
    """
    Generate a human-readable text summary of SHAP values.
    
    Identifies the top features that increase risk and decrease risk,
    then formats them into a natural language explanation.
    
    Args:
        shap_values: Array of SHAP values for each feature
        feature_names: List of feature names (should be human-readable)
        top_n: Number of top features to include in summary
    
    Returns:
        String summary of what's driving the prediction
    """
    shap_df = pd.DataFrame({
        'feature': feature_names,
        'shap_value': shap_values
    })
    
    shap_df['abs_shap'] = shap_df['shap_value'].abs()
    shap_df = shap_df.sort_values('abs_shap', ascending=False)
    
    # Top positive (increase risk)
    top_positive = shap_df[shap_df['shap_value'] > 0].head(3)
    # Top negative (decrease risk)
    top_negative = shap_df[shap_df['shap_value'] < 0].head(3)
    
    summary_parts = []
    
    if len(top_positive) > 0:
        pos_features = top_positive['feature'].tolist()
        summary_parts.append(f"**Risk is higher** mainly due to: {', '.join(pos_features)}.")
    
    if len(top_negative) > 0:
        neg_features = top_negative['feature'].tolist()
        summary_parts.append(f"**Risk is lower** due to: {', '.join(neg_features)}.")
    
    return " ".join(summary_parts)

def pretty_feature_name(col: str) -> str:
    """
    Convert internal feature names to human-readable labels.
    
    Model features use codes like 'sex_1', 'race_ethnicity_2', etc.
    This function converts them to readable names like 'Sex: Male',
    'Race/ethnicity: White', etc. for display in the UI.
    
    Args:
        col: Internal feature column name
    
    Returns:
        Human-readable feature name
    """
    base_label_map = {
        "age": "Age",
        "education_years": "Years of education",
        "family_income": "Family income",
        "years_in_us": "Years in U.S."
    }
    if col in base_label_map:
        return base_label_map[col]
    
    # Sex
    if col.startswith("sex_"):
        code = int(col.split("_")[-1])
        sex_map = {1: "Male", 2: "Female"}
        return f"Sex: {sex_map.get(code, code)}"
    
    # Insurance coverage
    if col.startswith("insurance_coverage_"):
        code = int(col.split("_")[-1])
        ins_map = {1: "Any private", 2: "Public only", 3: "Uninsured"}
        return f"Insurance: {ins_map.get(code, code)}"
    
    # Race / ethnicity
    if col.startswith("race_ethnicity_"):
        code = int(col.split("_")[-1])
        race_eth_map = {1: "Hispanic", 2: "White", 3: "Black", 4: "Asian", 5: "Other/multiple"}
        return f"Race/ethnicity: {race_eth_map.get(code, code)}"
    
    # Poverty category
    if col.startswith("poverty_category_"):
        code = int(col.split("_")[-1])
        pov_map = {1: "Poor", 2: "Low income", 3: "Middle income", 4: "High income", 5: "Unclassifiable"}
        return f"Poverty: {pov_map.get(code, code)}"
    
    # Region
    if col.startswith("region_"):
        code = int(col.split("_")[-1])
        region_map = {1: "Northeast", 2: "Midwest", 3: "South", 4: "West"}
        return f"Region: {region_map.get(code, f'Region {code}')}"
    
    # Smoker
    if col.startswith("smoker_"):
        code = int(col.split("_")[-1])
        smoker_map = {1: "Never smoked", 2: "Former smoker", 3: "Current smoker"}
        return f"Smoker: {smoker_map.get(code, f'Code {code}')}"
    
    # Chronic conditions (binary yes/no encoded as _1/_2)
    binary_dx_map = {
        "hypertension_dx": {1: "Yes", 2: "No"},
        "coronary_hd_dx": {1: "Yes", 2: "No"},
        "asthma_dx": {1: "Yes", 2: "No"},
        "diabetes_dx": {1: "Yes", 2: "No"},
    }
    for prefix, mapping in binary_dx_map.items():
        if col.startswith(prefix + "_"):
            code = int(col.split("_")[-1])
            label = prefix.replace("_dx", "").replace("_", " ").title()
            return f"{label}: {mapping.get(code, code)}"
    
    # Hispanic
    if col.startswith("hispanic_"):
        code = int(col.split("_")[-1])
        hisp_map = {1: "Hispanic", 2: "Not Hispanic"}
        return f"Hispanic: {hisp_map.get(code, code)}"
    
    # Born in USA
    if col.startswith("born_in_usa_"):
        code = int(col.split("_")[-1])
        born_map = {1: "Yes", 2: "No"}
        return f"Born in USA: {born_map.get(code, code)}"
    
    # Race simple
    if col.startswith("race_simple_"):
        code = int(col.split("_")[-1])
        race_simple_map = {1: "White", 2: "Black", 3: "American Indian/Alaska Native", 
                          4: "Asian", 5: "Native Hawaiian/Pacific Islander", 6: "Multiple"}
        return f"Race: {race_simple_map.get(code, f'Code {code}')}"
    
    # Insurance category
    if col.startswith("insurance_category_"):
        code = int(col.split("_")[-1])
        return f"Insurance Category: Code {code}"
    
    # Fallback: just replace underscores
    return col.replace("_", " ").title()

# Main app
st.title("🏥 Healthcare Prediction Playground")

# Sidebar with all the input controls
st.sidebar.markdown("## Patient Profile")
st.sidebar.markdown("Set the patient characteristics here.")

age = st.sidebar.slider("Age", 0, 90, 45)
sex = st.sidebar.selectbox("Sex", ["Male", "Female"])
race_ethnicity = st.sidebar.selectbox(
    "Race/Ethnicity",
    ["Hispanic", "White", "Black", "Asian", "Other/multiple"]
)
poverty_category = st.sidebar.selectbox(
    "Poverty Category",
    ["Poor", "Low income", "Middle income", "High income", "Unclassifiable"]
)
insurance_coverage = st.sidebar.selectbox(
    "Insurance Type",
    ["Any private", "Public only", "Uninsured"]
)

region = st.sidebar.selectbox(
    "Region",
    ["Northeast", "Midwest", "South", "West"],
    index=2  # Default to South
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Chronic Conditions")
hypertension = st.sidebar.checkbox("Hypertension")
diabetes = st.sidebar.checkbox("Diabetes")
asthma = st.sidebar.checkbox("Asthma")

smoker = st.sidebar.selectbox(
    "Smoking Status",
    ["Never smoked", "Former smoker", "Current smoker"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Socioeconomic")
education_years = st.sidebar.slider(
    "Education (Years)",
    0, 20, int(default_values['education_years']),
    help="Years of education completed"
)
family_income = st.sidebar.number_input(
    "Family Income ($)",
    min_value=0,
    value=int(default_values['family_income']),
    step=1000,
    help="Annual family income in dollars"
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Model Settings")

threshold = st.sidebar.slider(
    "Classification Threshold",
    0.1, 0.9, 0.5, 0.1,
    format="%.1f",
    help="Used to convert the predicted probability into a Hospitalized/Not Hospitalized label."
)

# Main content area
st.markdown("**Local Explainability** — Interact with the model one person at a time")
st.markdown("---")

# Build the feature vector from user inputs
feature_vector = create_feature_vector(
        age=age,
        sex=sex,
        race_ethnicity=race_ethnicity,
        poverty_category=poverty_category,
        insurance_coverage=insurance_coverage,
        region=region,
        hypertension=hypertension,
        diabetes=diabetes,
        asthma=asthma,
        smoker=smoker,
        education_years=education_years,
        family_income=family_income
)

# Step 1: Get the prediction
st.header("Step 1: Model Prediction")

# Makeing sure features are in the right order 
feature_vector_aligned = feature_vector[model_feature_order].copy()
pred_proba = model.predict_proba(feature_vector_aligned)[0, 1]
pred_risk = pred_proba * 100
risk_category, risk_emoji = get_risk_category(pred_proba)

# Quick explanation
with st.expander("ℹ️ What does 'Predicted Hospitalization Risk' mean?"):
        st.markdown(f"""
**Predicted Hospitalization Risk** is the model's estimate of the probability that a person with these characteristics 
will be hospitalized (have inpatient expenditures > $0) in the given year.

- **Risk Percentage**: The probability expressed as a percentage (0–100%)
- **Risk Categories**:
  - **Low**: &lt; 10% risk  
  - **Medium**: 10–25% risk  
  - **High**: &gt; 25% risk  
- **Prediction label**: The app uses the classification threshold you set in the sidebar  
  (currently **{threshold*100:.0f}%**) to turn the probability into **Hospitalized** vs **Not Hospitalized**.

This model was trained on MEPS (Medical Expenditure Panel Survey) data using demographic, socioeconomic, 
and health status features.
        """)


st.markdown("") 
col1, col2, col3 = st.columns(3)
with col1:
        st.metric("Predicted Hospitalization Risk", f"{pred_risk:.1f}%")
with col2:
        st.metric("Risk Category", f"{risk_emoji} {risk_category}")
with col3:
        pred_label = "Hospitalized" if pred_proba >= threshold else "Not Hospitalized"
        st.metric(
            "Prediction",
            pred_label,
            help=f"Uses a {threshold*100:.0f}% threshold on the predicted probability."
        )

st.markdown("---")

# Step 2: SHAP explanation
st.header("Step 2: Local SHAP Explanation")
st.markdown("**See which features matter most for this prediction**")
st.markdown("")  # Spacing

# Compute SHAP values
shap_values_raw = explainer.shap_values(feature_vector_aligned)
# LightGBM returns a list for binary classification, need the positive class
if isinstance(shap_values_raw, (list, tuple)):
        shap_values = shap_values_raw[1][0]
        base_value = explainer.expected_value[1] if isinstance(explainer.expected_value, (list, tuple)) else explainer.expected_value
else:
        shap_values = shap_values_raw[0]
        base_value = explainer.expected_value

# Clean up feature names for display
feature_names = feature_vector_aligned.columns.tolist()
pretty_feature_names = [pretty_feature_name(f) for f in feature_names]

# Create SHAP explanation object
shap_explanation = shap.Explanation(
        values=shap_values,
        base_values=base_value,
        data=feature_vector_aligned.iloc[0].values,
        feature_names=pretty_feature_names
)

# Display SHAP waterfall
st.subheader("SHAP Waterfall Plot (Top 10 Features)")

with st.expander("ℹ️ How to read this chart"):
        st.markdown("""
The waterfall chart shows how each feature contributes to move the prediction from the **expected value** (base value) to this specific patient's prediction.

- **Left (base value)**: The model's average output across all training examples (expected value)
- **Middle bars**: Each feature's SHAP contribution - how much it pushes the prediction away from the base value
  - Red bars: Positive contributions (increase risk)
  - Blue bars: Negative contributions (decrease risk)
- **Right (final value)**: The sum of base value + all SHAP contributions = this patient's predicted risk

The features are ordered by the magnitude of their contribution, with the most influential features at the top. The bars accumulate from left to right, showing how each feature moves the prediction toward the final value.
        """)


st.caption("Features are ordered by their impact on the prediction. Red bars increase risk, blue bars decrease risk.")
st.markdown("")  # Spacing

try:
        # Try to use SHAP's waterfall plot
        plt.style.use('default')
        shap.plots.waterfall(shap_explanation, max_display=10, show=False)
        
        fig = plt.gcf()
        
        # Make it look nicer
        fig.patch.set_facecolor('white')
        fig.set_size_inches(14, 8)
        
        # Fix font sizes
        for ax in fig.get_axes():
            ax.set_facecolor('white')
            for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
                try:
                    item.set_fontsize(11)
                except:
                    pass
        
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
except Exception as e:
        # If waterfall fails, make a simple bar chart instead
        st.warning(f"Waterfall plot unavailable. Showing bar plot instead. Error: {str(e)}")
        # Get top features by absolute SHAP value
        shap_df = pd.DataFrame({
            'feature': feature_names,
            'pretty_name': pretty_feature_names,
            'shap_value': shap_values
        })
        shap_df['abs_shap'] = shap_df['shap_value'].abs()
        top_features = shap_df.nlargest(10, 'abs_shap')
        
        fig, ax = plt.subplots(figsize=(12, 8))
        colors = ['#d32f2f' if x < 0 else '#1976d2' for x in top_features['shap_value']]
        ax.barh(range(len(top_features)), top_features['shap_value'], color=colors, alpha=0.8)
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features['pretty_name'].tolist(), fontsize=11)
        ax.set_xlabel('SHAP Value', fontsize=12, fontweight='500')
        ax.set_title('Top 10 Features by SHAP Value', fontsize=14, fontweight='600', pad=15)
        ax.axvline(x=0, color='#666', linestyle='-', linewidth=1)
        ax.grid(axis='x', alpha=0.2, linestyle='--')
        ax.set_facecolor('white')
        fig.patch.set_facecolor('white')
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

# Textual summary
st.markdown("")  # Spacing
st.subheader("Summary")

with st.expander("ℹ️ Understanding the Summary"):
        st.markdown("""
This summary highlights the **top 3 features** that are most responsible for increasing or decreasing 
this patient's predicted hospitalization risk.

- **Features that increase risk** (positive SHAP values): These characteristics push the prediction 
  higher than the average patient, making hospitalization more likely.
  
- **Features that decrease risk** (negative SHAP values): These characteristics push the prediction 
  lower than the average patient, making hospitalization less likely.

The features are ranked by the magnitude of their SHAP contribution - the features listed have the 
strongest impact on this specific prediction. Note that the same feature might affect different 
patients differently depending on their other characteristics.
        """)

summary_text = generate_shap_summary(shap_values, pretty_feature_names)
if summary_text:
        st.info(summary_text)
else:
        st.info("No significant feature contributions identified.")

# Step 3: What-if analysis
st.header("Step 3: What If? Analysis")
st.markdown("**Explore counterfactual scenarios: change features and see how the prediction changes**")

with st.expander("ℹ️ What is counterfactual analysis?"):
        st.markdown("""
**Counterfactual analysis** (also called "What If?" analysis) lets you explore how changing specific features 
would affect the model's prediction, while keeping everything else the same.

This is useful for:
- Understanding which features have the biggest impact on predictions
- Exploring fairness: seeing how demographic changes affect risk
- Testing scenarios: "What if this person had better insurance?" or "What if they were 10 years older?"

The comparison shows you the original risk, the new risk after changes, and the difference between them.
        """)

st.markdown("Change some features below and see how the prediction changes:")

col1, col2 = st.columns(2)

with col1:
        whatif_insurance = st.selectbox(
            "Change Insurance Type",
            ["Any private", "Public only", "Uninsured"],
            index=INSURANCE_MAP[insurance_coverage] - 1,
            key="whatif_ins"
        )
        
        whatif_sex = st.selectbox(
            "Change Sex",
            ["Male", "Female"],
            index=0 if sex == "Male" else 1,
            key="whatif_sex"
        )
        
        whatif_hypertension = st.checkbox(
            "Toggle Hypertension",
            value=hypertension,
            key="whatif_htn"
        )

with col2:
        whatif_age = st.slider(
            "Change Age",
            0, 90, age,
            key="whatif_age"
        )
        
        whatif_poverty = st.selectbox(
            "Change Poverty Category",
            ["Poor", "Low income", "Middle income", "High income", "Unclassifiable"],
            index=["Poor", "Low income", "Middle income", "High income", "Unclassifiable"].index(poverty_category),
            key="whatif_pov"
        )

# Create what-if feature vector
whatif_vector = create_feature_vector(
        age=whatif_age,
        sex=whatif_sex,
        race_ethnicity=race_ethnicity,
        poverty_category=whatif_poverty,
        insurance_coverage=whatif_insurance,
        region=region,
        hypertension=whatif_hypertension,
        diabetes=diabetes,
        asthma=asthma,
        smoker=smoker,
        education_years=education_years,
        family_income=family_income
)

# Get the new prediction
whatif_vector_aligned = whatif_vector[model_feature_order].copy()
whatif_proba = model.predict_proba(whatif_vector_aligned)[0, 1]
whatif_risk = whatif_proba * 100
risk_change = whatif_risk - pred_risk

# Show the comparison
st.subheader("Prediction Comparison")

comparison_col1, comparison_col2, comparison_col3 = st.columns(3)
with comparison_col1:
        st.metric("Original Risk", f"{pred_risk:.1f}%")
with comparison_col2:
        st.metric("New Risk", f"{whatif_risk:.1f}%", delta=f"{risk_change:+.1f}%")
with comparison_col3:
        change_pct = ((whatif_risk - pred_risk) / pred_risk * 100) if pred_risk > 0 else 0
        st.metric("Relative Change", f"{change_pct:+.1f}%")

# Build a summary of what changed
changes = []
if whatif_insurance != insurance_coverage:
        changes.append(f"insurance from {insurance_coverage} to {whatif_insurance}")
if whatif_sex != sex:
        changes.append(f"sex from {sex} to {whatif_sex}")
if whatif_hypertension != hypertension:
        changes.append(f"hypertension from {'Yes' if hypertension else 'No'} to {'Yes' if whatif_hypertension else 'No'}")
if whatif_age != age:
        changes.append(f"age from {age} to {whatif_age}")
if whatif_poverty != poverty_category:
        changes.append(f"poverty from {poverty_category} to {whatif_poverty}")

if changes:
        change_text = " and ".join(changes)
        st.info(f"**If this person had** {change_text}, **risk would change from {pred_risk:.1f}% → {whatif_risk:.1f}%**.")

st.markdown("---")

# Step 4: ICE curve for age
st.header("Step 4: ICE Curve - Age Effect")
st.markdown("**See how risk changes as age changes (keeping everything else the same)**")
st.markdown("")  # Spacing

# Generate predictions for different ages
age_range = np.arange(0, 91, 5)
ice_probs = []

for a in age_range:
        ice_vector = create_feature_vector(
            age=a,
            sex=sex,
            race_ethnicity=race_ethnicity,
            poverty_category=poverty_category,
            insurance_coverage=insurance_coverage,
            region=region,
            hypertension=hypertension,
            diabetes=diabetes,
            asthma=asthma,
            smoker=smoker,
            education_years=education_years,
            family_income=family_income
        )
        ice_vector_aligned = ice_vector[model_feature_order].copy()
        prob = model.predict_proba(ice_vector_aligned)[0, 1]
        ice_probs.append(prob * 100)

# Make the plot with Plotly
fig = go.Figure()

# Main line
fig.add_trace(go.Scatter(
        x=age_range,
        y=ice_probs,
        mode='lines+markers',
        name='Predicted Risk',
        line=dict(color='#1976d2', width=3),
        marker=dict(size=4, color='#1976d2'),
        fill='tozeroy',
        fillcolor='rgba(25, 118, 210, 0.15)',
        hovertemplate='<b>Age:</b> %{x} years<br><b>Predicted Risk:</b> %{y:.2f}%<extra></extra>'
))

# Mark where the current age is
fig.add_vline(
        x=age,
        line_dash="dash",
        line_color="#d32f2f",
        line_width=2,
        annotation_text=f"Current Age: {age} years",
        annotation_position="top",
        annotation=dict(
            font_size=12,
            font_color='#d32f2f',
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor='#d32f2f',
            borderwidth=1
        )
)

# Mark the current risk level
fig.add_hline(
        y=pred_risk,
        line_dash="dash",
        line_color="#d32f2f",
        line_width=2,
        opacity=0.8,
        annotation_text=f"Current Risk: {pred_risk:.1f}%",
        annotation_position="right",
        annotation=dict(
            font_size=12,
            font_color='#d32f2f',
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor='#d32f2f',
            borderwidth=1
        )
)

# Style the plot
fig.update_layout(
        title=dict(
            text='Individual Conditional Expectation (ICE) Curve for Age',
            font=dict(size=18, color='#000000'),
            x=0.5,
            xanchor='center'
        ),
        xaxis=dict(
            title='Age (years)',
            title_font=dict(size=13, color='#000000'),
            tickfont=dict(size=11, color='#000000'),
            showgrid=True,
            gridcolor='rgba(0,0,0,0.1)',
            gridwidth=1,
            zeroline=False
        ),
        yaxis=dict(
            title='Predicted Hospitalization Risk (%)',
            title_font=dict(size=13, color='#000000'),
            tickfont=dict(size=11, color='#000000'),
            showgrid=True,
            gridcolor='rgba(0,0,0,0.1)',
            gridwidth=1,
            ticksuffix='%',
            zeroline=False
        ),
        hovermode='x unified',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=12, color='#000000'),
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor='rgba(0,0,0,0.3)',
            borderwidth=1
        ),
        height=550,
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=60, r=60, t=80, b=60)
)

st.plotly_chart(fig, use_container_width=True)

st.markdown("")
with st.expander("ℹ️ What is an ICE Curve?"):
        st.markdown(f"""
**Individual Conditional Expectation (ICE)** curves show how the model's prediction changes as a single feature varies, 
while all other features remain constant at their current values.

- **X-axis**: Age (0-90 years)
- **Y-axis**: Predicted hospitalization risk (%)
- **Blue line**: Shows how risk changes across different ages for this specific patient profile
- **Red dashed line**: Marks your current age ({age} years) and corresponding risk ({pred_risk:.1f}%)

This helps you understand how age affects predictions for someone with your exact combination of other characteristics 
(sex, race, insurance, health conditions, etc.). Unlike partial dependence plots which show average effects across all 
patients, ICE curves show the effect for this specific individual.
        """)

st.markdown(f"**Interpretation:** The curve shows how hospitalization risk changes with age while keeping all other features constant. Your current age ({age} years) corresponds to a predicted risk of {pred_risk:.1f}%.")

