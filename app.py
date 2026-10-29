
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# --------------------------------------------------
# 1. Project paths and page configuration
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "house_price_gradient_boosting.joblib"
DATA_PATH = BASE_DIR / "Data" / "data.csv"
IMPORTANCE_PATH = BASE_DIR / "outputs" / "feature_importance_xai.csv"

st.set_page_config(
    page_title="Smart House Price Predictor",
    page_icon="🏠",
    layout="wide"
)

# --------------------------------------------------
# 2. Load model and dataset
# --------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


st.title("🏠 Smart House Price Predictor")
st.write(
    "Estimate a house sale price using a trained "
    "Gradient Boosting machine learning model."
)

try:
    model = load_model()
    df = load_data()
    features = list(model.feature_names_in_)

except Exception as e:
    st.error(f"Could not load the model or dataset: {e}")
    st.stop()

if "SalePrice" not in df.columns:
    st.error("The dataset must contain the SalePrice target column.")
    st.stop()

missing_features = [col for col in features if col not in df.columns]

if missing_features:
    st.error(f"Dataset is missing required features: {missing_features}")
    st.stop()

# --------------------------------------------------
# 3. Create sensible default property values
# --------------------------------------------------
defaults = {}

for col in features:
    if pd.api.types.is_numeric_dtype(df[col]):
        defaults[col] = df[col].median()
    else:
        modes = df[col].mode(dropna=True)
        defaults[col] = modes.iloc[0] if not modes.empty else None

st.info(
    "Enter the main property details below. Features not shown "
    "in the form use typical values from the dataset. The result "
    "is a model estimate, not a formal property valuation."
)

# --------------------------------------------------
# 4. Property input form
# --------------------------------------------------

with st.form("house_details"):
    st.subheader("Enter property details")

    c1, c2, c3 = st.columns(3)

    with c1:
        overall_qual = st.slider(
            "Overall quality (1–10)",
            min_value=1,
            max_value=10,
            value=max(1, min(10, int(defaults["OverallQual"]))),
            key="overall_qual"
        )

        living_area = st.number_input(
            "Living area (sq. ft.)",
            min_value=1,
            value=max(1, int(defaults["GrLivArea"])),
            step=100,
            key="living_area"
        )

        lot_area = st.number_input(
            "Lot area (sq. ft.)",
            min_value=1,
            value=max(1, int(defaults["LotArea"])),
            step=100,
            key="lot_area"
        )

    with c2:
        year_built = st.number_input(
            "Year built",
            min_value=1800,
            max_value=2026,
            value=max(1800, min(2026, int(defaults["YearBuilt"]))),
            key="year_built"
        )

        basement_area = st.number_input(
            "Basement area (sq. ft.)",
            min_value=0,
            value=max(0, int(defaults["TotalBsmtSF"])),
            step=100,
            key="basement_area"
        )

        garage_cars = st.number_input(
            "Garage capacity (cars)",
            min_value=0,
            max_value=5,
            value=max(0, min(5, int(defaults["GarageCars"]))),
            key="garage_cars"
        )

    with c3:
        bedrooms = st.number_input(
            "Bedrooms",
            min_value=0,
            max_value=10,
            value=max(0, min(10, int(defaults["BedroomAbvGr"]))),
            key="bedrooms"
        )

        full_baths = st.number_input(
            "Full bathrooms",
            min_value=0,
            max_value=5,
            value=max(0, min(5, int(defaults["FullBath"]))),
            key="full_baths"
        )

        neighborhoods = sorted(df["Neighborhood"].dropna().unique().tolist())
        default_neighborhood = defaults["Neighborhood"]

        neighborhood_index = (
            neighborhoods.index(default_neighborhood)
            if default_neighborhood in neighborhoods
            else 0
        )

        selected_neighborhood = st.selectbox(
            "Neighborhood",
            neighborhoods,
            index=neighborhood_index,
            key="selected_neighborhood"
        )

    submitted = st.form_submit_button(
        "Estimate house price",
        type="primary",
        use_container_width=True
    )

# --------------------------------------------------
# 5. Generate prediction using submitted inputs
# --------------------------------------------------
if submitted:
    row = defaults.copy()

    # Apply the current form values to the model inputs
    row["OverallQual"] = overall_qual
    row["GrLivArea"] = living_area
    row["LotArea"] = lot_area
    row["YearBuilt"] = year_built
    row["TotalBsmtSF"] = basement_area
    row["GarageCars"] = garage_cars
    row["BedroomAbvGr"] = bedrooms
    row["FullBath"] = full_baths
    row["Neighborhood"] = selected_neighborhood

    # Preserve the exact feature order expected by the model
    input_df = pd.DataFrame([row], columns=features)

    try:
        prediction = float(model.predict(input_df)[0])

        st.success("Prediction completed successfully!")

        st.metric(
            label="Estimated house sale price",
            value=f"${prediction:,.0f}"
        )

        # Show the inputs actually sent to the model
        st.subheader("Your submitted property details")

        summary = pd.DataFrame({
            "Property feature": [
                "Overall quality",
                "Living area (sq. ft.)",
                "Lot area (sq. ft.)",
                "Year built",
                "Basement area (sq. ft.)",
                "Garage capacity",
                "Bedrooms",
                "Full bathrooms",
                "Neighborhood"
            ],
            "Submitted value": [
                overall_qual,
                living_area,
                lot_area,
                year_built,
                basement_area,
                garage_cars,
                bedrooms,
                full_baths,
                selected_neighborhood
            ]
        })

        st.dataframe(summary, hide_index=True, use_container_width=True)

        # Temporary diagnostic output for our debugging
        with st.expander("Debug: model input verification"):
            st.write("Input feature count:", input_df.shape[1])
            st.write("Living area received:", input_df["GrLivArea"].iloc[0])
            st.write("Overall quality received:", input_df["OverallQual"].iloc[0])
            st.write("Neighborhood received:", input_df["Neighborhood"].iloc[0])

        st.caption(
            "This is an estimated price from a machine learning model. "
            "Actual prices depend on market conditions and other "
            "property characteristics. Unexposed features use "
            "dataset-based defaults."
        )

    except Exception as e:
        st.error(f"Prediction failed: {e}")

# --------------------------------------------------
# 6. Explainable AI: Global feature importance
# --------------------------------------------------
st.divider()
st.header("📊 What influences house prices?")

st.write(
    "This chart shows which transformed features the trained model "
    "relies on most overall. Importance scores do not represent "
    "causal effects or percentage changes in price."
)

if IMPORTANCE_PATH.exists():
    try:
        importance_df = pd.read_csv(IMPORTANCE_PATH)

        required_columns = {"Feature", "Importance"}

        if not required_columns.issubset(importance_df.columns):
            st.error(
                "The feature-importance CSV must contain "
                "'Feature' and 'Importance' columns."
            )
        elif importance_df.empty:
            st.info("The feature-importance file is empty.")
        else:
            max_features = min(15, len(importance_df))

            top_n = st.slider(
                "Number of features to display",
                min_value=1,
                max_value=max_features,
                value=min(10, max_features),
                key="importance_slider"
            )

            chart_df = (
                importance_df
                .nlargest(top_n, "Importance")
                .sort_values("Importance", ascending=True)
                .set_index("Feature")
            )

            st.bar_chart(chart_df["Importance"])

            st.caption(
                "These are global feature-importance scores. "
                "They explain overall model reliance, not the exact "
                "reason for an individual prediction."
            )

    except Exception as e:
        st.error(f"Could not display feature importance: {e}")

else:
    st.info(
        "Feature-importance CSV not found. Run the notebook's "
        "feature-importance analysis cell first."
    )

# --------------------------------------------------
# 7. Project information
# --------------------------------------------------
with st.expander("About this project"):
    st.write(
        "This project uses a trained Gradient Boosting Regressor "
        "to estimate house prices from property characteristics."
    )
    st.write(
        "Dataset: Ames, Iowa housing data. Predictions are intended "
        "for educational demonstration and are not professional "
        "real-estate valuations."
    )
