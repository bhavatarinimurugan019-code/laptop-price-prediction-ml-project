
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(
    page_title="Laptop Price Predictor",
    page_icon="💻",
    layout="wide"
)

st.title("Laptop Price Prediction")
st.write("Machine Learning project using a Kaggle dataset.")
st.divider()

uploaded = st.file_uploader(
    "Upload your laptop CSV dataset",
    type=["csv"]
)

if uploaded is None:
    st.info("Upload your Kaggle CSV file to begin.")
    st.stop()

# Try common CSV separators
df = None
for sep in [",", ";", "\t", "|"]:
    try:
        uploaded.seek(0)
        temp = pd.read_csv(
            uploaded,
            sep=sep,
            encoding="latin1",
            on_bad_lines="error"
        )
        if temp.shape[1] > 1:
            df = temp
            break
    except Exception:
        continue

if df is None:
    st.error(
        "Cannot read this CSV correctly. "
        "Please download the original CSV from Kaggle."
    )
    st.stop()

df.columns = df.columns.astype(str).str.strip()

st.subheader("1. Dataset overview")
c1, c2, c3 = st.columns(3)
c1.metric("Rows", df.shape[0])
c2.metric("Columns", df.shape[1])
c3.metric("Missing cells", int(df.isna().sum().sum()))

st.write("Dataset preview")
st.dataframe(df.head(10), use_container_width=True)

st.write("Missing values in each column")
st.dataframe(
    df.isna().sum().rename("Missing values"),
    use_container_width=True
)

# Detect likely price columns
price_cols = [
    c for c in df.columns
    if any(word in c.lower()
           for word in ["price", "cost", "eur", "amount"])
]

numeric_cols = df.select_dtypes(
    include=np.number
).columns.tolist()

possible_targets = [
    c for c in price_cols if c in numeric_cols
]

if not possible_targets:
    possible_targets = numeric_cols

if not possible_targets:
    st.error(
        "No numeric target column found. "
        "Check the CSV format and column names."
    )
    st.write("Detected columns:", df.columns.tolist())
    st.stop()

st.subheader("2. Select the price column")

default_target = possible_targets[0]
target = st.selectbox(
    "Which column contains laptop prices?",
    possible_targets,
    index=possible_targets.index(default_target)
)

# Convert selected target to numbers
data = df.copy()
data[target] = pd.to_numeric(
    data[target].astype(str).str.replace(",", "", regex=False),
    errors="coerce"
)

data = data.dropna(subset=[target])
data = data.replace([np.inf, -np.inf], np.nan)

# Remove rows with invalid target values
X = data.drop(columns=[target])
y = data[target]

# Remove columns that are entirely empty or constant
X = X.dropna(axis=1, how="all")
X = X.loc[:, X.nunique(dropna=True) > 1]

# Avoid training on ID-like columns
id_cols = [
    c for c in X.columns
    if c.lower() in ["id", "index", "unnamed: 0"]
]
X = X.drop(columns=id_cols, errors="ignore")

if len(X) < 10:
    st.error(
        "Not enough valid rows for training. "
        "Please check the original dataset."
    )
    st.stop()

if X.shape[1] == 0:
    st.error("No useful feature columns found.")
    st.stop()

# Separate numeric and categorical features
numeric_features = X.select_dtypes(
    include=np.number
).columns.tolist()

categorical_features = X.select_dtypes(
    exclude=np.number
).columns.tolist()

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

transformers = []

if numeric_features:
    transformers.append(
        ("num", numeric_pipeline, numeric_features)
    )

if categorical_features:
    transformers.append(
        ("cat", categorical_pipeline, categorical_features)
    )

preprocessor = ColumnTransformer(transformers)

model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=100,
        random_state=42
    ))
])

st.subheader("3. Train the ML model")

st.write(
    f"Valid records: {len(X)} | "
    f"Features: {X.shape[1]}"
)

if st.button("Train Model", type="primary"):
    if y.nunique() < 2:
        st.error("The price column needs at least two different values.")
        st.stop()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    st.session_state["trained_model"] = model
    st.session_state["feature_columns"] = X.columns.tolist()
    st.session_state["target_column"] = target
    st.session_state["numeric_features"] = numeric_features
    st.session_state["categorical_features"] = categorical_features
    st.session_state["sample"] = X

    st.success("Model trained successfully!")

    a, b = st.columns(2)
    a.metric("Mean Absolute Error", f"{mae:,.2f}")
    b.metric("R² Score", f"{r2:.3f}")

    st.caption(
        "Lower MAE is better. R² closer to 1 generally "
        "indicates better fit. Test results depend on the dataset."
    )

    comparison = pd.DataFrame({
        "Actual price": y_test.values[:10],
        "Predicted price": predictions[:10]
    })
    st.subheader("4. Actual vs predicted prices")
    st.dataframe(comparison, use_container_width=True)

# Prediction interface
if "trained_model" in st.session_state:
    st.divider()
    st.subheader("5. Predict a laptop price")

    model = st.session_state["trained_model"]
    columns = st.session_state["feature_columns"]
    sample = st.session_state["sample"]

    with st.form("prediction_form"):
        user_values = {}

        for col in columns:
            if col in st.session_state["numeric_features"]:
                valid_values = pd.to_numeric(
                    sample[col], errors="coerce"
                ).dropna()

                default = (
                    float(valid_values.median())
                    if not valid_values.empty else 0.0
                )

                user_values[col] = st.number_input(
                    col,
                    value=default
                )
            else:
                choices = sample[col].dropna().astype(str).unique().tolist()

                if choices:
                    user_values[col] = st.selectbox(
                        col, choices
                    )
                else:
                    user_values[col] = st.text_input(col)

        submitted = st.form_submit_button("Predict Price")

    if submitted:
        input_df = pd.DataFrame([user_values], columns=columns)
        result = model.predict(input_df)[0]

        st.success(f"Estimated laptop price: {result:,.2f}")
        st.caption(
            "This is a model estimate, not a guaranteed market price."
        )
