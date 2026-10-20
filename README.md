
# LaptopIQ — Laptop Price Prediction

An interactive machine learning project that explores laptop specifications,
trains regression models, evaluates prediction errors, and estimates laptop prices.

## Features
- CSV dataset upload and exploration
- Missing-value preprocessing
- Random Forest and Extra Trees regression
- MAE and R² model evaluation
- Actual vs predicted comparisons
- Interactive laptop price estimates
- Downloadable test predictions

## Tech Stack
- Python
- Pandas and NumPy
- Scikit-learn
- Streamlit

## Run Locally

1. Clone or download this repository.
2. Install the dependencies:

   `python -m pip install -r requirements.txt`

3. Start the application:

   `python -m streamlit run app.py`

4. Upload a valid laptop dataset CSV.
5. Select the price column and train the model.

## Dataset
Upload a laptop dataset from Kaggle that contains a numeric price target
and relevant laptop specifications.

## Limitations
Predictions depend on the quality, size, and representativeness of the
uploaded dataset. This project is educational and not a guaranteed pricing tool.
