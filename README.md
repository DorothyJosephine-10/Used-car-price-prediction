# Used Car Price Prediction

## Project Overview

This project provides an end-to-end machine learning pipeline for predicting used car prices. It includes raw data handling, preprocessing, feature engineering, model training, evaluation, 
and an interactive Streamlit-based user interface with visual enhancements (background image and input bar sizing).

---

## Project Structure

- **Car_sales.ipynb** — Jupyter notebook containing data inspection, cleaning, feature engineering, model training, and evaluation.
- **combined_df.csv** — Raw dataset containing used car details. All preprocessing and transformations are performed inside `Car_sales.ipynb`.
- **encoders.pkl** — Serialized encoders used for categorical feature transformation.
- **Randomforest_regression.pkl** — Trained Random Forest regression model.
- **Final_prediction.csv** — Output file containing predicted prices for test or new data.
- **UI.py** — Streamlit application for interactive used car price prediction.
- **Background-image.jpg** & **Car_image-removebg-preview.png** — Visual assets used in the Streamlit dashboard.

---

## Data Processing & Model Training
- Load the raw dataset from `combined_df.csv`
- Handle missing and inconsistent values
- Encode categorical features
- Perform feature engineering
- Split the dataset into training and testing sets

### Model Development & Selection
- Train multiple regression models:
  - Linear Regression
  - Decision Tree Regression
  - Random Forest Regression
  - Gradient Boosting Regression
- Compare model performance using MAE, MSE, RMSE, and R² score
- Identify Random Forest Regression as the best-performing model

### Model Tuning & Final Selection
- Perform hyperparameter tuning on the Random Forest model
- Observe that the tuned model resulted in lower performance
- Use the baseline Random Forest model for final predictions based on better evaluation metrics
- Save the final selected model and encoders for reuse

---

## Model Artifacts (Not Included in Repository)

The following file is generated during model training but is **not included** in this repository due to GitHub's file size limitations:

- `Car_price_artifacts.pkl`

This file contains trained model artifacts and preprocessing objects required for prediction.

### How to Generate
1. Open and run `Car_sales.ipynb` completely.
2. The artifact file will be created automatically in the project directory.
3. Ensure it is available before running `UI.py`.

---

## Requirements

Install the required Python libraries:

pip install pandas numpy scikit-learn matplotlib seaborn streamlit

# Conclusion

This project demonstrates a complete machine learning workflow for predicting used car prices, from raw data preprocessing and feature engineering to model training, evaluation, and deployment via a Streamlit interface. The Random Forest regression model provided the most nearest predictions, while the Streamlit dashboard—with visual enhancements like a background image and resized input fields—offers an intuitive and user-friendly interface for interacting with the model.
