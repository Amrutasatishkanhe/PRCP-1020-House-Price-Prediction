# House Price Prediction using Machine Learning

## Project Overview
This project predicts house sale prices using machine learning regression techniques. It uses housing data from Ames, Iowa, to explore property features and estimate sale prices.

## Objectives
- Perform exploratory data analysis (EDA).
- Clean and preprocess housing data.
- Train and compare regression models.
- Evaluate model performance using appropriate metrics.
- Identify important features influencing house prices.

## Dataset
- Target variable: `SalePrice`
- Dataset contains 1,460 records and 81 columns before preprocessing.
- The data includes property characteristics such as overall quality, living area, garage capacity, and basement area.

## Models
- Gradient Boosting Regressor
- Random Forest Regressor

Gradient Boosting performed better than Random Forest in the project's baseline comparison.

## Evaluation
The models were evaluated using:
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² score
- Cross-validation

## Feature Importance
The most influential features in the model included:
- Overall Quality (`OverallQual`)
- Above-ground Living Area (`GrLivArea`)
- Garage Capacity (`GarageCars`)
- Total Basement Area (`TotalBsmtSF`)

## Tools and Technologies
- Python
- Jupyter Notebook
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- Joblib

## How to Run
1. Download or clone this repository.
2. Install the required Python libraries.
3. Ensure the dataset is available at the path expected by the notebook.
4. Open `PRCP-1020_House_Price_Prediction.ipynb` in Jupyter Notebook.
5. Run the notebook cells in order.

## Project Outcome
The project demonstrates an end-to-end machine learning regression workflow, including data analysis, model comparison, evaluation, and feature importance analysis.

## Author
Amruta Satish Kanhe
