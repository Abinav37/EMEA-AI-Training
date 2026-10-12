# E-commerce Customer Data Analysis

This project analyzes a reconstructed e-commerce customer dataset. The goal is to inspect the available data, perform basic cleaning, explore patterns in the data, and build a simple linear regression model to predict yearly spending.

## Work completed

- Dataset inspection using Pandas
- Basic data cleaning
- Exploratory Data Analysis using Matplotlib and Seaborn
- Simple linear regression predicting Yearly Amount Spent from Length of Membership
- Model evaluation using MAE and R²

## Dataset

The dataset is located at `data/ecommerce_customers_reconstructed.csv`.

Important: this dataset was reconstructed from provided data fragments and its completeness has not been independently verified. It should be treated as the available reconstructed data, not as a guaranteed complete original dataset.

## How to open and run the notebook

1. Open a terminal in the repository root.
2. Create and activate a Python environment if needed.
3. Install the required packages:
   `pip install -r 03-eda/requirements.txt`
4. Start Jupyter Notebook:
   `jupyter notebook`
5. Open `03-eda/notebooks/customer_eda.ipynb` and run the cells from top to bottom.

This project is intended as a basic data analysis workflow using the cleaned DataFrame from the notebook. No additional model tuning or advanced analysis was added beyond the tasks listed above.
