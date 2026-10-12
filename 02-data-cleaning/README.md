# Pandas Data Cleaning Exercise

## Assignment title
Pandas Data Cleaning Exercise

## Objective
This project demonstrates a practical data-cleaning workflow using Python and pandas on a small customer dataset. The notebook walks through inspection, validation, cleaning, transformation, filtering, analysis, and final dataset preparation.

## Dataset description
The dataset is a small practice dataset containing 15 customer records and 10 original columns:

- Customer_ID
- Name
- Age
- Gender
- City
- Purchase_Amount
- Discount
- Department
- Joining_Date
- Email

The dataset includes realistic data-quality issues such as inconsistent capitalization, extra whitespace, missing values, invalid numeric values, invalid dates, duplicate records, and malformed email entries.

## Technologies used
- Python
- pandas
- NumPy
- Jupyter Notebook
- VS Code

## Work completed in the notebook
The notebook contains the full exercise from Part A through Part I and the Final Challenge.

### Part A – Initial data exploration
- Inspect records, columns, data types, and missing values
- Review row and column counts
- Summarize numerical fields and counts of unique values

### Part B – Missing value handling
- Identify missing values and percentages
- Remove rows with missing names when required
- Fill missing Gender, City, and Department values
- Fill missing Purchase_Amount and Discount values using the median

### Part C – Duplicate handling
- Identify duplicate rows and duplicate Customer_ID values
- Remove duplicate records and confirm the cleaned output

### Part D – Text cleaning
- Remove extra whitespace from names
- Standardize capitalization for Name, City, and Department
- Normalize Gender values to Male and Female

### Part E – Invalid values
- Find customers with invalid age values
- Replace invalid ages with missing values
- Find negative Purchase_Amount values
- Replace invalid purchase amounts with missing values
- Find Discount values above 100

### Part F – Date cleaning
- Convert Joining_Date to a proper datetime column
- Identify invalid date entries
- Extract Joining_Year and Joining_Month
- Filter customers who joined after January 1, 2022

### Part G – Email cleaning
- Find missing emails
- Find emails lacking @
- Find emails without a valid-looking domain such as .com
- Create an Email_Status column with Valid, Invalid, and Missing values

### Part H – Filtering
- Filter customers by city, purchase amount, age range, department, and combined conditions

### Part I – Data analysis
- Calculate average and median purchase amounts
- Determine minimum and maximum purchase amounts
- Compute average age
- Compute purchase statistics by city and department
- Count customers by city and department
- Calculate average discount by department

### Final Challenge
A separate cleaned DataFrame is created from the original data dictionary to ensure the final output is not affected by earlier question-by-question modifications. The final challenge includes the required cleaned dataset preparation and prints summary metrics such as:

- Number of records
- Number of columns
- Missing values
- Duplicate records
- Average age
- Average purchase amount
- Maximum purchase amount
- Minimum purchase amount
- Number of cities
- Number of departments
- Number of valid emails
- Number of invalid emails

The notebook keeps the original data dictionary intact and preserves missing values where the exercise requires them.

## Setup instructions
Use the existing project virtual environment or create one if it does not already exist.

### Option 1: use the existing virtual environment
```bash
cd /Users/abinav/pandas-data-cleaning
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Option 2: create a new virtual environment
```bash
cd /Users/abinav/pandas-data-cleaning
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run the notebook in VS Code
1. Open the project in VS Code.
2. Open the notebook at `notebooks/customer_data_cleaning.ipynb`.
3. Ensure the notebook is using the project virtual environment.
4. Run all cells in order from the first cell to the final challenge.

## Notes
- The project intentionally includes only the exercise files required for this assignment.
- The virtual environment and generated cache files are excluded via `.gitignore`.
- No unrelated projects, API keys, or sensitive files are included in this repository.
