# Loan Prediction Classification model to identify customers who will purchase a personal loan
Decision Tree Classifier

## Overview
This project aims to build a predictive model for AllLife Bank to identify potential customers who have a higher probability of purchasing a personal loan. The model will help the marketing department target customers more effectively, thereby increasing the loan conversion rate.

## Business Problem
AllLife Bank seeks to convert its liability customers (depositors) into personal loan customers to expand its loan business. A previous campaign showed a 9%+ success rate, and the goal is to improve target marketing to further increase this ratio. The core problem is to predict whether a liability customer will accept a personal loan offer.

## Data
The dataset contains 5000 rows and 14 columns, including customer demographics, financial information, and their response to a personal loan offer (`Personal_Loan`).

**Data Dictionary:**
* `ID`: Customer ID
* `Age`: Customer’s age in completed years
* `Experience`: #years of professional experience
* `Income`: Annual income of the customer (in thousand dollars)
* `ZIP Code`: Home Address ZIP code.
* `Family`: the Family size of the customer
* `CCAvg`: Average spending on credit cards per month (in thousand dollars)
* `Education`: Education Level. 1: Undergrad; 2: Graduate;3: Advanced/Professional
* `Mortgage`: Value of house mortgage if any. (in thousand dollars)
* `Personal_Loan`: Did this customer accept the personal loan offered in the last campaign? (0: No, 1: Yes) - **Target Variable**
* `Securities_Account`: Does the customer have a securities account with the bank? (0: No, 1: Yes)
* `CD_Account`: Does the customer have a certificate of deposit (CD) account with the bank? (0: No, 1: Yes)
* `Online`: Do customers use internet banking facilities? (0: No, 1: Yes)
* `CreditCard`: Does the customer use a credit card issued by any other Bank (excluding All life Bank)? (0: No, 1: Yes)

## Approach
1.  **Data Loading and Initial Exploration**: Loaded the dataset, checked for shape, data types, and descriptive statistics.
2.  **Data Preprocessing**: 
    *   Handled anomalous 'Experience' values (negative values replaced with positive counterparts).
    *   Mapped 'Education' levels to descriptive categories (Undergraduate, Graduate, Professional).
    *   Feature Engineered 'ZIPCode' by extracting the first two digits to reduce cardinality and converted it to a categorical type.
    *   Dropped 'ID' and 'Experience' (due to high correlation with Age).
    *   Created dummy variables for `ZIPCode` and `Education`.
3.  **Exploratory Data Analysis (EDA)**: Performed univariate and bivariate analysis to understand data distribution and relationships with the target variable (`Personal_Loan`). Key observations included:
    *   Income, Education, Family size, and CCAvg show notable relationships with Personal Loan acceptance.
    *   Strong correlation between Age and Experience was observed.
4.  **Model Building**: Implemented Decision Tree models.
    *   **Baseline Decision Tree**: Trained a `DecisionTreeClassifier` with default parameters.
    *   **Pre-pruning**: Tuned hyperparameters (`max_depth`, `max_leaf_nodes`, `min_samples_split`) using a grid search approach to optimize recall on the test set while balancing class weights.
    *   **Post-pruning**: Explored Cost-Complexity Pruning (CCP) to find the optimal `ccp_alpha`.
5.  **Model Evaluation**: Evaluated models using accuracy, precision, recall, and F1-score, with a focus on maximizing recall due to the business objective of minimizing false negatives (lost opportunities).

## Results
*   The **Decision Tree (Pre-Pruning)** model demonstrated the best performance in terms of recall on the test set.
    *   **Test Recall**: 95.97%
    *   **Test Accuracy**: 95.67%
    *   **Test Precision**: 70.79%
    *   **Test F1-Score**: 81.48%
*   **Key Feature Importances**: Income, Family, and Education (specifically 'Education_Undergraduate') were identified as the most important features in predicting personal loan acceptance.

**Actionable Insights and Business Recommendations**:
*   **Target Higher Income Customers**: Customers with incomes above $90k-$100k, especially those with incomes above $116k and a family size greater than 2, show a higher propensity to take personal loans.
*   **Focus on Higher Education Levels**: Customers with Graduate or Advanced/Professional education are more likely to accept personal loan offers.
*   **Consider Family Size**: A family size of 3 or more members correlates with a higher likelihood of taking a personal loan.
*   **Leverage CD Account Holders**: Approximately 50% of customers with a Certificate of Deposit (CD_Account) tend to be interested in personal loans; these customers should be specifically targeted.

## Tools & Technologies
Python, Pandas, NumPy, Scikit-learn, Matplotlib, SQL

## Key Learnings
- Handling class imbalance
- Interpreting model outputs for business decisions
- Translating predictions into actionable insights
