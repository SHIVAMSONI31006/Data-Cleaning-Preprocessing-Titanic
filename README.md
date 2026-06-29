# Data Cleaning & Preprocessing Workflow

This project focuses on executing key data cleaning and preprocessing techniques on raw data to prepare it for Machine Learning models using the Titanic Dataset.

## Project Execution Steps
1. **Data Exploration:** Imported the raw dataset and examined fundamental structural characteristics, data types, and total missing records.
2. **Handling Missing Values:** Resolved missing data fields by imputing the median value for the 'Age' attribute and the statistical mode for the 'Embarked' attribute. Dropped the high-cardinality 'Cabin' feature due to excessive missing entries.
3. **Categorical Encoding:** Converted non-numeric categorical attributes ('Sex' and 'Embarked') into machine-readable numeric formats using modern Label Encoding techniques.
4. **Feature Scaling:** Applied standard scaling and normalization via `StandardScaler` to align numeric distributions for 'Age' and 'Fare' attributes.
5. **Outlier Mitigation:** Conducted structural analysis using statistical Boxplots and removed extreme anomalous points from the 'Fare' variable using the standard Interquartile Range (IQR) method.

## Installation and Setup
To deploy this baseline environment, install the requisite data science dependencies first:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn