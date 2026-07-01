# Task 2: Exploratory Data Analysis (EDA)

## Project Objective
The objective of this task is to perform a comprehensive Exploratory Data Analysis (EDA) on the dataset to understand its underlying patterns, statistical properties, and feature relationships before moving toward predictive modeling.

## Features & Implementation
This script automates the primary workflows of an initial data exploration phase:
1. **Summary Statistics:** Generates structural information, central tendency data (`mean`, `median`), and dispersion insights. Missing values are automatically identified.
2. **Distribution & Outlier Analysis:** Utilizes Matplotlib and Seaborn to plot distribution histograms (with KDE) and boxplots to visually detect outliers and skewness.
3. **Feature Relationships:** 
   - Computes a numerical correlation matrix visualized via a Seaborn Heatmap.
   - Generates a pairwise relationship plot (`pairplot`) segmented by categorical groupings.
4. **Interactive Exploration:** Implements a Plotly interactive scatter plot allowing for multi-dimensional data analysis (Income vs. Savings across Departments).

## Prerequisites
Ensure you have the required data science libraries installed before running the script:

```bash
pip install numpy pandas matplotlib seaborn plotly