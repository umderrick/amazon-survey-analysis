# amazon-survey-analysis
R Script analyzing Amazon shopper's survey data from Kaggle.

# Amazon Customer Behavior Survey Analysis

This R script performs an analysis of an Amazon Customer Behavior Survey dataset. The workflow covers data pre-processing, exploratory analysis, statistical testing, and model building for predicting shopping satisfaction and cart completion frequency.

---

## Overview

The script processes survey data to extract meaningful insights about customer behavior. It cleans and transforms the raw data, visualizes distributions, performs statistical tests, and builds predictive models. Two primary modeling approaches are implemented:
- **Multiple Regression**: To predict the shopping satisfaction score.
- **Random Forest Classification**: To predict the frequency of cart completion.

---

## Key Features

### 1. Data Pre-Processing
- **Data Cleaning:**  
  - Removes the unnecessary `Timestamp` column.
  - Splits and simplifies the `Purchase_Categories` column.
  - Collapses categories in `Cart_Completion_Frequency` (e.g., combining "Always" and "Often").
  
- **Filtering Categories:**  
  - Excludes infrequent categories in variables such as `Improvement_Areas`, `Service_Appreciation`, and `Product_Search_Method`.

- **Outlier Handling:**  
  - Detects and removes outliers in the `age` variable using the Interquartile Range (IQR) method.

- **Data Transformation:**  
  - Converts character columns to factors.
  - Selects relevant variables for further analysis.

### 2. Exploratory Data Analysis (EDA)
- **Descriptive Statistics:**  
  - Provides summary statistics for demographics like age and gender.

- **Visualization:**  
  - Generates histograms and bar plots for:
    - Age distribution.
    - Gender distribution.
    - A variety of other behavioral and survey response variables.
  - Saves the generated plots as PNG files for further review.

### 3. Statistical Analysis
- **Correlation Analysis:**  
  - Computes correlation coefficients between numeric variables and `Shopping_Satisfaction`.

- **ANOVA Testing:**  
  - Performs ANOVA for each categorical variable to assess its effect on `Shopping_Satisfaction`.

### 4. Modeling
#### Multiple Regression Model for Shopping Satisfaction
- **Data Splitting:**  
  - Divides the dataset into training (80%) and validation (20%) sets.
  
- **Model Building:**  
  - Constructs a multiple regression model with stepwise selection based on AIC.
  - Builds an additional log-transformed model to account for non-linear effects.

- **Model Evaluation:**  
  - Generates diagnostic plots to check model assumptions.
  - Evaluates model performance using RMSE (Root Mean Squared Error) and MAE (Mean Absolute Error).
  - Creates a scatter plot comparing predicted versus actual shopping satisfaction scores.

#### Random Forest Model for Cart Completion Frequency
- **Data Preparation:**  
  - Balances the training data by sampling from different `Cart_Completion_Frequency` categories.
  
- **Model Training:**  
  - Uses cross-validation and a hyperparameter grid (varying `mtry`) to train the Random Forest model.
  
- **Evaluation:**  
  - Predicts on the validation set and computes a confusion matrix.
  - Calculates and compares the model accuracy against a baseline (most common class).
  - Visualizes the confusion matrix and plots the top 20 most important features.

---

## Requirements

- **R Environment:**  
  Ensure you have R (version 3.6+ recommended) or RStudio installed.

- **Required Libraries:**  
  - `rpart`
  - `rpart.plot`
  - `caret`
  - `tidyverse`
  - `dplyr`
  - `MASS`
  - `glmnet`
  - `randomForest`
  - `class`
  - `ggplot2`

- **Dataset:**  
  - **File:** `Amazon Customer Behavior Survey.csv`  
    Make sure this file is located in your working directory or update the file path accordingly.

---

## How to Use

1. **Install Required Packages:**  
   If you haven't already installed the required libraries, run:
   ```r
   install.packages(c("rpart", "rpart.plot", "caret", "tidyverse", "dplyr", "MASS", "glmnet", "randomForest", "class", "ggplot2"))
   ```

2. **Run the Script:**  
   Execute the R script in your R or RStudio environment.

3. **Review Outputs:**  
   - Check the console for summary statistics, model performance metrics, and diagnostic outputs.
   - Review the generated PNG files for visual insights on distributions, model diagnostics, and variable importance.

4. **Modify as Needed:**  
   Adapt file paths, parameters, or model settings based on your specific needs.
