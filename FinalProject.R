# Load necessary libraries
library(rpart)
library(rpart.plot)
library(caret)
library(tidyverse)
library(dplyr)
library(MASS)
library(glmnet)
library(randomForest)
library(class)
library(ggplot2)

# Load the dataset
df <- read.csv("Amazon Customer Behavior Survey.csv")
summary(df)

#DATA PRE-PROCESSING-----------------------------------------------------------------------------------

# Data Pre-Processing
df <- df %>% dplyr::select(-Timestamp) %>%
  mutate(Purchase_Categories = str_split(Purchase_Categories, ";", simplify = TRUE)[, 1])

df <- df %>%
  mutate(Cart_Completion_Frequency = fct_collapse(Cart_Completion_Frequency,
                                                  Always_Often = c("Always", "Often"),
                                                  Never_Rarely = c("Never", "Rarely")
  ))
# Handling categories with few observations
category_counts <- df %>% 
  group_by(Improvement_Areas) %>% 
  summarise(Count = n()) %>% 
  ungroup()
categories_to_keep <- category_counts %>% 
  filter(Count > 1) %>% 
  pull(Improvement_Areas)
df <- df %>% filter(Improvement_Areas %in% categories_to_keep)

category_counts <- df %>% 
  group_by(Service_Appreciation) %>% 
  summarise(Count = n()) %>% 
  ungroup()
categories_to_keep <- category_counts %>% 
  filter(Count > 1) %>% 
  pull(Service_Appreciation)
df <- df %>% filter(Service_Appreciation %in% categories_to_keep)

category_counts <- df %>% 
  group_by(Product_Search_Method) %>% 
  summarise(Count = n()) %>% 
  ungroup()
categories_to_keep <- category_counts %>% 
  filter(Count > 2) %>% 
  pull(Product_Search_Method)
df <- df %>% filter(Product_Search_Method %in% categories_to_keep)
summary(df)
# Handling outliers
# Assuming 'age' is a numeric variable where outliers are possible
Q1 <- quantile(df$age, 0.25)
Q3 <- quantile(df$age, 0.75)
IQR <- IQR(df$age)
df <- df %>% filter(age >= (Q1 - 1.5 * IQR) & age <= (Q3 + 1.5 * IQR))

# Convert all character columns to factors
df <- df %>% 
  mutate_if(is.character, as.factor)


# Select relevant variables for the project
df <- df %>% dplyr::select(age, Gender, Purchase_Frequency, Purchase_Categories, 
                           Personalized_Recommendation_Frequency, Browsing_Frequency, 
                           Product_Search_Method, Search_Result_Exploration, 
                           Customer_Reviews_Importance, Add_to_Cart_Browsing, 
                           Cart_Completion_Frequency, Cart_Abandonment_Factors, 
                           Saveforlater_Frequency, Review_Left, Review_Reliability, 
                           Review_Helpfulness, Recommendation_Helpfulness, 
                           Rating_Accuracy, Shopping_Satisfaction, 
                           Service_Appreciation, Improvement_Areas)

# EXPLORATORY DATA ANALYSIS-------------------------------------------------------------------

# Descriptive statistics for demographic variables (age, gender)
df_summary <- df %>% 
  summarise(
    Age_Mean = mean(age, na.rm = TRUE),
    Age_SD = sd(age, na.rm = TRUE),
    Gender_Distribution = table(Gender)
  )

# Plotting age distribution
agedist <- ggplot(df, aes(x = age)) +
  geom_histogram(binwidth = 5, fill = 'blue', color = 'black') +
  labs(title = "Age Distribution", x = "Age", y = "Frequency") +
  theme_classic() +
  theme(
  text = element_text(size = 12), # Changes global text size
  axis.title = element_text(size = 14), # Changes axis titles size
  plot.title = element_text(size = 16) # Changes plot title size
  )
ggsave(filename = paste0("Age.png"), plot = agedist, width = 10, height = 8, dpi = 300)

# Plotting gender distribution
genderdist <- ggplot(df, aes(x = Gender)) +
  geom_bar(fill = 'orange', color = 'black') +
  labs(title = "Gender Distribution", x = "Gender", y = "Count") +
  theme_classic()
ggsave(filename = paste0("Gender.png"), plot = genderdist, width = 10, height = 8, dpi = 300)

# Distribution analysis for browsing and purchasing habits
# List of variables to plot
variables_to_plot <- c("Purchase_Frequency", "Purchase_Categories", 
                       "Personalized_Recommendation_Frequency", "Browsing_Frequency", 
                       "Product_Search_Method", "Search_Result_Exploration", 
                       "Customer_Reviews_Importance", "Add_to_Cart_Browsing", 
                       "Cart_Completion_Frequency", "Cart_Abandonment_Factors", 
                       "Saveforlater_Frequency", "Review_Left", "Review_Reliability", 
                       "Review_Helpfulness", "Recommendation_Helpfulness", 
                       "Rating_Accuracy", "Shopping_Satisfaction", 
                       "Service_Appreciation", "Improvement_Areas")

plot_and_save <- function(df, variable_name) {
  # Create a neat title by replacing underscores with spaces
  neat_title <- gsub("_", " ", variable_name)
  
  # Create the plot with labels
  p <- ggplot(df, aes_string(x = variable_name, fill = variable_name)) +
    geom_bar(color = 'black') +
    geom_text(stat = 'count', aes(label = after_stat(count)), vjust = -0.5) +
    theme_classic() +
    theme(
      text = element_text(size = 12),
      axis.title = element_text(size = 14),
      plot.title = element_text(size = 16),
      axis.text.x = element_text(angle = 45, hjust = 1, size = 10)
    ) +
    labs(title = neat_title, x = neat_title, y = "Count") +
    guides(fill=FALSE) # This removes the legend
  
  # Save the plot as an image file
  ggsave(filename = paste0(variable_name, ".png"), plot = p, width = 10, height = 8, dpi = 300)
  
  # Return the plot object in case it needs to be printed or further modified
  return(p)
}



# Loop through the variable names, plot, save, and print each one
for (variable_name in variables_to_plot) {
  plot <- plot_and_save(df, variable_name)
  print(plot)
}
#CORRELATION AND ANOVA ANALYSIS--------------------------------------------------------------------------

# Identify all numeric variables except 'Shopping_Satisfaction'
numeric_vars <- df %>% 
  select_if(is.numeric)%>% 
  dplyr::select(-Shopping_Satisfaction)

# Calculate correlations with 'Shopping_Satisfaction'
correlations <- sapply(numeric_vars, function(x) cor(x, df$Shopping_Satisfaction, 
                                                     use = "complete.obs"))

# View the correlation coefficients
print(correlations)

# Select only categorical variables
cat_vars <- df %>% select_if(is.factor)

# Loop through each categorical variable and perform ANOVA
anova_results <- lapply(names(cat_vars), function(var) {
  formula <- as.formula(paste("Shopping_Satisfaction ~", var))
  anova_test <- aov(formula, data = df)
  return(summary(anova_test))
})

# Names the list elements for easier identification
names(anova_results) <- names(cat_vars)

# Now you can view the ANOVA results for each categorical variable
anova_results

#SHOPPING SATISFACTION MULTIPLE REGRESSION MODEL--------------------------------------------------------------------------------------------

# Split the data into training and test sets
set.seed(1)
train.index <- sample(1:nrow(df), 0.8*nrow(df))
train.df <- df[train.index, ]
valid.df <- df[-train.index, ]


# Build the multiple regression model with Shopping_Satisfaction as the response variable
# and all other variables as predictors.
model <- lm(Shopping_Satisfaction ~ ., data = train.df)
model <- stepAIC(model, direction = "both")
log_model <- lm(log(Shopping_Satisfaction) ~ ., data = train.df)
log_model <- stepAIC(log_model, direction = "both")

# Summarize the model to view coefficients and statistics
model_summary <- summary(model)
print(model_summary)

# Check for model assumptions using diagnostic plots
par(mfrow = c(2, 2)) # Set up the graphics layout for 4 plots
plot(model) # Plot diagnostic plots for the model
plot(log_model)
par(mfrow = c(1, 1)) # Reset the graphics layout

# Predict Shopping_Satisfaction on the validation set
valid.predictions <- predict(model, newdata = valid.df)
valid.predictions2 <- predict(log_model, newdata = valid.df)
valid.predictions2 <- exp(valid.predictions2)
# Compare predictions to actual Shopping_Satisfaction scores using a performance metric, such as RMSE
rmse <- sqrt(mean((valid.predictions - valid.df$Shopping_Satisfaction)^2))
print(paste("Validation RMSE:", rmse))
mae <- mean(abs(valid.predictions - valid.df$Shopping_Satisfaction))
print(paste("Validation MAE:", mae))
log_rmse <- sqrt(mean((valid.predictions2 - valid.df$Shopping_Satisfaction)^2))
print(paste("Log Validation RMSE:", log_rmse))

# It's also useful to look at a scatter plot of predicted vs actual values
ggplot(data = valid.df, aes(x = valid.predictions, y = Shopping_Satisfaction)) +
  geom_point() +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "red") +
  labs(x = "Predicted Shopping Satisfaction", y = "Actual Shopping Satisfaction", 
       title = "Predicted vs Actual Shopping Satisfaction")

#CART COMPLETION RANDOM FOREST MODEL------------------------------------------------------------

train.df.part1 <- train.df[train.df$Cart_Completion_Frequency=="Sometimes",]
train.df.part2 <- train.df[train.df$Cart_Completion_Frequency=="Always_Often",]
set.seed(1)
train.df.part1 <- train.df.part1[sample(1:nrow(train.df.part1), 64),]
train.df.part2 <- train.df.part2[sample(1:nrow(train.df.part2), 64),]
train.df.part3 <- train.df[train.df$Cart_Completion_Frequency=="Never_Rarely",]
train.df <- rbind(train.df.part1,train.df.part2,train.df.part3)

# Define the control using a cross-validation approach
train_control <- trainControl(method="cv", number=5)

# Define the tune grid (hyperparameter grid)
tune_grid <- expand.grid(mtry = c(2, 3, 4))

# Train the model
set.seed(1) # For reproducibility
rf_model <- train(Cart_Completion_Frequency ~ ., data = train.df,
                  method = "rf",
                  trControl = train_control,
                  tuneGrid = tune_grid)
# Predict on the validation set
rf_predictions <- predict(rf_model, newdata = valid.df)
confusion <- confusionMatrix(rf_predictions, valid.df$Cart_Completion_Frequency)

# Output the confusion matrix
print(confusion)

# Extract the accuracy from the confusion matrix
accuracy <- confusion$overall['Accuracy']

# Print the accuracy
print(paste("Accuracy:", accuracy))

# Find the most common class in the Cart_Completion_Frequency variable
most_common_class <- names(which.max(table(df$Cart_Completion_Frequency)))

# Calculate the baseline accuracy by dividing the count of the most common class 
# by the total number of observations
baseline_accuracy <- max(table(df$Cart_Completion_Frequency)) / nrow(df)

# Print the baseline accuracy
print(paste("Baseline Accuracy:", baseline_accuracy))

# Compare with your model's accuracy
model_accuracy <- accuracy # This is the accuracy you obtained from your model
print(paste("Model Accuracy:", model_accuracy))

# Check if the model accuracy is significantly better than the baseline
if (model_accuracy > baseline_accuracy) {
  print("The model's accuracy is better than the baseline.")
} else {
  print("The model's accuracy is not better than the baseline.")
}

# Convert the confusion matrix to a table
confusion_data <- as.data.frame(confusion$table)

# Create the confusion matrix plot
confusion_plot <- ggplot(data = confusion_data, aes(x = Reference, y = Prediction, fill = Freq)) +
  geom_tile(color = "white") +  # Use geom_tile to create a heatmap-like visualization
  geom_text(aes(label = sprintf("%d", Freq)), vjust = 1) +  # Add text to each tile
  scale_fill_gradient(low = "white", high = "steelblue") +  # Use a gradient for filling
  labs(title = "Confusion Matrix", x = "Actual Class", y = "Predicted Class") +
  theme_minimal() +  # Use a minimal theme
  theme(axis.text.x = element_text(angle = 45, hjust = 1))  # Angle the x-axis text for readability

# Print the plot
print(confusion_plot)

# Assuming 'rf_model' is your trained Random Forest model
importance <- varImp(rf_model, scale = FALSE)

# Convert the variable importance to a data frame for plotting
importance_df <- as.data.frame(importance$importance)

# Convert row names to a column
importance_df$Feature <- rownames(importance_df)

# Ensure the 'Feature' column is now the first column
importance_df <- importance_df[, c("Feature", "Overall")]


# Sort by importance and select the top N features, let's say top 20
top_n <- 20
importance_df <- head(importance_df[order(-importance_df$Overall), ], top_n)

# Create the ggplot with adjusted settings
feature_importance_plot <- ggplot(importance_df, aes(x = reorder(Feature, Overall), y = Overall)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  coord_flip() + # Flip the axis to make the plot horizontal
  labs(y = "Variable Importance", x = "Features") +
  theme_minimal() +
  theme(axis.text.y = element_text(size = 8)) # Rotate and resize text

# Increase the size of the plot when saving
ggsave("feature_importance_plot.png", feature_importance_plot, width = 12, height = 8)
print(feature_importance_plot)
