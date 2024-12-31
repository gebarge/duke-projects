# Ensuring that proper libraries are downloaded
library(dplyr)
library(tidyr)
library(stringr)
library(caret)  
library(fastDummies)  
library(glmnet)
library(pROC)
library(randomForest)
library(class)
# Loading the dataset into the workspace
loandata <- read.csv("Loan_default.csv")

set.seed(123) 

loandata <- loandata[sample(nrow(loandata), 10000), ]

results = loandata[,]

#######################
#### DATA CLEANING ####
#######################

# Step 1: Check for missing values, and then view a summary of how many missing values per column
missing_summary <- colSums(is.na(loandata))
print(missing_summary)
# It appears as though we do not have any missing values in any of our variables
# Therefore, there is no need to handle these values (remove, or impute values)

# Step 3: Clean categorical variables
# Convert categorical variables like Education, EmploymentType, MaritalStatus, LoanPurpose to factor variables
loandata$Education <- factor(loandata$Education)
loandata$EmploymentType <- factor(loandata$EmploymentType)
loandata$MaritalStatus <- factor(loandata$MaritalStatus)
loandata$LoanPurpose <- factor(loandata$LoanPurpose)

# Step 4: Change multiple categorical variables to binary variables
# Convert categorical, Yes/No, to binary, 1/0, for variables: HasMortgage, HasDependents, HasCoSigner, and Default
binaryvalues <- c("HasMortgage", "HasDependents", "HasCoSigner")
loandata <- loandata %>%
  mutate(across(all_of(binaryvalues), ~ ifelse(. == "Yes", 1, 0)))

# Step 5: Group Age into categories and create dummy variables
loandata$AgeGroup <- cut(loandata$Age,
                         breaks = c(18, 25, 35, 45, 55, Inf),
                         labels = c("18-25", "26-35", "36-45", "46-55", "56 and above"),
                         right = FALSE)

# Step 6: Scale numeric variables
# Scale continuous variables (mean 0, stdev 1) to ensure structure and comparison later on since we will be using modeling
# (IF YOU GUYS DISAGREE WITH SCALING IT, WE CAN LEAVE OUT, BUT IT'S HELPED ME IN THE PAST TO COMPARE NUMBERS ULTIMATELY)
# loandata <- loandata %>%
#  mutate(across(c(Age, Income, CreditScore, InterestRate, DTIRatio), scale))

# Step 7: Ensuring all variables/columns are the proper data types
str(loandata)

# Step 8: Convert to Dummies for other categorical variables
dummy_vars <- dummyVars("~ AgeGroup + Education + EmploymentType + MaritalStatus + LoanPurpose", data = loandata)

loan_data_dummies <- predict(dummy_vars, newdata = loandata)

loan_data_final <- cbind(loandata, loan_data_dummies)

loandata <- loan_data_final %>%
  select(-Education, -EmploymentType, -MaritalStatus, -LoanPurpose, -LoanID, -InterestRate, -Age, -AgeGroup)

# write.csv(loandata, "loan_default_clean.csv", row.names = FALSE)

############################
#### DATA VISUALIZATION ####
############################

visdata = read.csv("Loan_default.csv")
visdata$Education <- factor(visdata$Education)
visdata$EmploymentType <- factor(visdata$EmploymentType)
visdata$MaritalStatus <- factor(visdata$MaritalStatus)
visdata$LoanPurpose <- factor(visdata$LoanPurpose)

binaryvalues <- c("HasMortgage", "HasDependents", "HasCoSigner")
visdata <- visdata %>%
  mutate(across(all_of(binaryvalues), ~ ifelse(. == "Yes", 1, 0)))

# Calculate average Loan Amount by Credit Score
avg_loan_data <- visdata %>%
  group_by(CreditScore) %>%
  summarise(AvgLoanAmount = mean(LoanAmount, na.rm = TRUE))

# Plot Credit Score against Average Loan Amount
ggplot(avg_loan_data, aes(x = CreditScore, y = AvgLoanAmount)) +
  geom_point(alpha = 0.6, color = "blue") +  # Points colored blue
  labs(title = "Relationship between Credit Score and Average Loan Amount",
       x = "Credit Score",
       y = "Average Loan Amount") +
  theme_minimal() +
  geom_smooth(method = "lm", color = "red")



# Create CreditScoreRating column
visdata <- visdata %>%
  mutate(CreditScoreRating = case_when(
    CreditScore >= 720 & CreditScore <= 850 ~ "Excellent",
    CreditScore >= 690 & CreditScore < 720 ~ "Good",
    CreditScore >= 630 & CreditScore < 690 ~ "Fair",
    CreditScore >= 300 & CreditScore < 630 ~ "Poor",
    TRUE ~ "Unknown"  # This handles any scores outside the specified ranges
  ))


# Plot CreditScoreRating against Default
ggplot(visdata, aes(x = CreditScoreRating, fill = as.factor(Default))) + 
  geom_bar(position = "fill") + 
  scale_fill_manual(values = c("lightgray", "red"), name = "Default") + 
  labs(title = "Loan Default Rate by Credit Score Rating", x = "Credit Score Rating", y = "Proportion") +
  theme_minimal() 

# Create AgeGroup column
visdata <- visdata %>%
  mutate(AgeGroup = cut(Age,
                        breaks = c(17, 25, 35, 45, 55, 65, Inf),  # Define breaks
                        labels = c("18-25", "26-35", "36-45", "46-55", "56-65", "66+"),
                        right = TRUE))


# Count the number of Poor ratings by Age Group
poor_ratings_summary <- visdata %>%
  filter(CreditScoreRating == "Poor") %>%
  group_by(AgeGroup) %>%
  summarize(CountPoorRatings = n(), .groups = 'drop') %>%
  arrange(desc(CountPoorRatings))

# Display the results
print(poor_ratings_summary)


# Visualize Poor Ratings by Age Group
ggplot(poor_ratings_summary, aes(x = reorder(AgeGroup, -CountPoorRatings), y = CountPoorRatings)) +
  geom_bar(stat = "identity", fill = "red") +
  labs(title = "Number of Poor Credit Ratings by Age Group", 
       x = "Age Group", 
       y = "Count of Poor Ratings") +
  theme_minimal() 


# Calculate the average loan amount for Poor credit ratings by Age Group
poor_loan_summary <- visdata %>%
  filter(CreditScoreRating == "Poor") %>%
  group_by(AgeGroup) %>%
  summarize(AverageLoanAmount = mean(LoanAmount, na.rm = TRUE), .groups = 'drop') %>%
  arrange(desc(AverageLoanAmount))


# Create a scatter plot of Average Loan Amount by Age Group for Poor Ratings
ggplot(poor_loan_summary, aes(x = AgeGroup, y = AverageLoanAmount)) +
  geom_point(size = 3, color = "orange") +
  geom_line(aes(group = 1), color = "skyblue", size = 1) +  # Adding a line to connect the points
  labs(title = "Average Loan Amount for Poor Credit Ratings by Age Group", 
       x = "Age Group", 
       y = "Average Loan Amount") +
  theme_minimal()


# Summarize defaults by age group
default_summary <- visdata %>%
  group_by(AgeGroup) %>%
  summarize(DefaultCount = sum(Default),  # Count of defaults
            TotalCount = n(),           # Total number of loans in each group
            ProportionDefault = DefaultCount / TotalCount,  # Proportion of defaults
            .groups = 'drop') %>%
  arrange(desc(ProportionDefault))

# Display the results
print(default_summary)

# Create a bar plot of defaults by age group
ggplot(default_summary, aes(x = reorder(AgeGroup, -DefaultCount), y = DefaultCount)) +
  geom_bar(stat = "identity", fill = "orange") +
  labs(title = "Number of Defaults by Age Group", 
       x = "Age Group", 
       y = "Count of Defaults") +
  theme_minimal() 


####################################
###### Model For Default Prob ######
####################################

X <- loandata %>% select(-Default)  
y <- loandata$Default

folds <- createFolds(y, k = 5)

logistic_auc <- c()  
post_lasso_auc <- c()  

logistic_brier <- c()  
post_lasso_brier <- c() 

rf_auc <- c()   
rf_brier <- c()  

lasso_rf_auc <- c()   
lasso_rf_brier <- c()  

knn_auc <- c()  
knn_brier <- c()  
best_k_values <- c()

for (i in 1:5) {
  cat("k = ", i,"\n")
  test_index <- folds[[i]]  
  X_train <- X[-test_index, ]
  y_train <- y[-test_index]
  X_test <- X[test_index, ]
  y_test <- y[test_index]
  # Regular Logistic
  logistic_model <- glm(y_train ~ ., data = data.frame(y_train = y_train, X_train), family = "binomial")
  logistic_pred <- predict(logistic_model, newdata = data.frame(X_test), type = "response")
  
  #Calculate ACU and Brier of Regular Logistic
  logistic_auc[i] <- roc(y_test, logistic_pred)$auc
  logistic_brier[i] <- mean((logistic_pred - y_test)^2)
  # Lasso Logistic
  lasso_model <- cv.glmnet(as.matrix(X_train), y_train, family = "binomial", alpha = 1)
  selected_features <- which(coef(lasso_model, s = "lambda.min") != 0)[-1] - 1
  which(coef(lasso_model, s = "lambda.min") != 0)
  # Check if features selected are more than 1
  if (length(selected_features) > 1) {

    X_train_selected <- as.matrix(X_train[, selected_features])
    X_test_selected <- as.matrix(X_test[, selected_features])

    post_lasso_model <- glm(y_train ~ ., data = data.frame(y_train = y_train, X_train_selected), family = "binomial")
    post_lasso_pred <- predict(post_lasso_model, newdata = data.frame(X_test_selected), type = "response")
    
    #Calculate ACU and Brier of Post-Lasso Logistic
    post_lasso_auc[i] <- roc(y_test, post_lasso_pred)$auc
    post_lasso_brier[i] <- mean((post_lasso_pred - y_test)^2)
  } else {
    post_lasso_auc[i] <- NA
    post_lasso_brier[i] <- NA
  }
  
  cat("Logistic AUC:", logistic_auc[i], "\n")
  cat("Logistic Brier:", logistic_brier[i], "\n")
  cat("Post-Lasso Logistic AUC:", post_lasso_auc[i], "\n")
  cat("Post-Lasso Logistic Brier:", post_lasso_brier[i], "\n")

  # Random Forest
  rf_model <- randomForest(x = X_train, y = as.factor(y_train), ntree = 1000, mtry = 5)
  rf_pred_prob <- predict(rf_model, X_test, type = "prob")[, 2]

  #Calculate ACU and Brier of Random Forest
  rf_auc[i] <- roc(y_test, rf_pred_prob)$auc

  rf_brier[i] <- mean((rf_pred_prob - y_test)^2)

  #Post-Lasso Random Forest
  if (length(selected_features) > 0) {
    rf_lasso_model <- randomForest(x = X_train_selected, y = as.factor(y_train), ntree = 1000, mtry = 5)
    rf_lasso_pred_prob <- predict(rf_lasso_model, X_test_selected, type = "prob")[, 2]

    #Calculate ACU and Brier of Post-Lasso Random Forest
    lasso_rf_auc[i] <- roc(y_test, rf_lasso_pred_prob)$auc
    lasso_rf_brier[i] <- mean((rf_lasso_pred_prob - y_test)^2)
  } else {
    lasso_rf_auc[i] <- NA
    lasso_rf_brier[i] <- NA
  }
  cat("Random Forest AUC:", rf_auc[i], "\n")
  cat("Random Forest Brier:", rf_brier[i], "\n")
  cat("Post-Lasso Random Forest AUC:", lasso_rf_auc[i], "\n")
  cat("Post-Lasso Random Forest Brier:", lasso_rf_brier[i], "\n")
  
  ## KNN
  
  # ## FOLLOWING CODE IS TO FIND THE BEST K AT EACH FOLD, WE FINAALY DETERMINE THE BEST K TO BE 170 ON AVG.
  # ## TO SAME CV TIME, WE COMMENT THE FOLLOWING CODE
  
  # best_auc <- 0  
  # best_k <- 1  
  # k_range <- 1:200  
  # 
  # for (k_value in k_range) {
  #   knn_pred <- knn(train = X_train, test = X_test, cl = y_train, k = k_value, prob = TRUE)
  #   knn_pred_prob <- attr(knn_pred, "prob")
  # 
  #   # If class is 0, then return the prob. of being class 1
  #   knn_pred_prob[knn_pred == 0] <- 1 - knn_pred_prob[knn_pred == 0]
  # 
  #   current_auc <- roc(y_test, knn_pred_prob)$auc
  #   if (current_auc > best_auc) {
  #     best_auc <- current_auc
  #     best_k <- k_value
  #   }
  # }
  # 
  # best_k_values[i] <- best_k
  # cat("Best k for fold", i, ":", best_k, "\n")
  # 

  knn_pred <- knn(train = X_train, test = X_test, cl = y_train, k = 170, prob = TRUE)
  knn_pred_prob <- attr(knn_pred, "prob")
  
  # If class is 0, then return the prob. of being class 1
  knn_pred_prob[knn_pred == 0] <- 1 - knn_pred_prob[knn_pred == 0]
  
  # Calculate AUC and Brier of KNN
  knn_auc[i] <- roc(y_test, knn_pred_prob)$auc
  knn_brier[i] <- mean((knn_pred_prob - y_test)^2)
  cat("KNN AUC:", knn_auc[i], "\n")
  cat("KNN Brier:", knn_brier[i], "\n")
}

cat(selected_features)

cat("Logistic Regression Average AUC:", mean(logistic_auc, na.rm = TRUE), "\n")
cat("Logistic Regression Average Brier:", mean(logistic_brier, na.rm = TRUE), "\n")

cat("Post-Lasso Logistic Regression Average AUC:", mean(post_lasso_auc, na.rm = TRUE), "\n")
cat("Post-Lasso Logistic Regression Average Brier:", mean(post_lasso_brier, na.rm = TRUE), "\n")

cat("Random Forest Average AUC:", mean(rf_auc, na.rm = TRUE), "\n")
cat("Random Forest Average Brier:", mean(rf_brier, na.rm = TRUE), "\n")

cat("Post-Lasso Random Forest Average AUC:", mean(lasso_rf_auc, na.rm = TRUE), "\n")
cat("Post-Lasso Random Forest Average Brier:", mean(lasso_rf_brier, na.rm = TRUE), "\n")

cat("KNN Average AUC:", mean(knn_auc, na.rm = TRUE), "\n")
cat("KNN Average Brier:", mean(knn_brier, na.rm = TRUE), "\n")

## WE FOUND THE BEST MODEL IS THE REGULAR LOGISTICS, WHICH IS THE BEST ON BOTH AUC AND BRIER

#########################
#### LOAN SIMULATION ####
#########################

simudata <- read.csv("Loan_default.csv")

# Clean and convert data like before

simudata$Education <- factor(simudata$Education)
simudata$EmploymentType <- factor(simudata$EmploymentType)
simudata$MaritalStatus <- factor(simudata$MaritalStatus)
simudata$LoanPurpose <- factor(simudata$LoanPurpose)

binaryvalues <- c("HasMortgage", "HasDependents", "HasCoSigner")
loandata <- simudata %>%
  mutate(across(all_of(binaryvalues), ~ ifelse(. == "Yes", 1, 0)))

simudata$AgeGroup <- cut(simudata$Age,
                         breaks = c(18, 25, 35, 45, 55, Inf),
                         labels = c("18-25", "26-35", "36-45", "46-55", "56 and above"),
                         right = FALSE)

dummy_vars <- dummyVars("~ AgeGroup + Education + EmploymentType + MaritalStatus + LoanPurpose", data = simudata)

loan_data_dummies <- predict(dummy_vars, newdata = simudata)

loan_data_final <- cbind(simudata, loan_data_dummies)

simudata <- loan_data_final %>%
  select(-Education, -EmploymentType, -MaritalStatus, -LoanPurpose, -LoanID, -Age, -AgeGroup)


# Define variables to store the results of each loop
accuracy_by_profit_list <- numeric(10)
accuracy_by_logistic_list <- numeric(10)
total_profit_by_expected_profit_list <- numeric(10)
total_profit_by_logistic_list <- numeric(10)

# Set the random seed to ensure repeatable results
set.seed(123) 

for (i in 1:10) {
  # Randomly draw 5000 samples each time
  sampled_data <- simudata[sample(1:nrow(simudata), 5000, replace = FALSE), ]
  train_data <- sampled_data %>% select(-InterestRate)
  
  # Logistic regression model training
  logistic_model <- glm(Default ~ ., data = train_data, family = "binomial")
  
  # Predict the probability of default for sample data
  logistic_pred_all <- predict(logistic_model, newdata = train_data, type = "response")
  
  # Add the predicted default probability to the dataset
  sampled_data$Default_Probability <- logistic_pred_all
  
  # Calculate profit
  sampled_data$Profit <- sampled_data$LoanAmount * ((1 + sampled_data$InterestRate / 100 / 12) ^ sampled_data$LoanTerm - 1)
  
  # Calculate expected profit: E = Profit * (1 - Default_Probability) - LoanAmount * Default_Probability
  sampled_data$Expected_Profit <- sampled_data$Profit * (1 - sampled_data$Default_Probability) - sampled_data$LoanAmount * sampled_data$Default_Probability
  
  # Predict default (based on expected profit: positive expectation -> no default, negative expectation -> default)
  sampled_data$predicted_default_by_profit <- ifelse(sampled_data$Expected_Profit < 0, 1, 0)
  
  # Calculate the prediction accuracy based on expected profit
  accuracy_by_profit_list[i] <- mean(sampled_data$predicted_default_by_profit == sampled_data$Default)
  
  # Prediction based on logistic regression threshold 0.5
  sampled_data$predicted_default_by_logistic <- ifelse(sampled_data$Default_Probability >= 0.5, 1, 0)
  
  # Calculate the prediction accuracy based on logistic regression
  accuracy_by_logistic_list[i] <- mean(sampled_data$predicted_default_by_logistic == sampled_data$Default)
  
  # Actual Benefit of logistic regression threshold 0.5
  sampled_data$Benefit_logistic <- ifelse(sampled_data$predicted_default_by_logistic == 0, 
                                 ifelse(sampled_data$Default == 1, -sampled_data$LoanAmount, sampled_data$Profit), 
                                 0)
  # Actual Benefit of expected profit method
  sampled_data$Benefit_profit <- ifelse(sampled_data$predicted_default_by_profit == 0, 
                                 ifelse(sampled_data$Default == 1, -sampled_data$LoanAmount, sampled_data$Profit), 
                                 0)
  # Calculate the total benefits of the two methods
  total_profit_by_expected_profit_list[i] <- sum(sampled_data$Benefit_profit)
  total_profit_by_logistic_list[i] <- sum(sampled_data$Benefit_logistic)
}

# Calculate the average of ten cycles
mean_accuracy_by_profit <- mean(accuracy_by_profit_list)
mean_accuracy_by_logistic <- mean(accuracy_by_logistic_list)
mean_total_profit_by_expected_profit <- mean(total_profit_by_expected_profit_list)
mean_total_profit_by_logistic <- mean(total_profit_by_logistic_list)

# Output the average result
cat("The average accuracy of predicting defaults based on expected profits after ten cycles:", mean_accuracy_by_profit * 100, "%\n")
cat("Average accuracy of predicting defaults by the logistic regression model after ten cycles:", mean_accuracy_by_logistic * 100, "%\n")
cat("After ten cycles, the average total return without default is predicted by expected profit:", mean_total_profit_by_expected_profit, "\n")
cat("The average total return after ten cycles predicted by the logistic regression model without default is:", mean_total_profit_by_logistic, "\n")

mean_total_profit_by_expected_profit > mean_total_profit_by_logistic















