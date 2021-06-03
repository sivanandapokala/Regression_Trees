# Working directory
getwd()

# Install the packages which are needed
#install.packages('plyr')
#install.packages('readr')
#install.packages('dplyr')

# Libraries used
library(readxl)
library(plyr)
library(readr)
library(dplyr)
library(caret)
library(ggplot2)
library(rpart)    
library(rpart.plot)
library(randomForest)


# Load the data
# Data set has 9568 data points collected from a Combined Cycle Power Plant over 6 years (2006-2011)
# Features are  hourly average ambient variables Temperature (T), Ambient Pressure (AP), Relative Humidity (RH) and Exhaust Vacuum (V)
# We are interested to predict the net hourly electrical energy output (PE) of the plant
data <- read_excel("Folds5x2_pp.xlsx")

# Display the first few records
head(data)

# Summary of all features
summary(data)

# Check missing values : No missing values
sum(is.na(data))


# Structure of the dataset
str(data)

# Split the data into train and test
set.seed(100) 
index = sample(1:nrow(data), 0.8*nrow(data)) 
# The training data 
train = data[index,] 
# The test data
test = data[-index,]
# Dimension of train and test dataset
dim(train)
dim(test)


# Model: Regression tree
# rpart: Recursive partitioning and Regression trees
# data used: training dataset
# method is anova for regression trees
RT_model = rpart(PE ~ ., 
                 data=train, 
                 method = "anova", 
                 control=rpart.control(minsplit=50, cp=0.001))

#Summary of the RT_model
summary(RT_model)

# Results evaluation function
results_eval_func <- function(true, predicted, df) {
  SSE <- sum((predicted - true)^2)
  SST <- sum((true - mean(true))^2)
  R_square <- 1 - SSE / SST
  RMSE = sqrt(SSE/nrow(df))
  
# Model performance 
  data.frame(
    RMSE = RMSE,
    Rsquare = R_square
  )
  
}

# Predict and evaluate the model: train data

predictions_train = predict(RT_model, data = train)
results_eval_func(train$PE, predictions_train, train)

# Predict and evaluate the model: test data

predictions_test = predict(RT_model, newdata = test)
results_eval_func(test$PE, predictions_test, test)

#--------------------------------------------------------------

# Default Random Forest model using randomForest function: Train data
RF_model = randomForest(PE ~ ., data=train)
summary(RF_model)

# importance of variable
importance(RF_model)

# Predict and evaluate the model: train data
predictions_train_rf = predict(RF_model, data = train)
results_eval_func(train$PE, predictions_train_rf, train)

# Predict and evaluate the model: test data
predictions_test_rf = predict(RF_model, newdata = test)
results_eval_func(test$PE, predictions_test_rf, test)

#-------------------------------------------------------------
# train function and hyper parameter tuning

set.seed(1234)

# Use trainControl function for k fold cross validation(cv)
# In method specify the resampling method: cv is specified here
# In number: enter the number of folds
# In search: enter either grid or random

trControl <- trainControl(method = "cv",
                          number = 5,
                          search = "random")

# We are interested to tune subset of features(mtry)
tuneGrid <- expand.grid(.mtry = c(1: 4))

# Function train: To fit predictive models over different tuning parameters 
# Method chosen is random forest on training dataset

RF_mtry <- train(PE~.,
                 data = train,
                 method = "rf",
                 metric = "RMSE",
                 tuneGrid = tuneGrid,
                 trControl = trControl,
                 importance = TRUE,
                 nodesize = 14,
                 ntree = 30)

print(RF_mtry)


# Best value of mtry
best_mtry <- RF_mtry$bestTune$mtry
best_mtry

# Tune maxnodes by considering already tuned mtry parameter

list_maxnode <- list()
tuneGrid <- expand.grid(.mtry = best_mtry)

for (maxnodes in c(5: 15)) {
  set.seed(1234)
  RF_maxnode <- train(PE~.,
                      data = train,
                      method = "rf",
                      metric = "RMSE",
                      tuneGrid = tuneGrid,
                      trControl = trControl,
                      importance = TRUE,
                      nodesize = 14,
                      maxnodes = maxnodes,
                      ntree = 30)
  current_iteration <- toString(maxnodes)
  list_maxnode[[current_iteration]] <- RF_maxnode
}
results_mtry <- resamples(list_maxnode)
summary(results_mtry)

# Usually, in random forest trees are not pruned : As we increase maxnodes, MAE decreses
list_maxnode <- list()
tuneGrid <- expand.grid(.mtry = best_mtry)
for (maxnodes in c(1500: 1505)) {
  set.seed(1234)
  RF_maxnode <- train(PE~.,
                      data = train,
                      method = "rf",
                      metric = "RMSE",
                      tuneGrid = tuneGrid,
                      trControl = trControl,
                      importance = TRUE,
                      nodesize = 14,
                      maxnodes = maxnodes,
                      ntree = 30)
  current_iteration <- toString(maxnodes)
  list_maxnode[[current_iteration]] <- RF_maxnode
}
results_mtry <- resamples(list_maxnode)
summary(results_mtry)

# Search the best ntrees

list_maxtrees <- list()
for (ntree in c(60, 70, 80, 90, 100)) {
  set.seed(5678)
  RF_maxtrees <- train(PE~.,
                       data = train,
                       method = "rf",
                       metric = "RMSE",
                       tuneGrid = tuneGrid,
                       trControl = trControl,
                       importance = TRUE,
                       nodesize = 14,
                       maxnodes = 30,
                       ntree = ntree)
  key <- toString(ntree)
  list_maxtrees[[key]] <- RF_maxtrees
}
results_tree <- resamples(list_maxtrees)
summary(results_tree)

# Final model based on parameter tuning

fit_RF <- train(PE~.,
                train,
                method = "rf",
                metric = "RMSE",
                tuneGrid = tuneGrid,
                trControl = trControl,
                importance = TRUE,
                nodesize = 14,
                ntree = 80,
                maxnodes = 1500)


# predict and evaluate the model on train data
predictions_hp_train_rf = predict(fit_RF, data = train)
results_eval_func(train$PE, predictions_hp_train_rf, train)

# predict and evaluate the model on test data
predictions_hp_test_rf = predict(fit_RF, newdata = test)
results_eval_func(test$PE, predictions_hp_test_rf, test)
