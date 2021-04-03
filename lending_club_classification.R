# Importing Libraries 
library(tidyverse)
library(caret)
library(tree)
library(class)
library(glmnet)
library(ROCR) 

# Importing Data 
lc <- read_csv("LendingClub_LoanStats_2011_v2.csv")  #read the Lending Club dataset in R

#create target variable: fully paid
#remove any rows where y is NA
lc <- lc %>%
  mutate(y = as.factor(ifelse(loan_status == "Fully Paid", "Paid", "Not Paid"))) %>%
  filter(!is.na(y))

#set seed and randomly downsample 15k instances 
#(otherwise training kNN will take hours)
set.seed(1)
lc_small <- sample(nrow(lc), 15000)
lc <- lc[lc_small,]

#then calculate the training/validation row numbers, but don't yet split
va_inst <- sample(nrow(lc), .3*nrow(lc))

# Data Cleaning 
lc1 <- lc %>% 
  select(grade, sub_grade, home_ownership, addr_state, loan_amnt, emp_length, annual_inc, purpose, dti, mths_since_last_delinq, int_rate, y)


# Binning emp_length
lc1$emp_length <- sub(" .*", "", lc1$emp_length)
lc1$emp_length <- ifelse(lc1$emp_length == "<", "0", lc1$emp_length)
lc1$emp_length <- ifelse(lc1$emp_length == "10+", "10", lc1$emp_length)
lc1$emp_length <- ifelse(lc1$emp_length == "n/a", "-1", lc1$emp_length)
lc1$emp_length <- as.numeric(lc1$emp_length)

# Cleaning emp_length, annual_inc, dti, mths_since_last_delinq, int_rate, y 
lc1 <- lc1 %>% 
  mutate(emp_length = case_when(
    emp_length < 0 ~ "unknown",
    emp_length < 1 ~ "<1 year",
    emp_length >= 1 & emp_length <= 3 ~ "1-3 years", 
    emp_length >= 4 & emp_length <= 6 ~ "4-6 years",
    emp_length >= 7 & emp_length <= 9 ~ "7-9 years",
    emp_length == 10 ~ "10+ years"),
    annual_inc = ifelse(is.na(annual_inc), mean(annual_inc, na.rm = TRUE), annual_inc),
    annual_inc = case_when(
      annual_inc < 40000 ~ "<40,000", 
      annual_inc >= 40000 & annual_inc < 59000 ~ "40,000-58,999",
      annual_inc >= 59000 & annual_inc < 82000 ~ "59,000-81,999",
      annual_inc >= 82000 ~ "82,000+"),
    dti = cut(dti, 5, labels = c("Very Low", "Low", "Mid", "High", "Very High")),
    mths_since_last_delinq = case_when(
      mths_since_last_delinq < 12 ~ "< 1 year", 
      mths_since_last_delinq >= 12 & mths_since_last_delinq < 24 ~ "1-2 years", 
      mths_since_last_delinq >+ 24 & mths_since_last_delinq < 36 ~ "2-3 years", 
      mths_since_last_delinq >= 36 ~ "3+ years"),
    int_rate = as.numeric(sub("%", "", int_rate)),
    y = as.factor(y))

lc1$mths_since_last_delinq = ifelse(is.na(lc1$mths_since_last_delinq) == TRUE, "never", lc1$mths_since_last_delinq)

# Processing Purpose 
sum_data <- lc1 %>%
  group_by(purpose) %>% 
  summarise(no_rows = length(purpose))

sum_data <- filter(sum_data, no_rows < 200)

lc1 <- lc1 %>% 
  mutate( 
    purpose = ifelse(purpose %in% sum_data$purpose, "other", purpose),
    purpose = ifelse(purpose == "credit_card" | purpose == "debt_consolidation", "debt", purpose))

# Encoding Dummy Variables 
dummy <- dummyVars(~. + annual_inc:emp_length, data=lc1)
one_hot_lc1 <- data.frame(predict(dummy, newdata = lc1))
one_hot_lc1$y.Paid = NULL 
one_hot_lc1$y.Not.Paid <- as.factor(one_hot_lc1$y.Not.Paid)

# Partitioning Data 
train <- one_hot_lc1[-va_inst,]
test <- one_hot_lc1[va_inst,]

# Decision Tree 
mycontrol = tree.control(nrow(train), mincut = 5, minsize = 10, mindev = 0.0005)
lc.full.tree=tree(y.Not.Paid ~ .,control = mycontrol, train)

# Computing number of terminal and decision nodes
term_nodes <- length(unique(lc.full.tree$where))
decision_nodes <- nrow(lc.full.tree$frame) - term_nodes

# Accuracy Function 
accuracy <- function(classifications, actuals){
  correct_classifications <- ifelse(classifications == actuals, 1, 0)
  acc <- sum(correct_classifications)/length(classifications)
  return(acc)}

# Pruning Function : Returns accuracy 
prune_tree_acc <- function(tree_size){
  lc.pruned.tree=prune.tree(lc.full.tree, best = tree_size)
  train_preds <- predict(lc.pruned.tree,newdata=train)
  test_preds <- predict(lc.pruned.tree, newdata=test)
  train_class <- ifelse(train_preds[,2] >= 0.5, 1, 0)
  test_class <- ifelse(test_preds[,2] >= 0.5, 1, 0)
  
  train_acc <- accuracy(train_class, train$y.Not.Paid)
  test_acc <- accuracy(test_class, test$y.Not.Paid)
  acc <- c(train_acc, test_acc)
  return(acc)
}

# Loops through each tree_size and returns accuracy 
tree_size <- c(2, 4, 6, 8, 10, 15, 20, 25, 30, 35, 40)
train_acc <- c()
test_acc <- c()
for (i in 1:length(tree_size)){
  a <- prune_tree_acc(tree_size = tree_size[i])
  train_acc[i] <- a[1]
  test_acc[i] <- a[2]
}

acc_df <- data.frame(tree_size = tree_size, 
                     train_acc = train_acc,
                     test_acc = test_acc)
acc_df$difference = acc_df$train_acc - acc_df$test_acc


# Plotting the fitting curve 
colors <- c("Train" = "steelblue", "Test" = "darkred")
ggplot(acc_df, aes(tree_size)) + 
  geom_line(aes(y = train_acc, color = "Train")) + geom_point(y = train_acc, color = "steelblue") + 
  geom_line(aes(y = test_acc, color = "Test")) + geom_point(y = test_acc, color = "darkred") + 
  labs(title="Fitting Curve",
       x="Tree Size", y="Accuracy", color = "Data Set")+ 
  scale_color_manual(values = colors)

# Getting the predictions for the optimal tree size 
lc.pruned.tree=prune.tree(lc.full.tree, best = 20)
train_preds <- predict(lc.pruned.tree,newdata=train)
best.tree.preds <- predict(lc.pruned.tree, newdata=test)
best.tree.preds = c(best.tree.preds[,2])

lc.pruned.tree

# Notes Tree_size = 20 is the best because the difference 
#between the training and testing accuracy is smallest meaning
# that the model is overfitting and underfitting the least compared to 
#the other tree_size values. 

# KNN Model 
# Creating Input Variables 
train.X <- select(train, -one_of(c("y.Not.Paid")))
test.X <- select(test, -one_of(c("y.Not.Paid")))
# Creating y Variables 
train.y <- as.numeric(train$y.Not.Paid) - 1
test.y <- as.numeric(test$y.Not.Paid) - 1


k_values <- c(2, 4, 6, 8, 10, 15, 20)
acc_train <- c()
acc_test <- c()

for (i in 1:length(k_values)){
  knn.pred.train <- knn(train.X, train.X, train.y, k = k_values[i], prob = TRUE)
  knn.pred.test <- knn(train.X,test.X,train.y,k=k_values[i], prob = TRUE)
  acc_train[i] <- accuracy(knn.pred.train, train.y)
  acc_test[i] <- accuracy(knn.pred.test, test.y)
}

knn_acc_df <- data.frame(k_value = k_values, 
                         acc_test = acc_test, 
                         acc_train = acc_train)

# Plotting the fitting curve 
colors <- c("Train" = "steelblue", "Test" = "orange")
ggplot(knn_acc_df, aes(k_value)) + 
  geom_line(aes(y = acc_train, color = "Train")) + geom_point(y = acc_train, color = "steelblue") + 
  geom_line(aes(y = acc_test, color = "Test")) + geom_point(y = acc_test, color = "orange") + 
  labs(title="Fitting Curve",
       x="k number", y="Accuracy", color = "Data Set")+ 
  scale_color_manual(values = colors)

# Getting probabilities for k = 20 
best_knn <- knn(train.X,test.X,train.y,k=20, prob = TRUE)
best_prob <- attr(best_knn, "prob")

# Lasso Logistic Model 
# Setting Up Grid Search 
grid <- 10^seq(10,-4,length=100)

lc2 <- rbind(train, test)
lc2_x <- model.matrix(y.Not.Paid~., lc2)
lc2_y <- lc2$y.Not.Paid

# Training/Test Splits
x_train <- lc2_x[-va_inst,]
x_valid <- lc2_x[va_inst,]

y_train <- lc2_y[-va_inst]
y_valid <- lc2_y[va_inst]

k <- 5

lasso.cv.out <- cv.glmnet(x_train, y_train, family="binomial", alpha=1, lambda=grid, nfolds=k)
plot(lasso.cv.out)

lasso.bestlam <- lasso.cv.out$lambda.min

best.lasso.preds <- predict(lasso.cv.out, s=lasso.bestlam, newx = x_valid,type="response")

lasso_class <- ifelse(best.lasso.preds >= 0.5, 1, 0)


best_lasso_acc <- accuracy(lasso_class,y_valid)

# Ridge Log Regression Model 
k <- 5

ridge.cv.out <- cv.glmnet(x_train, y_train, family="binomial", alpha=0, lambda=grid, nfolds=k)
plot(ridge.cv.out)

ridge.bestlam <- ridge.cv.out$lambda.min

best.ridge.preds <- predict(ridge.cv.out, s=lasso.bestlam, newx = x_valid,type="response")
ridge_class <- ifelse(best.ridge.preds >= 0.5, 1, 0)


best_ridge_acc <- accuracy(ridge_class,y_valid)

# Plotting ROC Curves for Comparison 
tree_preds <- prediction(best.tree.preds, test$y.Not.Paid)
roc_tree <- performance(tree_preds, "tpr", "fpr")

knn.prob_of_1 <- ifelse(best_knn == 1, best_prob, 1-best_prob)
knn_preds <- prediction(knn.prob_of_1, test$y.Not.Paid)
roc_knn <- performance(knn_preds, "tpr", "fpr")

lasso_preds <- prediction(best.lasso.preds, test$y.Not.Paid)
roc_lasso <- performance(lasso_preds, "tpr", "fpr")

ridge_preds <- prediction(best.ridge.preds, test$y.Not.Paid)
roc_ridge <- performance(ridge_preds, "tpr", "fpr")

plot(roc_tree, col = "steelblue")
plot(roc_knn, col = "darkred", add = TRUE)
plot(roc_lasso, col = "orange", add = TRUE)
plot(roc_ridge, col = "black", add = TRUE)

legend("right", c("Decision Tree", "knn", "Log Lasso", "Log Ridge"), lty=1, 
       col = c("steelblue", "darkred", "orange", "black"), bty="n", inset=c(0,-0.15))

# AUC Calculations
performance(tree_preds, measure = "auc")@y.values[[1]]
performance(knn_preds, measure = "auc")@y.values[[1]]
performance(lasso_preds, measure = "auc")@y.values[[1]]
performance(ridge_preds, measure = "auc")@y.values[[1]]

