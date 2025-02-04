library(caret)
library(e1071)
library(class)
library(randomForest)
library(rpart)
library(ggplot2)

# Load dataset
dataset <- read.csv(file.choose())
View(dataset)
names(dataset) <- make.names(names(dataset))  # Clean column names to make valid

# Preprocessing
dataset <- na.omit(dataset)  # Remove missing values
dataset$Rain <- ifelse(dataset$Precip.Type == "rain", 1, 0)
dataset$Rain <- as.factor(dataset$Rain)

features <- c("Temperature..C.", "Apparent.Temperature..C.", "Humidity",
              "Wind.Speed..km.h.", "Visibility..km.", "Pressure..millibars.")
dataset <- dataset[, c(features, "Rain")]
dataset[, features] <- lapply(dataset[, features], as.numeric)

# Split data
set.seed(123)
trainIndex <- createDataPartition(dataset$Rain, p = 0.7, list = FALSE)
trainData <- dataset[trainIndex, ]
testData <- dataset[-trainIndex, ]

# Function to calculate performance metrics
calculate_metrics <- function(true, pred) {
  cm <- confusionMatrix(pred, true)
  accuracy <- cm$overall["Accuracy"]
  precision <- cm$byClass["Pos Pred Value"]
  recall <- cm$byClass["Sensitivity"]
  f1 <- 2 * ((precision * recall) / (precision + recall))
  error_rate <- 1 - accuracy
  return(c(accuracy, precision, recall, f1, error_rate))
}

# Initialize results dataframe
results <- data.frame(
  Model = c("KNN", "SVM", "Decision Tree", "Random Forest", "Naive Bayes", "K-Means"),
  Accuracy = NA, Precision = NA, Recall = NA, F1 = NA, ErrorRate = NA
)

# 1. KNN
knn_pred <- knn(
  train = trainData[, -which(names(trainData) == "Rain")],
  test = testData[, -which(names(testData) == "Rain")],
  cl = trainData$Rain, k = 3
)
knn_metrics <- calculate_metrics(testData$Rain, knn_pred)
results[1, 2:6] <- knn_metrics
cat("\nKNN Metrics:\n")
print(knn_metrics)

# 2. SVM
svm_model <- svm(Rain ~ ., data = trainData, kernel = "linear", cost = 1)
svm_pred <- predict(svm_model, testData)
svm_metrics <- calculate_metrics(testData$Rain, svm_pred)
results[2, 2:6] <- svm_metrics
cat("\nSVM Metrics:\n")
print(svm_metrics)

# 3. Decision Tree
dt_model <- rpart(Rain ~ ., data = trainData, method = "class")
dt_pred <- predict(dt_model, testData, type = "class")
dt_metrics <- calculate_metrics(testData$Rain, dt_pred)
results[3, 2:6] <- dt_metrics
cat("\nDecision Tree Metrics:\n")
print(dt_metrics)

# 4. Random Forest
rf_model <- randomForest(Rain ~ ., data = trainData, ntree = 100)
rf_pred <- predict(rf_model, testData)
rf_metrics <- calculate_metrics(testData$Rain, rf_pred)
results[4, 2:6] <- rf_metrics
cat("\nRandom Forest Metrics:\n")
print(rf_metrics)

# 5. Naive Bayes
nb_model <- naiveBayes(Rain ~ ., data = trainData)
nb_pred <- predict(nb_model, testData)
nb_metrics <- calculate_metrics(testData$Rain, nb_pred)
results[5, 2:6] <- nb_metrics
cat("\nNaive Bayes Metrics:\n")
print(nb_metrics)

# 6. K-Means Clustering
set.seed(123)
kmeans_model <- kmeans(dataset[, features], centers = 2)  # 2 clusters for binary classification
kmeans_pred <- as.factor(ifelse(kmeans_model$cluster == 1, 1, 0))  # Map clusters to 1 or 0
kmeans_metrics <- calculate_metrics(dataset$Rain, kmeans_pred)
results[6, 2:6] <- kmeans_metrics
cat("\nK-Means Clustering Metrics:\n")
print(kmeans_metrics)

# Display consolidated results
cat("\nFinal Results Table:\n")
print(results)

# Plot Results
results_long <- reshape2::melt(results, id.vars = "Model", variable.name = "Metric", value.name = "Value")
ggplot(results_long, aes(x = Model, y = Value, fill = Metric)) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(title = "Model Performance Comparison", x = "Model", y = "Metric Value") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

