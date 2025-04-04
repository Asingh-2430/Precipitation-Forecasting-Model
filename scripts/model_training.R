# Load necessary libraries
library(caret)
library(e1071)
library(class)
library(randomForest)
library(rpart)
library(ggplot2)
library(reshape2)

# Load dataset
dataset <- read.csv(file.choose())  # Replace with a static path if needed
names(dataset) <- make.names(names(dataset))  # Clean column names
dataset <- na.omit(dataset)  # Remove missing values

# Feature engineering
dataset$Rain <- ifelse(dataset$Precip.Type == "rain", 1, 0)
dataset$Rain <- as.factor(dataset$Rain)

# Select features
features <- c("Temperature..C.", "Apparent.Temperature..C.", "Humidity",
              "Wind.Speed..km.h.", "Visibility..km.", "Pressure..millibars.")
dataset <- dataset[, c(features, "Rain")]
dataset[, features] <- lapply(dataset[, features], as.numeric)

# Split dataset
set.seed(123)
trainIndex <- createDataPartition(dataset$Rain, p = 0.7, list = FALSE)
trainData <- dataset[trainIndex, ]
testData <- dataset[-trainIndex, ]

# Function to calculate metrics
calculate_metrics <- function(true, pred) {
  cm <- confusionMatrix(pred, true)
  accuracy <- cm$overall["Accuracy"]
  precision <- cm$byClass["Pos Pred Value"]
  recall <- cm$byClass["Sensitivity"]
  f1 <- 2 * ((precision * recall) / (precision + recall))
  error_rate <- 1 - accuracy
  return(c(accuracy, precision, recall, f1, error_rate))
}

# Initialize results
results <- data.frame(
  Model = c("KNN", "SVM", "Decision Tree", "Random Forest", "Naive Bayes", "K-Means"),
  Accuracy = NA, Precision = NA, Recall = NA, F1 = NA, ErrorRate = NA
)

# ---- Model Training & Evaluation ----

# 1. KNN
knn_pred <- knn(train = trainData[, -which(names(trainData) == "Rain")],
                test = testData[, -which(names(testData) == "Rain")],
                cl = trainData$Rain, k = 3)
results[1, 2:6] <- calculate_metrics(testData$Rain, knn_pred)

# 2. SVM
svm_model <- svm(Rain ~ ., data = trainData, kernel = "linear", cost = 1)
svm_pred <- predict(svm_model, testData)
results[2, 2:6] <- calculate_metrics(testData$Rain, svm_pred)

# 3. Decision Tree
dt_model <- rpart(Rain ~ ., data = trainData, method = "class")
dt_pred <- predict(dt_model, testData, type = "class")
results[3, 2:6] <- calculate_metrics(testData$Rain, dt_pred)

# 4. Random Forest
rf_model <- randomForest(Rain ~ ., data = trainData, ntree = 100)
rf_pred <- predict(rf_model, testData)
results[4, 2:6] <- calculate_metrics(testData$Rain, rf_pred)

# 5. Naive Bayes
nb_model <- naiveBayes(Rain ~ ., data = trainData)
nb_pred <- predict(nb_model, testData)
results[5, 2:6] <- calculate_metrics(testData$Rain, nb_pred)

# 6. K-Means
set.seed(123)
kmeans_model <- kmeans(dataset[, features], centers = 2)
kmeans_pred <- as.factor(ifelse(kmeans_model$cluster == 1, 1, 0))
results[6, 2:6] <- calculate_metrics(dataset$Rain, kmeans_pred)

# ---- Results ----

# Print results
print("Model Performance Metrics:")
print(results)

# Plot results
results_long <- melt(results, id.vars = "Model", variable.name = "Metric", value.name = "Value")
ggplot(results_long, aes(x = Model, y = Value, fill = Metric)) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(title = "Model Performance Comparison", x = "Model", y = "Metric Value") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
