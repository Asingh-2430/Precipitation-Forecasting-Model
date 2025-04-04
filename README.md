# 🌧️ Precipitation Forecasting Using Machine Learning in R

This project focuses on predicting rainfall based on weather attributes using various supervised and unsupervised machine learning algorithms implemented in **R**. A comprehensive evaluation is performed to compare model performance on classification metrics.

---

## 📁 Project Structure


```
precipitation-forecasting/
├── data/           # Dataset(s) used for training/testing
├── scripts/        # R scripts for preprocessing and training
├── results/        # Output metrics and result tables
├── plots/          # Visualizations and comparative graphs
└── README.md       # Project overview and instructions
```

---

## 🧠 Models Implemented

- 🔹 K-Nearest Neighbors (KNN)  
- 🔹 Support Vector Machine (SVM)  
- 🔹 Decision Tree  
- 🔹 Random Forest  
- 🔹 Naive Bayes  
- 🔹 K-Means Clustering (Unsupervised Baseline)

---

## 📊 Dataset Overview

The dataset includes weather attributes such as:

- 🌡️ Temperature (°C)  
- 🌡️ Apparent Temperature (°C)  
- 💧 Humidity  
- 🌬️ Wind Speed (km/h)  
- 👁️ Visibility (km)  
- 🧭 Pressure (millibars)  
- ☁️ Precipitation Type (Categorical Target)

The target variable `Rain` is derived from `Precip.Type`, where:
- `"rain"` → 1 (Rain)
- Other/No Precipitation → 0 (No Rain)

> All missing values are removed, and feature columns are cleaned and converted to numeric formats.

---

## ⚙️ Workflow

1. Data Cleaning & Feature Selection  
2. Train-Test Split (70-30)  
3. Model Training & Prediction  
4. Metric Calculation:  
   - ✅ Accuracy  
   - 🎯 Precision  
   - 🔁 Recall  
   - 📐 F1 Score  
   - ❌ Error Rate  
5. Result Comparison & Visualization

---

## 📈 Visualization

A grouped bar plot is generated to visualize and compare the performance of all models across different metrics.  
The plot is saved inside the `plots/` directory.

---

## 🚀 Getting Started

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/Asingh-2430/Precipitation-Forcasting-Model.git
cd Precipitation-Forcasting-Model

```

### 2️⃣ Install Required Packages
```r
install.packages(c("caret", "e1071", "class", "randomForest", "rpart", "ggplot2", "reshape2"))
```

### 3️⃣ Run the Training Script

- Open `scripts/model_training.R` in **RStudio**
- Load the dataset via the `file.choose()` dialog or manually specify the path
- Execute the script to:
  - Train all models
  - Output evaluation metrics
  - Generate comparison plots

---

## 📌 Dependencies

This project uses the following R libraries:

- `caret`  
- `e1071`  
- `class`  
- `randomForest`  
- `rpart`  
- `ggplot2`  
- `reshape2`
