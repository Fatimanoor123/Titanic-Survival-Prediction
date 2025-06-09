# 🚢 Titanic Survival Prediction Pipeline

> A complete **machine learning pipeline** for **Titanic survival analysis**, featuring **data preprocessing**, **feature engineering**, and **class balancing** using **SMOTE** and **StandardScaler** in **Python**.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue) ![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2.0-brightgreen) ![imbalanced-learn](https://img.shields.io/badge/imbalanced--learn-0.10.1-yellow) ![License](https://img.shields.io/badge/License-MIT-blue) 

---

## Overview

This project implements a **robust data preprocessing** and **machine learning pipeline** for predicting passenger survival on the Titanic. Leveraging real-world passenger data from Kaggle, it guides you through:

- **Missing value imputation**  
- **Categorical encoding**  
- **Custom feature engineering**  
- **Feature standardization**  
- **Imbalanced class handling** using **SMOTE**  

Ideal for data scientists, ML enthusiasts, and Kaggle competitors learning end-to-end pipeline design.

---

## Features

- **Missing Values Handling**  
  - Prints first rows and null counts  
  - Imputes `Age` with mean value  

- **Categorical Encoding**  
  - One-hot encode `Embarked` (drop first)  
  - Binary code `Sex`  

- **Feature Engineering**  
  - **Family Size**: `SibSp + Parch + 1`  
  - **Ticket Prefix**: Extracts prefix from `Ticket`, cleans non-alphanumeric  

- **Standardization**  
  - Scales `Age` & `Fare` via `StandardScaler`  

- **Class Balancing**  
  - Uses **SMOTE** to oversample minority (`Survived=1`) class  

- **Modularity**  
  - Separate functions for each step, easy reuse  

---

## Tech Stack

- **Python** 3.8+  
- **pandas** for data loading & manipulation  
- **NumPy** for numerical operations  
- **scikit-learn** for preprocessing & scaling  
- **imbalanced-learn** for SMOTE oversampling  

---

## Installation

1. **Clone the repo**  
   ```bash
   git clone https://github.com/yourusername/titanic-survival-pipeline.git
   cd titanic-survival-pipeline
