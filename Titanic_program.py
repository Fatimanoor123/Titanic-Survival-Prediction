import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE


def missing_values(df):
    print(df.head())
    print(df.isnull().sum())
    df['Age'].fillna(df['Age'].mean(), inplace=True)
    return df


def encoding(df):
    df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)
    df['Sex'] = df['Sex'].astype('category').cat.codes
    return df


def feature_engineering(df):
    df["family_size"] = df['SibSp'] + df['Parch'] + 1
    df["ticket_prefix"] = df["Ticket"].str.split().str[0]
    df["ticket_prefix"] = df["ticket_prefix"].fillna(df["Ticket"])
    df["ticket_prefix"] = df["ticket_prefix"].str.replace("[^A-Za-z0-9\s]", "", regex=True)
    return df


def standardization(df):
    scaler = StandardScaler()
    df[['Age', 'Fare']] = scaler.fit_transform(df[['Age', 'Fare']])
    return df


def balance_classes(X, y):
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X, y)
    print("After resampling class distribution:")
    print(pd.Series(y_res).value_counts())
    return X_res, y_res


# Load the data
df = pd.read_csv('Titanic-Dataset.csv')

# Step-by-step transformation
df = missing_values(df)
df = encoding(df)
df = feature_engineering(df)
df = standardization(df)

# Drop irrelevant/non-numeric columns
df.drop(['Name', 'Ticket', 'Cabin', 'PassengerId', 'ticket_prefix'], axis=1, inplace=True)

# Prepare X and y
X = df.drop('Survived', axis=1)
y = df['Survived']

# Balance classes
X_balanced, y_balanced = balance_classes(X, y)
