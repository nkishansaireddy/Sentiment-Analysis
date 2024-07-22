import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Load the dataset
def load_data(filepath):
    try:
        return pd.read_csv(filepath, encoding='utf-8')
    except UnicodeDecodeError:
        return pd.read_csv(filepath, encoding='ISO-8859-1')


# Data cleaning: handle missing values and duplicates
def clean_data(df):
    df.drop_duplicates(inplace=True)
    df.drop(['unnamed1', 'Status'], axis=1, inplace=True)  # Dropping irrelevant columns
    df.fillna(df.mean(numeric_only=True), inplace=True)  # Fill missing numerical values with mean
    return df


# Descriptive statistics
def describe_data(df):
    return df.describe(include='all')


# Data visualization
def visualize_data(df):
    # Distribution of Age
    plt.figure(figsize=(10, 6))
    sns.histplot(df['Age'], kde=True)
    plt.title('Age Distribution')
    plt.show()

    # Gender Distribution
    plt.figure(figsize=(10, 6))
    sns.countplot(x='Gender', data=df)
    plt.title('Gender Distribution')
    plt.show()

    # Orders by Product Category
    plt.figure(figsize=(10, 6))
    sns.countplot(y='Product_Category', data=df, order=df['Product_Category'].value_counts().index)
    plt.title('Orders by Product Category')
    plt.show()

    # Amount by Age Group
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Age Group', y='Amount', data=df)
    plt.title('Amount Spent by Age Group')
    plt.show()

    # Correlation matrix for numerical features
    numeric_df = df.select_dtypes(include=[np.number])
    plt.figure(figsize=(12, 8))
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation Matrix')
    plt.show()


# Anomaly detection: Z-score method
def detect_anomalies(df):
    numeric_df = df.select_dtypes(include=[np.number])
    threshold = 3
    z_scores = np.abs((numeric_df - numeric_df.mean()) / numeric_df.std())
    anomalies = df[(z_scores > threshold).any(axis=1)]
    return anomalies


# Main function to run EDA
def main(filepath):
    df = load_data(filepath)
    print("Data Loaded")

    df = clean_data(df)
    print("Data Cleaned")

    print("Descriptive Statistics:")
    print(describe_data(df))

    print("Data Visualization:")
    visualize_data(df)

    anomalies = detect_anomalies(df)
    print("Anomalies Detected:")
    print(anomalies)


# Replace 'your_dataset.csv' with your actual dataset path
if __name__ == "__main__":
    main('dataset.csv')
