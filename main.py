# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def load_data(file_path):
    """Load the Titanic dataset"""
    try:
        data = pd.read_csv(file_path)
        return data
    except FileNotFoundError:
        print("File not found. Make sure the file path is correct.")
        return None
    except Exception as e:
        print("An error occurred while loading the dataset:", e)
        return None

def preprocess_data(data):
    """Preprocess the dataset"""
    if data is None:
        return None
    
    # Handling missing values
    data['Age'].fillna(data['Age'].median(), inplace=True)
    data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)
    data.drop(columns=['Cabin'], inplace=True)
    
    # Encoding categorical variables
    label_encoder = LabelEncoder()
    data['Sex'] = label_encoder.fit_transform(data['Sex'])
    data['Embarked'] = label_encoder.fit_transform(data['Embarked'])
    
    return data

def train_model(X_train, Y_train):
    """Train the logistic regression model"""
    model = LogisticRegression(random_state=42)
    model.fit(X_train, Y_train)
    return model

def evaluate_model(model, X_val, Y_val):
    """Evaluate the model"""
    Y_pred = model.predict(X_val)
    accuracy = accuracy_score(Y_val, Y_pred)
    print(f'Accuracy:{accuracy:.2f}')
    print(classification_report(Y_val, Y_pred))
    conf_matrix = confusion_matrix(Y_val, Y_pred)
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Prediction')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()
    
    print("Predictions for the validation set:")
    for i, prediction in enumerate(Y_pred):
        print(f"Sample {i+1}: {'Survived' if prediction == 1 else 'Not Survived'}")

def main():
    # Load the dataset
    file_path = "Titanic-Dataset.csv"  # Update with correct file path
    data = load_data(file_path)
    if data is None:
        return
    
    # Preprocess the dataset
    data = preprocess_data(data)
    if data is None:
        return
    
    # Define features and target variable
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    X = data[features]
    Y = data['Survived']
    
    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Split the dataset into train and validation sets
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)
    
    # Train the model
    model = train_model(X_train, Y_train)
    
    # Evaluate the model
    evaluate_model(model, X_val, Y_val)

if __name__ == "__main__":
    main()
