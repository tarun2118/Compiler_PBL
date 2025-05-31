import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report
import joblib

class ModelTrainer:
    def __init__(self):
    
        self.models = {
            'Random Forest': RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42), 
            'Decision Tree': DecisionTreeClassifier(max_depth=8, random_state=42),  # Limited depth
            'Naive Bayes': GaussianNB(),  
            'Logistic Regression': LogisticRegression(max_iter=500, random_state=42),
            'KNN': KNeighborsClassifier(n_neighbors=3),  # Reduced neighbors
            'SVM': SVC(kernel='rbf', C=1.0, random_state=42) 
        }
        self.best_model = None
        self.best_accuracy = 0
        self.scaler = StandardScaler()
    
    def prepare_data(self, df: pd.DataFrame, algorithm_type: str):
        if algorithm_type == 'traditional':
            X = df[['burst_time', 'arrival_time', 'priority', 'num_processes', 'time_quantum']]
            y = df['best_algorithm']
        else: 
            X = df[['burst_time', 'arrival_time', 'priority', 'deadline', 'period']]
            y = df['best_algorithm']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_and_evaluate(self, X_train, X_test, y_train, y_test):
        results = {}
        
        for name, model in self.models.items():

            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)
            
            accuracy = accuracy_score(y_test, y_pred)
            results[name] = {
                'accuracy': accuracy,
                'model': model,
                'report': classification_report(y_test, y_pred)
            }
            
            if accuracy > self.best_accuracy:
                self.best_accuracy = accuracy
                self.best_model = model
        
        return results
    
    def save_best_model(self, filename: str):
        if self.best_model is not None:
            model_data = {
                'model': self.best_model,
                'scaler': self.scaler,
                'accuracy': self.best_accuracy
            }
            joblib.dump(model_data, filename)
    
    def load_model(self, filename: str):
        model_data = joblib.load(filename)
        self.best_model = model_data['model']
        self.scaler = model_data['scaler']
        self.best_accuracy = model_data['accuracy']
        return model_data
    
    def predict(self, features: pd.DataFrame):
        if self.best_model is None:
            raise ValueError("No model has been trained or loaded yet!")
        
        scaled_features = self.scaler.transform(features)
        return self.best_model.predict(scaled_features) 