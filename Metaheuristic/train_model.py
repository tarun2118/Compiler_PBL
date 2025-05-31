import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

def load_and_preprocess_data():
    """Load and preprocess the dataset."""
    print("Loading dataset...")
    df = pd.read_csv("perfectly_balanced_dataset.csv")
    
    # Separate features and target
    X = df.drop('scheduler', axis=1)
    y = df['scheduler']
    
    # Encode target variable
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    
    return X_scaled, y_encoded, le, scaler

def train_and_evaluate_models(X, y, le):
    """Train and evaluate multiple models."""
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Define simpler and faster models
    models = {
        'Decision Tree': DecisionTreeClassifier(
            max_depth=10,
            random_state=42
        ),
        'KNN': KNeighborsClassifier(
            n_neighbors=5,
            n_jobs=-1
        ),
        'Naive Bayes': GaussianNB(),
        'Logistic Regression': LogisticRegression(
            multi_class='multinomial',
            max_iter=1000,
            n_jobs=-1
        ),
        'SVM': SVC(
            kernel='rbf',
            C=1.0,
            random_state=42
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=100,  # Reduced from 500
            max_depth=10,      # Added max_depth
            n_jobs=-1,
            random_state=42
        )
    }
    
    best_model = None
    best_score = 0
    results = {}
    
    print("\nTraining and evaluating models...")
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Train model
        model.fit(X_train, y_train)
        
        # Evaluate on test set
        test_score = model.score(X_test, y_test)
        y_pred = model.predict(X_test)
        
        # Store results
        results[name] = {
            'test_score': test_score,
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'classification_report': classification_report(
                y_test, y_pred, 
                target_names=le.classes_,
                zero_division=0
            )
        }
        
        print(f"Test Score: {test_score:.4f}")
        print("\nClassification Report:")
        print(results[name]['classification_report'])
        
        if test_score > best_score:
            best_score = test_score
            best_model = model
    
    return best_model, results

def plot_feature_importance(model, feature_names, output_file='feature_importance.png'):
    """Plot feature importance for tree-based models."""
    plt.figure(figsize=(10, 6))
    
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.title('Feature Importances')
        plt.barh(range(len(indices)), importances[indices])
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel('Relative Importance')
        plt.tight_layout()
        plt.savefig(output_file)
        plt.close()
    elif hasattr(model, 'coef_'):
        # For linear models like Logistic Regression
        importances = np.abs(model.coef_).mean(axis=0)
        indices = np.argsort(importances)[::-1]
        
        plt.title('Feature Importances')
        plt.barh(range(len(indices)), importances[indices])
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel('Absolute Coefficient Value')
        plt.tight_layout()
        plt.savefig(output_file)
        plt.close()

def plot_confusion_matrices(results, le, output_file='confusion_matrices.png'):
    """Plot confusion matrices for all models."""
    n_models = len(results)
    n_cols = 3
    n_rows = (n_models + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    axes = axes.flatten()
    
    for idx, (name, res) in enumerate(results.items()):
        sns.heatmap(
            res['confusion_matrix'],
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=le.classes_,
            yticklabels=le.classes_,
            ax=axes[idx]
        )
        axes[idx].set_title(f'{name} Confusion Matrix')
        axes[idx].set_xlabel('Predicted')
        axes[idx].set_ylabel('True')
    
    # Hide empty subplots
    for idx in range(len(results), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

def main():
    try:
        # Load and preprocess data
        print("Starting model training...")
        X, y, le, scaler = load_and_preprocess_data()
        print(f"Loaded data shape: {X.shape}")
        print(f"Number of classes: {len(le.classes_)}")
        print(f"Features: {list(X.columns)}")
        
        # Train and evaluate models
        print("\nStarting model training and evaluation...")
        best_model, results = train_and_evaluate_models(X, y, le)
        print(f"\nBest model: {type(best_model).__name__}")
        
        # Plot results
        print("\nGenerating visualizations...")
        plot_feature_importance(best_model, X.columns)
        plot_confusion_matrices(results, le)
        
        # Save best model and preprocessing objects
        print("\nSaving models and preprocessing objects...")
        joblib.dump(best_model, 'scheduler_model.joblib')
        joblib.dump(scaler, 'scheduler_scaler.joblib')
        joblib.dump(le, 'scheduler_encoder.joblib')
        
        print("\nDone! Models and visualizations have been saved.")
        
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        import traceback
        print("\nFull traceback:")
        traceback.print_exc()

if __name__ == "__main__":
    main() 