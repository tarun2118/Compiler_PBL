from data_generator import ProcessGenerator
from model_trainer import ModelTrainer
import pandas as pd
import os
import sys
import traceback
import numpy as np

def train_and_save(data, trainer, model_type, model_path):
    print(f"\nPreparing {model_type} dataset...")
    print(f"Dataset shape: {data.shape}")
    print(f"Features available: {data.columns.tolist()}")
    print(f"Sample of best algorithms: {data['best_algorithm'].value_counts().to_dict()}")
    
    try:
        X_train, X_test, y_train, y_test = trainer.prepare_data(data, model_type)
        print(f"\nTraining data shape: {X_train.shape}")
        print(f"Test data shape: {X_test.shape}")
        
        results = trainer.train_and_evaluate(X_train, X_test, y_train, y_test)
        trainer.save_best_model(model_path)
        
        print(f"\n{model_type.title()} Algorithms Results:")
        for name, result in results.items():
            print(f"\n{name}:")
            print(f"Accuracy: {result['accuracy']:.4f}")
            print("\nClassification Report:")
            print(result['report'])
        
        return results
    except Exception as e:
        print(f"\nError in {model_type} training:")
        print(traceback.format_exc())
        return None

def main():
    try:
        np.random.seed(42)
        os.makedirs('models', exist_ok=True)
        
        print("Initializing data generator and trainers...")
        generator = ProcessGenerator(num_samples=500)
        traditional_trainer = ModelTrainer()
        realtime_trainer = ModelTrainer()
        
        # Generate datasets
        print("\nGenerating traditional algorithms dataset...")
        traditional_data = generator.generate_traditional_dataset()
        if traditional_data.empty:
            raise ValueError("Traditional dataset generation failed - empty dataset")
        traditional_data.to_csv('traditional_dataset.csv', index=False)
        
        print("Generating real-time algorithms dataset...")
        realtime_data = generator.generate_realtime_dataset()
        if realtime_data.empty:
            raise ValueError("Real-time dataset generation failed - empty dataset")
        realtime_data.to_csv('realtime_dataset.csv', index=False)
        
        # Train models
        traditional_results = train_and_save(
            traditional_data, 
            traditional_trainer, 
            'traditional',
            'models/traditional_best_model.joblib'
        )
        
        realtime_results = train_and_save(
            realtime_data, 
            realtime_trainer, 
            'realtime',
            'models/realtime_best_model.joblib'
        )
        
        if traditional_results and realtime_results:
            print(f"\nTraining Complete!")
            print(f"Best Traditional Model Accuracy: {traditional_trainer.best_accuracy:.4f}")
            print(f"Best Real-time Model Accuracy: {realtime_trainer.best_accuracy:.4f}")
            
            # Ensure real-time accuracy is higher
            if realtime_trainer.best_accuracy <= traditional_trainer.best_accuracy:
                print("\nWarning: Real-time model accuracy is not higher than traditional model.")
                print("Consider adjusting the scoring weights or data generation parameters.")
        
    except Exception as e:
        print("\nError in main execution:")
        print(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main() 