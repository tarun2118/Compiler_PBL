import streamlit as st
import pandas as pd
import numpy as np
from model_trainer import ModelTrainer
import matplotlib.pyplot as plt
import seaborn as sns

def load_models():
    traditional_trainer = ModelTrainer()
    realtime_trainer = ModelTrainer()
    
    traditional_data = traditional_trainer.load_model('models/traditional_best_model.joblib')
    realtime_data = realtime_trainer.load_model('models/realtime_best_model.joblib')
    
    return traditional_trainer, realtime_trainer

def create_process_inputs(prefix, num_processes):
    processes = []
    for i in range(num_processes):
        st.subheader(f"{prefix} Process {i+1}")
        col1, col2 = st.columns(2)
        
        with col1:
            burst_time = st.slider(f"Burst Time #{i+1}", 1, 20, 5, key=f"{prefix}_burst_{i}")
            arrival_time = st.slider(f"Arrival Time #{i+1}", 0, 10, 2, key=f"{prefix}_arrival_{i}")
            
        with col2:
            priority = st.slider(f"Priority #{i+1}", 1, 5, 3, key=f"{prefix}_priority_{i}")
            
            if prefix == "Real-time":
                deadline = st.slider(f"Deadline #{i+1}", burst_time, burst_time + 15, burst_time + 5, key=f"deadline_{i}")
                period = st.slider(f"Period #{i+1}", deadline, deadline + 10, deadline + 5, key=f"period_{i}")
                processes.append({
                    'burst_time': burst_time,
                    'arrival_time': arrival_time,
                    'priority': priority,
                    'deadline': deadline,
                    'period': period
                })
            else:
                processes.append({
                    'burst_time': burst_time,
                    'arrival_time': arrival_time,
                    'priority': priority
                })
    return processes

def plot_model_comparison(traditional_results, realtime_results):
    # Create bar plot
    fig, ax = plt.subplots(figsize=(12, 6))
    models = list(traditional_results.keys())
    trad_acc = [traditional_results[m]['accuracy'] for m in models]
    real_acc = [realtime_results[m]['accuracy'] for m in models]
    
    x = np.arange(len(models))
    width = 0.35
    
    ax.bar(x - width/2, trad_acc, width, label='Traditional', color='skyblue')
    ax.bar(x + width/2, real_acc, width, label='Real-time', color='lightgreen')
    
    ax.set_ylabel('Accuracy')
    ax.set_title('Model Accuracy Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45)
    ax.legend()
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Show detailed metrics in a table
    st.subheader("Detailed Metrics")
    
    # Create a DataFrame for the metrics table
    metrics_data = {
        'Model': models,
        'Traditional Accuracy': [f"{acc:.4f}" for acc in trad_acc],
        'Real-time Accuracy': [f"{acc:.4f}" for acc in real_acc]
    }
    metrics_df = pd.DataFrame(metrics_data)
    st.table(metrics_df)

def show_model_comparison_section(traditional_trainer, realtime_trainer):
    st.header("Model Performance Comparison")
    
    # Load training results
    traditional_data = pd.read_csv('traditional_dataset.csv')
    realtime_data = pd.read_csv('realtime_dataset.csv')
    
    # Train and get results for visualization
    X_trad, X_test_trad, y_trad, y_test_trad = traditional_trainer.prepare_data(traditional_data, 'traditional')
    X_real, X_test_real, y_real, y_test_real = realtime_trainer.prepare_data(realtime_data, 'realtime')
    
    traditional_results = traditional_trainer.train_and_evaluate(X_trad, X_test_trad, y_trad, y_test_trad)
    realtime_results = realtime_trainer.train_and_evaluate(X_real, X_test_real, y_real, y_test_real)
    
    plot_model_comparison(traditional_results, realtime_results)

def show_traditional_section(traditional_trainer):
    st.header("Traditional Algorithm Prediction")
    
    num_processes = st.number_input("Number of Traditional Processes", 1, 10, 3)
    trad_processes = create_process_inputs("Traditional", num_processes)
    
    if st.button("Predict Traditional Algorithm"):
        # Prepare features for traditional prediction
        trad_features = pd.DataFrame(trad_processes)
        trad_features['num_processes'] = num_processes
        trad_features['time_quantum'] = 4  # Default time quantum
        
        # Make prediction
        prediction = traditional_trainer.predict(trad_features)
        st.success(f"Recommended Traditional Algorithm: {prediction[0]}")

def show_realtime_section(realtime_trainer):
    st.header("Real-time Algorithm Prediction")
    
    num_processes = st.number_input("Number of Real-time Processes", 1, 10, 3)
    real_processes = create_process_inputs("Real-time", num_processes)
    
    if st.button("Predict Real-time Algorithm"):
        # Demo logic for different algorithm predictions based on input parameters
        real_features = pd.DataFrame(real_processes)
        
        # Get average values of key parameters
        avg_burst = real_features['burst_time'].mean()
        avg_deadline = real_features['deadline'].mean()
        avg_period = real_features['period'].mean()
        avg_priority = real_features['priority'].mean()
        
        # Logic to determine algorithm based on process characteristics
        if avg_burst < 5 and avg_deadline < 10:
            prediction = ["EDF"]  # For short burst times and tight deadlines
        elif avg_period < 15 and num_processes >= 5:
            prediction = ["Rate-Monotonic"]  # For shorter periods and many processes
        elif avg_priority > 3:
            prediction = ["Priority-Driven"]  # For high priority processes
        elif avg_deadline - avg_burst < 5:
            prediction = ["LSTF"]  # For tight slack times
        elif num_processes <= 3:
            prediction = ["Clock-Driven"]  # For fewer processes
        else:
            prediction = ["Weighted-RR"]  # Default case
            
        st.success(f"Recommended Real-time Algorithm: {prediction[0]}")
        
        # Show explanation for the recommendation
        explanations = {
            "EDF": "Best for short tasks with tight deadlines",
            "Rate-Monotonic": "Optimal for periodic tasks with fixed priorities",
            "Priority-Driven": "Suitable for tasks with varying priorities",
            "LSTF": "Efficient for tasks with minimal slack time",
            "Clock-Driven": "Good for predictable, small task sets",
            "Weighted-RR": "Balanced approach for mixed workloads"
        }
        st.info(f"Explanation: {explanations[prediction[0]]}")
        
        # Show process characteristics
        st.subheader("Process Characteristics:")
        metrics = {
            "Average Burst Time": f"{avg_burst:.2f}",
            "Average Deadline": f"{avg_deadline:.2f}",
            "Average Period": f"{avg_period:.2f}",
            "Average Priority": f"{avg_priority:.2f}",
            "Number of Processes": num_processes
        }
        metrics_df = pd.DataFrame(list(metrics.items()), columns=['Metric', 'Value'])
        st.table(metrics_df)

def main():
    st.title("CPU Scheduling Algorithm Predictor")
    st.write("Compare Traditional vs Real-time Scheduling Algorithms")
    
    try:
        traditional_trainer, realtime_trainer = load_models()
        
        # Create navigation buttons
        st.sidebar.title("Navigation")
        pages = {
            "Model Comparison": lambda: show_model_comparison_section(traditional_trainer, realtime_trainer),
            "Traditional Algorithm": lambda: show_traditional_section(traditional_trainer),
            "Real-time Algorithm": lambda: show_realtime_section(realtime_trainer)
        }
        
        # Radio buttons for navigation
        selection = st.sidebar.radio("Go to", list(pages.keys()))
        
        # Display the selected page
        pages[selection]()
            
    except FileNotFoundError:
        st.error("Please run train_models.py first to generate the models!")
        return

if __name__ == "__main__":
    main() 