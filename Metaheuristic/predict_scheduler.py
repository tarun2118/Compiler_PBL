import streamlit as st
import pandas as pd
import numpy as np
import joblib
from realtime_algo import Task, clock_driven_scheduler, weighted_rr_scheduler, priority_scheduler, edf_scheduler, lst_scheduler, rms_scheduler
from pso_algorithm import pso_scheduler

# Load the saved model and preprocessing objects
@st.cache_resource
def load_model():
    try:
        model = joblib.load('new_model.joblib')
        scaler = joblib.load('new_scaler.joblib')
        encoder = joblib.load('new_encoder.joblib')
        return model, scaler, encoder
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None

def extract_features(tasks):
    """Extract features from tasks for model prediction."""
    num_tasks = len(tasks)
    executions = [t.execution for t in tasks]
    priorities = [t.priority for t in tasks]
    deadlines = [t.deadline for t in tasks]
    periods = [t.period for t in tasks]
    
    # Calculate features
    avg_execution = np.mean(executions)
    avg_priority = np.mean(priorities)
    avg_deadline = np.mean(deadlines)
    periodic_ratio = sum(1 for p in periods if p > 0) / num_tasks
    utilization = sum(e/p if p > 0 else e/d for e, p, d in zip(executions, periods, deadlines)) / num_tasks
    
    # Create DataFrame with features
    features = pd.DataFrame([{
        'num_tasks': num_tasks,
        'avg_execution': avg_execution,
        'avg_priority': avg_priority,
        'avg_deadline': avg_deadline,
        'periodic_ratio': periodic_ratio,
        'utilization': utilization
    }])
    
    return features

def evaluate_scheduler(scheduler_name, tasks):
    """Evaluate a specific scheduler on the given tasks."""
    scheduler_map = {
        "clock_driven": clock_driven_scheduler,
        "weighted_rr": weighted_rr_scheduler,
        "priority": priority_scheduler,
        "edf": edf_scheduler,
        "lst": lst_scheduler,
        "rms": rms_scheduler,
        "pso": pso_scheduler
    }
    
    scheduler = scheduler_map.get(scheduler_name)
    if not scheduler:
        return float('inf')
    
    try:
        # Create a deep copy of tasks with proper Task objects
        tasks_copy = []
        for t in tasks:
            if isinstance(t, tuple):
                # If task is a tuple, convert to Task object
                # Assuming order: pid, arrival, execution, priority, deadline, period
                task_copy = Task(
                    pid=t[0] if isinstance(t[0], (int, float)) else 0,
                    arrival=float(t[1]),
                    execution=float(t[2]),
                    priority=int(t[3]),
                    deadline=float(t[4]),
                    period=float(t[5])
                )
            elif isinstance(t, list):
                # If task is a list, convert to Task object
                task_copy = Task(
                    pid=t[0] if isinstance(t[0], (int, float)) else 0,
                    arrival=float(t[1]),
                    execution=float(t[2]),
                    priority=int(t[3]),
                    deadline=float(t[4]),
                    period=float(t[5])
                )
            elif isinstance(t, dict):
                # If task is a dictionary
                task_copy = Task(
                    pid=t['pid'] if isinstance(t['pid'], (int, float)) else 0,
                    arrival=float(t['arrival']),
                    execution=float(t['execution']),
                    priority=int(t['priority']),
                    deadline=float(t['deadline']),
                    period=float(t['period'])
                )
            elif isinstance(t, Task):
                # If already a Task object, create a new copy
                task_copy = Task(
                    pid=t.pid,
                    arrival=t.arrival,
                    execution=t.execution,
                    priority=t.priority,
                    deadline=t.deadline,
                    period=t.period
                )
            else:
                raise ValueError(f"Unknown task type: {type(t)}")
            tasks_copy.append(task_copy)
        
        # Run the scheduler
        avg_waiting_time = scheduler(tasks_copy)
        return avg_waiting_time
    except Exception as e:
        print(f"Error evaluating {scheduler_name}: {str(e)}")
        return float('inf')

def main():
    st.title("Real-time Task Scheduler Predictor")
    st.write("Enter task parameters to find the best scheduling algorithm")
    
    # Load model
    model, scaler, encoder = load_model()
    if not model:
        st.error("Failed to load model. Please ensure model files exist.")
        return
    
    # Task input section
    st.subheader("Task Parameters")
    num_tasks = st.slider("Number of Tasks", min_value=2, max_value=10, value=3)
    
    tasks = []
    with st.form("task_form"):
        for i in range(num_tasks):
            st.write(f"\nTask {i+1}")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                arrival = st.number_input(f"Arrival Time (Task {i+1})", 
                                       min_value=0.0, 
                                       max_value=100.0, 
                                       value=float(i*2))
                
                execution = st.number_input(f"Execution Time (Task {i+1})", 
                                         min_value=1.0, 
                                         max_value=50.0, 
                                         value=float(np.random.randint(1, 10)))
            
            with col2:
                priority = st.number_input(f"Priority (Task {i+1})", 
                                        min_value=1, 
                                        max_value=10, 
                                        value=np.random.randint(1, 10))
                
                deadline = st.number_input(f"Deadline (Task {i+1})", 
                                        min_value=float(execution), 
                                        max_value=100.0, 
                                        value=float(np.random.randint(10, 20)))
            
            with col3:
                period = st.number_input(f"Period (Task {i+1})", 
                                      min_value=0.0, 
                                      max_value=100.0, 
                                      value=float(np.random.randint(10, 20)))
            
            task = Task(
                pid=i+1,
                arrival=arrival,
                execution=execution,
                priority=priority,
                deadline=deadline,
                period=period
            )
            tasks.append(task)
        
        submitted = st.form_submit_button("Predict Best Scheduler")
    
    if submitted:
        # Extract features
        features = extract_features(tasks)
        
        # Scale features
        X_scaled = scaler.transform(features)
        
        # Predict scheduler
        predicted_idx = model.predict(X_scaled)[0]
        predicted_scheduler = encoder.inverse_transform([predicted_idx])[0]
        
        # Evaluate all schedulers
        results = {}
        for scheduler_name in ["clock_driven", "weighted_rr", "priority", "edf", "lst", "rms", "pso"]:
            waiting_time = evaluate_scheduler(scheduler_name, tasks)
            results[scheduler_name] = waiting_time
        
        # Display results
        st.subheader("Prediction Results")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("ML Model Prediction:")
            st.info(f"Predicted Best Scheduler: {predicted_scheduler}")
            st.write(f"Predicted Waiting Time: {results[predicted_scheduler]:.2f}s")
        
        with col2:
            st.write("Actual Best Scheduler:")
            best_scheduler = min(results.items(), key=lambda x: x[1])
            st.success(f"Best Scheduler: {best_scheduler[0]}")
            st.write(f"Best Waiting Time: {best_scheduler[1]:.2f}s")
        
        # Show all results
        st.subheader("All Scheduler Results")
        results_df = pd.DataFrame({
            'Scheduler': list(results.keys()),
            'Average Waiting Time (s)': list(results.values())
        })
        results_df = results_df.sort_values('Average Waiting Time (s)')
        st.dataframe(results_df)
        
        # Visualization
        st.subheader("Waiting Time Comparison")
        chart_data = pd.DataFrame({
            'Scheduler': list(results.keys()),
            'Waiting Time (s)': list(results.values())
        })
        st.bar_chart(chart_data.set_index('Scheduler'))

if __name__ == "__main__":
    main() 