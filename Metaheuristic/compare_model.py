import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import time
import sys
from pathlib import Path

try:
    from realtime_algo import (
        Task,
        clock_driven_scheduler,
        weighted_rr_scheduler,
        priority_scheduler,
        edf_scheduler,
        lst_scheduler,
        rms_scheduler,
        evaluate_schedulers
    )
    from pso_algorithm import pso_scheduler
    from generate_set import generate_task_set
except ImportError as e:
    print(f"Error importing required modules: {e}")
    sys.exit(1)

try:
    import joblib
except ImportError:
    print("Error: joblib not found. Please install it using: pip install joblib")
    sys.exit(1)

def extract_features(tasks):
    """Extract all required features from tasks."""
    num_tasks = len(tasks)
    executions = [t.execution for t in tasks]
    priorities = [t.priority for t in tasks]
    deadlines = [t.deadline for t in tasks]
    periods = [t.period for t in tasks]
    
    # Calculate basic features
    avg_execution = np.mean(executions)
    avg_priority = np.mean(priorities)
    avg_deadline = np.mean(deadlines)
    periodic_ratio = sum(1 for p in periods if p > 0) / num_tasks
    utilization = sum(e/p if p > 0 else e/d for e, p, d in zip(executions, periods, deadlines)) / num_tasks
    
    # Create DataFrame with features in the exact order used during training
    features = pd.DataFrame([{
        'num_tasks': num_tasks,
        'avg_execution': avg_execution,
        'avg_priority': avg_priority,
        'avg_deadline': avg_deadline,
        'periodic_ratio': periodic_ratio,
        'utilization': utilization
    }])
    
    # Return features in the exact order they were used in training
    return features[[
        'num_tasks',
        'avg_execution',
        'avg_priority',
        'avg_deadline',
        'periodic_ratio',
        'utilization'
    ]]

def convert_to_tasks(tasks_raw):
    tasks = []
    for t in tasks_raw:
        task = Task(
            pid=t['pid'],
            arrival=t['arrival'],
            execution=t['execution'],
            priority=t['priority'],
            deadline=t['deadline'],
            period=t['period']
        )
        tasks.append(task)
    return tasks

def run_algorithm(algo_name, tasks_raw):
    algo_map = {
        "clock_driven": clock_driven_scheduler,
        "weighted_rr": weighted_rr_scheduler,
        "priority": priority_scheduler,
        "edf": edf_scheduler,
        "lst": lst_scheduler,
        "rms": rms_scheduler,
    }
    tasks = convert_to_tasks(tasks_raw)
    algo_func = algo_map.get(algo_name)
    if not algo_func:
        raise ValueError(f"Unknown algorithm: {algo_name}")
    avg_waiting_time = algo_func(tasks)
    return avg_waiting_time

def run_pso(tasks_raw):
    tasks = convert_to_tasks(tasks_raw)
    avg_waiting_time = pso_scheduler(tasks)
    return avg_waiting_time

def main(iterations=100):
    # Load your saved model here (adjust path as needed)
    model = joblib.load("best_model_realtime.joblib")
    label_to_algo = {
        0: "clock_driven",
        1: "weighted_rr",
        2: "priority",
        3: "edf",
        4: "lst",
        5: "rms",
    }


    pso_better_count = 0

    for i in range(iterations):
        tasks_raw = []
        for pid in range(1, 6):  # 5 tasks per iteration
            task_data = {
                "pid": pid,
                "arrival": random.randint(0, 10),
                "execution": random.randint(1, 10),
                "priority": random.randint(1, 10),
                "deadline": random.randint(10, 20),
                "period": random.randint(10, 20),
            }
            tasks_raw.append(task_data)

        # Extract features to pass to model.predict
        features_df = extract_features(tasks_raw)

        # Predict best scheduler label from model
        predicted_num_label = model.predict(features_df)[0]
        predicted_label = label_to_algo.get(predicted_num_label)
        if predicted_label is None:
            raise ValueError(f"Predicted label {predicted_num_label} not found in label_to_algo mapping.")

        # Run predicted algorithm and PSO to get avg waiting times
        ml_waiting_time = run_algorithm(predicted_label, tasks_raw)
        pso_waiting_time = run_pso(tasks_raw)

        if pso_waiting_time < ml_waiting_time:
            pso_better_count += 1

        print(
            f"Iteration {i+1}: Predicted Algo = {predicted_label}, "
            f"ML Wait = {ml_waiting_time:.2f}, PSO Wait = {pso_waiting_time:.2f}"
        )
        print(f"Features at iteration {i+1}:\n", features_df)


    pso_win_rate = (pso_better_count / iterations) * 100
    print(f"\nPSO was better in {pso_win_rate:.2f}% of the cases.")

    if pso_win_rate >= 85:
        print("PSO meets the threshold of 85%, consider saving PSO model.")
    else:
        print("PSO did not meet the threshold of 85%, continue with ML model.")

def generate_random_tasks(num_tasks, utilization_target):
    """Generate a random set of real-time tasks with controlled utilization."""
    tasks = []
    total_utilization = 0
    
    # Generate periods using log-uniform distribution
    periods = np.exp(np.random.uniform(low=np.log(10), high=np.log(100), size=num_tasks))
    
    # Generate execution times to meet utilization target
    for i in range(num_tasks):
        if i == num_tasks - 1:
            execution = (utilization_target - total_utilization) * periods[i]
        else:
            remaining = utilization_target - total_utilization
            fraction = np.random.uniform(0.1, 0.9)
            execution = (remaining * fraction) * periods[i]
        
        arrival = np.random.uniform(0, 20)
        priority = np.random.randint(1, 10)
        deadline = periods[i]
        
        tasks.append(Task(
            pid=i,
            arrival=arrival,
            execution=max(1, execution),
            priority=priority,
            deadline=max(execution, deadline),
            period=max(0, periods[i])
        ))
        
        total_utilization += execution / periods[i]
    
    return tasks

def compare_schedulers(num_experiments=100):
    """Compare PSO and ML model performance across multiple experiments."""
    # Load ML model artifacts
    model = joblib.load('production_model.joblib')
    scaler = joblib.load('production_scaler.joblib')
    le = joblib.load('production_encoder.joblib')
    
    # Results storage
    results = {
        'PSO': [],
        'ML': [],
        'Utilization': [],
        'NumTasks': []
    }
    
    # Run experiments
    for _ in range(num_experiments):
        # Generate random task set
        num_tasks = np.random.randint(3, 10)
        utilization = np.random.uniform(0.3, 0.9)
        tasks = generate_random_tasks(num_tasks, utilization)
        
        # Get PSO waiting time
        pso_wait = pso_scheduler(tasks)
        
        # Get ML model prediction and waiting time
        features = extract_features(tasks)
        scaled_features = scaler.transform(features)
        predicted_scheduler = le.inverse_transform(model.predict(scaled_features))[0]
        ml_wait = globals()[f"{predicted_scheduler}_scheduler"](tasks)
        
        # Store results
        results['PSO'].append(pso_wait)
        results['ML'].append(ml_wait)
        results['Utilization'].append(utilization)
        results['NumTasks'].append(num_tasks)
    
    return pd.DataFrame(results)

def plot_comparison(results):
    """Create visualizations comparing PSO and ML model performance."""
    if not st.runtime.exists():
        print("Streamlit runtime not detected. Skipping visualization.")
        return
        
    if results is None or len(results) == 0:
        st.error("No valid results to analyze. Please check the error messages above.")
        return
        
    st.title("Real-time Scheduler Comparison: PSO vs ML")
    
    # Summary metrics at the top
    st.header("Summary")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Test Cases", len(results))
    with col2:
        st.metric("PSO Better Cases", (results['performance_ratio'] > 1).sum())
    with col3:
        st.metric("ML Better Cases", (results['performance_ratio'] < 1).sum())
    
    # Waiting Time Comparison
    st.header("Waiting Time Analysis")
    
    # Box plot for waiting times
    fig, ax = plt.subplots(figsize=(10, 6))
    data = {
        'PSO': results['pso_wait_time'],
        'ML Model': results['ml_wait_time']
    }
    ax.boxplot(data.values(), labels=data.keys())
    ax.set_ylabel('Average Waiting Time (s)')
    ax.set_yscale('log')  # Log scale to better show differences
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    plt.close()
    
    # Detailed comparison table
    st.subheader("Detailed Results")
    comparison_df = pd.DataFrame({
        'Test Case': range(1, len(results) + 1),
        'Number of Tasks': results['num_tasks'],
        'PSO Wait Time (s)': results['pso_wait_time'].round(2),
        'ML Wait Time (s)': results['ml_wait_time'].round(2),
        'ML Algorithm': results['ml_scheduler'],
        'Better Algorithm': ['PSO' if r > 1 else 'ML' for r in results['performance_ratio']]
    })
    st.dataframe(comparison_df)
    
    # Computation Time Comparison
    st.header("Computation Time Analysis")
    fig, ax = plt.subplots(figsize=(10, 6))
    data = {
        'PSO': results['pso_time'],
        'ML Model': results['ml_time']
    }
    ax.boxplot(data.values(), labels=data.keys())
    ax.set_ylabel('Computation Time (s)')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    plt.close()
    
    # Statistics
    st.header("Performance Statistics")
    stats = pd.DataFrame({
        'Metric': [
            'Mean Waiting Time (s)',
            'Median Waiting Time (s)',
            'Min Waiting Time (s)',
            'Max Waiting Time (s)',
            'Mean Computation Time (s)'
        ],
        'PSO': [
            results['pso_wait_time'].mean(),
            results['pso_wait_time'].median(),
            results['pso_wait_time'].min(),
            results['pso_wait_time'].max(),
            results['pso_time'].mean()
        ],
        'ML Model': [
            results['ml_wait_time'].mean(),
            results['ml_wait_time'].median(),
            results['ml_wait_time'].min(),
            results['ml_wait_time'].max(),
            results['ml_time'].mean()
        ]
    })
    st.dataframe(stats.round(4))
    
    # ML Algorithm Distribution
    st.header("ML Algorithm Selection")
    algo_counts = results['ml_scheduler'].value_counts()
    fig, ax = plt.subplots(figsize=(10, 6))
    algo_counts.plot(kind='bar', ax=ax)
    ax.set_title('ML Model Algorithm Selection')
    ax.set_xlabel('Algorithm')
    ax.set_ylabel('Number of Times Selected')
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

def evaluate_pso_scheduler(tasks):
    """Evaluate PSO scheduler and return best waiting time."""
    try:
        waiting_time = pso_scheduler(tasks)
        return "pso", waiting_time
    except Exception as e:
        print(f"PSO scheduler evaluation failed: {str(e)}")
        return "pso", float('inf')

def evaluate_ml_scheduler(features, model, scaler, le):
    """Predict scheduler using ML model."""
    # Prepare features
    X = features.copy()
    X_scaled = scaler.transform(X)
    
    # Predict scheduler
    scheduler_idx = model.predict(X_scaled)[0]
    predicted_scheduler = le.inverse_transform([scheduler_idx])[0]
    
    # Skip LST scheduler if predicted
    if predicted_scheduler == 'lst':
        return 'edf'  # Fallback to EDF
        
    return predicted_scheduler

def generate_test_cases(n_samples=10):
    """Generate test cases for comparison."""
    test_cases = []
    
    for _ in range(n_samples):
        num_tasks = np.random.randint(2, 7)
        utilization = np.random.uniform(0.3, 0.9)
        tasks = generate_task_set(num_tasks, utilization)
        
        # Ensure tasks are Task objects
        task_objects = []
        for t in tasks:
            if isinstance(t, tuple):
                task_objects.append(Task(
                    pid=t[0],
                    arrival=t[1],
                    execution=t[2],
                    priority=t[3],
                    deadline=t[4],
                    period=t[5]
                ))
            else:
                task_objects.append(t)
        
        # Calculate features
        features = extract_features(task_objects)
        test_cases.append((task_objects, features))
    
    return test_cases

def compare_schedulers(n_samples=200):
    """Compare PSO and ML model predictions."""
    # Load ML model and preprocessing objects
    print("Loading ML model...")
    try:
        model = joblib.load('new_model.joblib')
        scaler = joblib.load('new_scaler.joblib')
        le = joblib.load('new_encoder.joblib')
    except Exception as e:
        print(f"Error loading ML model: {str(e)}")
        return None
    
    print(f"\nGenerating {n_samples} test cases...")
    test_cases = generate_test_cases(n_samples)
    
    results = []
    successful_cases = 0
    failed_cases = 0
    
    # Create a progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    print("\nEvaluating schedulers...")
    for i, (tasks, features) in enumerate(test_cases):
        # Update progress
        progress = (i + 1) / n_samples
        progress_bar.progress(progress)
        status_text.text(f"Processing test case {i+1}/{n_samples} (Success: {successful_cases}, Failed: {failed_cases})")
        
        try:
            # Get PSO scheduler and waiting time
            pso_start = time.time()
            pso_scheduler_name, pso_wait_time = evaluate_pso_scheduler(tasks.copy())
            pso_time = time.time() - pso_start
            
            if pso_wait_time == float('inf'):
                print(f"\nSkipping test case {i+1}: PSO failed to find solution")
                failed_cases += 1
                continue
            
            # Get ML prediction and evaluate its waiting time
            ml_start = time.time()
            ml_scheduler = evaluate_ml_scheduler(features, model, scaler, le)
            ml_time = time.time() - ml_start
            
            if not ml_scheduler:
                print(f"\nSkipping test case {i+1}: ML prediction failed")
                failed_cases += 1
                continue
            
            # Evaluate ML predicted scheduler's waiting time
            scheduler_func = globals()[f"{ml_scheduler}_scheduler"]
            tasks_copy = [Task(t.pid, t.arrival, t.execution, t.priority, t.deadline, t.period) for t in tasks]
            ml_wait_time = scheduler_func(tasks_copy)
            
            results.append({
                'num_tasks': len(tasks),
                'utilization': features['utilization'].iloc[0],
                'pso_scheduler': pso_scheduler_name,
                'ml_scheduler': ml_scheduler,
                'pso_wait_time': pso_wait_time,
                'ml_wait_time': ml_wait_time,
                'pso_time': pso_time,
                'ml_time': ml_time,
                'performance_ratio': ml_wait_time / pso_wait_time if pso_wait_time > 0 else float('inf')
            })
            
            successful_cases += 1
            
        except Exception as e:
            print(f"\nWarning: Error processing test case {i+1}: {str(e)}")
            failed_cases += 1
            continue
    
    # Clear progress bar and status
    progress_bar.empty()
    status_text.empty()
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    if len(df) == 0:
        print("No valid results to analyze")
        return None
    
    # Calculate and display metrics
    print(f"\nResults Summary ({successful_cases}/{n_samples} successful cases):")
    print("-" * 50)
    print(f"Total successful cases: {successful_cases}")
    print(f"Total failed cases: {failed_cases}")
    print(f"Average PSO waiting time: {df['pso_wait_time'].mean():.2f}s")
    print(f"Average ML waiting time: {df['ml_wait_time'].mean():.2f}s")
    print(f"Average PSO computation time: {df['pso_time'].mean():.2f}s")
    print(f"Average ML computation time: {df['ml_time'].mean():.2f}s")
    print(f"Cases where ML performed better: {(df['performance_ratio'] < 1).sum()}")
    print(f"Cases where PSO performed better: {(df['performance_ratio'] > 1).sum()}")
    
    # Save results for Streamlit visualization
    df.to_csv('comparison_results.csv', index=False)
    
    return df

if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Run comparison with 200 samples
    df = compare_schedulers(200)  # Using 200 samples
    
    # Plot results if we have a Streamlit app
    if st.runtime.exists():
        plot_comparison(df)
