import numpy as np
import pandas as pd
from realtime_algo import Task, clock_driven_scheduler, weighted_rr_scheduler, priority_scheduler, edf_scheduler, lst_scheduler, rms_scheduler
from multiprocessing import Pool, cpu_count
from functools import partial
import tqdm

def generate_task_set(num_tasks, utilization_target, scheduler_bias=None):
    """Generate a set of real-time tasks with controlled utilization and scheduler-specific characteristics."""
    tasks = []
    
    if scheduler_bias == 'edf':
        deadline_factor = np.random.uniform(1.1, 1.5, size=num_tasks)
        priority_range = (1, 5)
        period_range = (10, 50)
        periodic_prob = 0.3
    elif scheduler_bias == 'rms':
        deadline_factor = np.ones(num_tasks)  # Deadline = Period for RMS
        priority_range = (1, num_tasks)
        base_period = 10 
        
        periods = []
        for i in range(num_tasks):

            period = base_period * (2 ** np.random.randint(0, 3))  # Powers of 2
            periods.append(period)
        
    
        periods.sort()
        

        n = num_tasks
        rms_bound = n * (2 ** (1/n) - 1)
        target_util = min(utilization_target, rms_bound * 0.9)  # Stay below RMS bound
        
        utils = []
        sum_util = target_util
        for i in range(num_tasks - 1):
            next_util = sum_util * (np.random.random() ** (1.0 / (num_tasks - i)))
            utils.append(sum_util - next_util)
            sum_util = next_util
        utils.append(sum_util)
        
        # Create tasks
        for i in range(num_tasks):
            period = periods[i]
            execution = utils[i] * period
            arrival = np.random.uniform(0, min(5, period * 0.5))
            
            task = Task(
                pid=i,
                arrival=arrival,
                execution=execution,
                priority=num_tasks - i, 
                deadline=period, 
                period=period
            )
            tasks.append(task)
        
        return tasks
    elif scheduler_bias == 'lst':
    
        deadline_factor = np.random.uniform(1.5, 2.5, size=num_tasks)
        priority_range = (1, 10)
        period_range = (15, 45)
        periodic_prob = 0.4
    elif scheduler_bias == 'priority':

        deadline_factor = np.random.uniform(1.5, 2.0, size=num_tasks)
        priority_range = (1, 15)
        period_range = (10, 40)
        periodic_prob = 0.5
    elif scheduler_bias == 'clock_driven':
        # Clock-driven works best with similar periods
        deadline_factor = np.full(num_tasks, 1.5)
        priority_range = (1, 5)
        period_range = (18, 22)  # Close to 20
        periodic_prob = 0.7
    elif scheduler_bias == 'weighted_rr':
        # WRR works best with varied priorities
        deadline_factor = np.full(num_tasks, 1.8)
        priority_range = (1, 5)
        period_range = (15, 35)
        periodic_prob = 0.6
    else:
        deadline_factor = np.random.uniform(1.5, 2.0, size=num_tasks)
        priority_range = (1, 10)
        period_range = (10, 50)
        periodic_prob = 0.5
    
    # If we haven't returned yet, generate tasks normally
    if not tasks:
        # Generate periods based on scheduler type
        periods = np.zeros(num_tasks)
        for i in range(num_tasks):
            if np.random.random() < periodic_prob:
                periods[i] = np.random.uniform(period_range[0], period_range[1])
        
        # Generate execution times with controlled utilization
        fractions = np.random.uniform(0.3, 0.7, num_tasks)
        fractions = fractions / fractions.sum() * utilization_target
        
        executions = np.zeros(num_tasks)
        for i in range(num_tasks):
            if periods[i] > 0:
                executions[i] = fractions[i] * periods[i]
            else:
                # For non-periodic tasks, use a reasonable execution time
                executions[i] = np.random.uniform(1, 5)
        
        executions = np.clip(executions, 1, np.maximum(periods, 1) * 0.8)
        
        # Generate arrivals with controlled spread
        if scheduler_bias in ['lst', 'edf']:
            arrivals = np.random.uniform(0, np.maximum(periods, 20) * 0.3)
        elif scheduler_bias == 'weighted_rr':
            arrivals = np.random.uniform(0, 10, size=num_tasks)
        else:
            arrivals = np.random.uniform(0, np.minimum(15, np.maximum(periods, 15) * 0.3))
        
        # Generate priorities based on scheduler type
        if scheduler_bias == 'priority':
            priorities = np.linspace(priority_range[0], priority_range[1], num_tasks)
            np.random.shuffle(priorities)
        else:
            priorities = np.random.randint(priority_range[0], priority_range[1] + 1, size=num_tasks)
        
        # Calculate deadlines based on periods and deadline factors
        deadlines = np.zeros(num_tasks)
        for i in range(num_tasks):
            if periods[i] > 0:
                deadlines[i] = min(periods[i], executions[i] * deadline_factor[i])
            else:
                # For non-periodic tasks, set deadline relative to execution time
                deadlines[i] = executions[i] * deadline_factor[i]
        
        # Create tasks
        for i in range(num_tasks):
            task = Task(
                pid=i,
                arrival=float(arrivals[i]),
                execution=float(executions[i]),
                priority=int(priorities[i]),
                deadline=float(deadlines[i]),
                period=float(periods[i])
            )
            tasks.append(task)
    
    return tasks

def evaluate_schedulers(tasks, early_stopping=True):
    """Evaluate all scheduling algorithms on a task set with improved metrics and early stopping."""
    results = {}
    schedulers = {
        'clock_driven': clock_driven_scheduler,
        'weighted_rr': weighted_rr_scheduler,
        'priority': priority_scheduler,
        'edf': edf_scheduler,
        'lst': lst_scheduler,
        'rms': rms_scheduler
    }
    
    # Start with simpler schedulers first
    scheduler_order = ['edf', 'rms', 'clock_driven', 'priority', 'lst', 'weighted_rr']
    best_wait_time = float('inf')
    best_scheduler = None
    
    for name in scheduler_order:
        scheduler = schedulers[name]
        # Make a deep copy of tasks
        tasks_copy = [Task(t.pid, t.arrival, t.execution, t.priority, t.deadline, t.period) for t in tasks]
        
        try:
            # Add timeout protection
            wait_time = scheduler(tasks_copy)
            
            # Early stopping if we find a perfect scheduler
            if wait_time == 0:
                return {name: wait_time}, name
            
            # Calculate additional metrics
            completed = sum(1 for t in tasks_copy if t.finish_time is not None)
            if completed < len(tasks):
                wait_time = 1e6
            else:
                missed_deadlines = sum(1 for t in tasks_copy if t.finish_time > t.arrival + t.deadline)
                if missed_deadlines > 0:
                    wait_time *= (1 + missed_deadlines / len(tasks))
            
            results[name] = wait_time
            
            # Early stopping if we found a significantly better scheduler
            if early_stopping and wait_time < best_wait_time * 0.5:
                best_wait_time = wait_time
                best_scheduler = name
                break
        except Exception as e:
            print(f"Warning: Scheduler {name} failed with error: {e}")
            results[name] = float('inf')
    
    if best_scheduler is None:
        best_scheduler = min(results.items(), key=lambda x: x[1])[0]
    
    return results, best_scheduler

def generate_sample(args):
    """Generate a single sample targeting a specific scheduler."""
    target_scheduler = args
    max_attempts = 10  # Maximum attempts per sample
    
    for _ in range(max_attempts):
        try:
            # Generate smaller task sets for better performance
            num_tasks = np.random.randint(3, 7)
            utilization = np.random.uniform(0.3, 0.8)
            
            # Adjust parameters based on target scheduler
            if target_scheduler == 'rms':
                utilization = np.random.uniform(0.3, 0.7)  # RMS bound
            elif target_scheduler == 'lst':
                utilization = np.random.uniform(0.3, 0.6)  # LST needs slack
            elif target_scheduler == 'clock_driven':
                utilization = np.random.uniform(0.3, 0.9)  # Clock-driven can handle higher util
            
            # Generate task set with bias towards target scheduler
            tasks = generate_task_set(num_tasks, utilization, scheduler_bias=target_scheduler)
            
            # Evaluate schedulers with early stopping
            results, best_scheduler = evaluate_schedulers(tasks, early_stopping=True)
            
            # Calculate score difference to determine if this scheduler is significantly better
            best_score = results[best_scheduler]
            target_score = results.get(target_scheduler, float('inf'))
            
            # Accept the result if:
            # 1. This is the target scheduler, or
            # 2. The scores are close enough (within 20%), or
            # 3. We've tried enough times
            if (best_scheduler == target_scheduler or 
                (target_score < float('inf') and target_score/best_score > 0.8) or 
                _ == max_attempts - 1):
                
                # Calculate features
                periodic_tasks = sum(1 for task in tasks if task.period > 0)
                total_tasks = len(tasks)
                
                # Calculate additional features
                avg_slack = np.mean([t.deadline - t.execution for t in tasks])
                priority_range = max(t.priority for t in tasks) - min(t.priority for t in tasks)
                avg_period = np.mean([t.period for t in tasks if t.period > 0]) if any(t.period > 0 for t in tasks) else 0
                
                return {
                    'num_tasks': total_tasks,
                    'avg_execution': float(np.mean([task.execution for task in tasks])),
                    'avg_priority': float(np.mean([task.priority for task in tasks])),
                    'avg_deadline': float(np.mean([task.deadline for task in tasks])),
                    'periodic_ratio': float(periodic_tasks / total_tasks),
                    'utilization': float(utilization),
                    'avg_slack': float(avg_slack),
                    'priority_range': float(priority_range),
                    'avg_period': float(avg_period),
                    'best_scheduler': best_scheduler
                }
        except Exception as e:
            if _ == max_attempts - 1:
                print(f"Warning: Failed to generate sample after {max_attempts} attempts: {str(e)}")
                raise
            continue
    
    return None  # Should never reach here due to the raise above

def generate_dataset(num_samples=1500):
    """Generate a balanced dataset using mathematical properties."""
    samples_per_scheduler = num_samples // 6
    data = []
    
    generators = {
        'rms': generate_rms_taskset,
        'edf': generate_edf_taskset,
        'lst': generate_lst_taskset,
        'priority': generate_priority_taskset,
        'weighted_rr': generate_wrr_taskset,
        'clock_driven': generate_clock_driven_taskset
    }
    
    print("\nGenerating balanced dataset...")
    for scheduler, generator in generators.items():
        print(f"\nGenerating {samples_per_scheduler} samples for {scheduler}...")
        for i in range(samples_per_scheduler):
            _, sample = generator()
            data.append(sample)
            if (i + 1) % 50 == 0:
                print(f"Generated {i + 1} samples")
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Save dataset
    df.to_csv("perfectly_balanced_dataset.csv", index=False)
    print(f"\nGenerated {len(df)} samples with distribution:")
    print(df['scheduler'].value_counts(normalize=True))
    print("\nActual counts:")
    print(df['scheduler'].value_counts())
    
    return df

def generate_rms_taskset():
    """Generate a task set optimized for RMS."""
    num_tasks = np.random.randint(2, 4)
    base_period = 10
    
    # Generate harmonic periods (powers of 2)
    periods = [base_period * (2 ** i) for i in range(num_tasks)]
    
    # Calculate utilization using UUniFast
    util_bound = num_tasks * (2 ** (1/num_tasks) - 1) * 0.9  # Stay below RMS bound
    utils = []
    sum_util = np.random.uniform(0.3, util_bound)
    for i in range(num_tasks - 1):
        next_util = sum_util * (np.random.random() ** (1.0 / (num_tasks - i)))
        utils.append(sum_util - next_util)
        sum_util = next_util
    utils.append(sum_util)
    
    # Create tasks
    tasks = []
    for i in range(num_tasks):
        execution = utils[i] * periods[i]
        tasks.append(Task(
            pid=i,
            arrival=np.random.uniform(0, 5),
            execution=execution,
            priority=num_tasks - i,  # Higher priority for shorter periods
            deadline=periods[i],
            period=periods[i]
        ))
    
    return tasks, {
        'num_tasks': num_tasks,
        'avg_execution': float(np.mean([t.execution for t in tasks])),
        'avg_priority': float(np.mean([t.priority for t in tasks])),
        'avg_deadline': float(np.mean([t.deadline for t in tasks])),
        'periodic_ratio': 1.0,
        'utilization': float(sum(utils)),
        'scheduler': 'rms'
    }

def generate_edf_taskset():
    """Generate a task set optimized for EDF."""
    num_tasks = np.random.randint(3, 6)
    
    # Mix of periodic and aperiodic tasks
    periodic_count = np.random.randint(1, num_tasks)
    
    # Generate varied periods
    periods = np.zeros(num_tasks)
    for i in range(periodic_count):
        periods[i] = np.random.uniform(20, 100)
    
    # Calculate utilization using UUniFast
    utils = []
    sum_util = np.random.uniform(0.5, 0.9)
    for i in range(num_tasks - 1):
        next_util = sum_util * (np.random.random() ** (1.0 / (num_tasks - i)))
        utils.append(sum_util - next_util)
        sum_util = next_util
    utils.append(sum_util)
    
    # Create tasks
    tasks = []
    for i in range(num_tasks):
        if periods[i] > 0:
            execution = utils[i] * periods[i]
            deadline = np.random.uniform(execution * 1.2, periods[i])
        else:
            execution = np.random.uniform(1, 5)
            deadline = execution * np.random.uniform(1.2, 2.0)
        
        tasks.append(Task(
            pid=i,
            arrival=np.random.uniform(0, 20),
            execution=execution,
            priority=np.random.randint(1, 5),
            deadline=deadline,
            period=periods[i]
        ))
    
    return tasks, {
        'num_tasks': num_tasks,
        'avg_execution': float(np.mean([t.execution for t in tasks])),
        'avg_priority': float(np.mean([t.priority for t in tasks])),
        'avg_deadline': float(np.mean([t.deadline for t in tasks])),
        'periodic_ratio': float(periodic_count / num_tasks),
        'utilization': float(sum(utils)),
        'scheduler': 'edf'
    }

def generate_lst_taskset():
    """Generate a task set optimized for LST."""
    num_tasks = np.random.randint(3, 5)
    
    # Generate tasks with good slack time
    tasks = []
    total_util = 0
    
    for i in range(num_tasks):
        execution = np.random.uniform(1, 5)
        slack = np.random.uniform(2, 10)  # Significant slack time
        deadline = execution + slack
        period = np.random.uniform(deadline * 1.1, deadline * 1.5) if np.random.random() < 0.7 else 0
        
        if period > 0:
            total_util += execution / period
        
        tasks.append(Task(
            pid=i,
            arrival=np.random.uniform(0, 10),
            execution=execution,
            priority=np.random.randint(1, 5),
            deadline=deadline,
            period=period
        ))
    
    return tasks, {
        'num_tasks': num_tasks,
        'avg_execution': float(np.mean([t.execution for t in tasks])),
        'avg_priority': float(np.mean([t.priority for t in tasks])),
        'avg_deadline': float(np.mean([t.deadline for t in tasks])),
        'periodic_ratio': float(len([t for t in tasks if t.period > 0]) / num_tasks),
        'utilization': float(total_util),
        'scheduler': 'lst'
    }

def generate_priority_taskset():
    """Generate a task set optimized for fixed priority."""
    num_tasks = np.random.randint(3, 5)
    
    # Generate clear priority levels
    priorities = np.linspace(1, 10, num_tasks, dtype=int)
    
    tasks = []
    total_util = 0
    
    for i in range(num_tasks):
        execution = np.random.uniform(1, 5)
        deadline = execution * np.random.uniform(1.5, 3.0)
        period = deadline * np.random.uniform(1.1, 1.5) if np.random.random() < 0.5 else 0
        
        if period > 0:
            total_util += execution / period
        
        tasks.append(Task(
            pid=i,
            arrival=np.random.uniform(0, 15),
            execution=execution,
            priority=priorities[i],
            deadline=deadline,
            period=period
        ))
    
    return tasks, {
        'num_tasks': num_tasks,
        'avg_execution': float(np.mean([t.execution for t in tasks])),
        'avg_priority': float(np.mean([t.priority for t in tasks])),
        'avg_deadline': float(np.mean([t.deadline for t in tasks])),
        'periodic_ratio': float(len([t for t in tasks if t.period > 0]) / num_tasks),
        'utilization': float(total_util),
        'scheduler': 'priority'
    }

def generate_wrr_taskset():
    """Generate a task set optimized for weighted round robin."""
    num_tasks = np.random.randint(3, 5)
    
    # Generate varied priorities (weights)
    priorities = np.random.randint(1, 5, size=num_tasks)
    
    tasks = []
    total_util = 0
    
    for i in range(num_tasks):
        execution = priorities[i] * np.random.uniform(1, 3)  # Execution proportional to priority
        deadline = execution * np.random.uniform(2.0, 4.0)
        period = deadline * np.random.uniform(1.1, 1.5) if np.random.random() < 0.6 else 0
        
        if period > 0:
            total_util += execution / period
        
        tasks.append(Task(
            pid=i,
            arrival=np.random.uniform(0, 10),
            execution=execution,
            priority=priorities[i],
            deadline=deadline,
            period=period
        ))
    
    return tasks, {
        'num_tasks': num_tasks,
        'avg_execution': float(np.mean([t.execution for t in tasks])),
        'avg_priority': float(np.mean([t.priority for t in tasks])),
        'avg_deadline': float(np.mean([t.deadline for t in tasks])),
        'periodic_ratio': float(len([t for t in tasks if t.period > 0]) / num_tasks),
        'utilization': float(total_util),
        'scheduler': 'weighted_rr'
    }

def generate_clock_driven_taskset():
    """Generate a task set optimized for clock-driven scheduling."""
    num_tasks = np.random.randint(2, 4)
    base_period = 20
    
    # Generate similar periods
    periods = np.random.normal(base_period, 2, size=num_tasks)
    periods = np.clip(periods, base_period - 5, base_period + 5)
    
    tasks = []
    total_util = 0
    
    for i in range(num_tasks):
        execution = np.random.uniform(1, periods[i] * 0.3)
        total_util += execution / periods[i]
        
        tasks.append(Task(
            pid=i,
            arrival=np.random.uniform(0, 5),
            execution=execution,
            priority=np.random.randint(1, 3),
            deadline=periods[i],
            period=periods[i]
        ))
    
    return tasks, {
        'num_tasks': num_tasks,
        'avg_execution': float(np.mean([t.execution for t in tasks])),
        'avg_priority': float(np.mean([t.priority for t in tasks])),
        'avg_deadline': float(np.mean([t.deadline for t in tasks])),
        'periodic_ratio': 1.0,
        'utilization': float(total_util),
        'scheduler': 'clock_driven'
    }

if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate dataset
    df = generate_dataset(num_samples=1500)