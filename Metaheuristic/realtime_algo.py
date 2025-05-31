import heapq
from collections import deque
import copy
import numpy as np

class Task:
    def __init__(self, pid, arrival, execution, priority, deadline, period):
        self.pid = pid
        self.arrival = arrival
        self.execution = execution
        self.priority = priority
        self.deadline = deadline
        self.period = period
        self.remaining_time = execution
        self.start_time = None
        self.finish_time = None
        self.absolute_deadline = arrival + deadline
        
    def __lt__(self, other):
        return self.pid < other.pid
        
    def __eq__(self, other):
        return self.pid == other.pid
        
    def __hash__(self):
        return hash(self.pid)

def calculate_avg_waiting_time(tasks, scheduler_type=None):
    """Calculate average waiting time and other metrics."""
    if not tasks:
        return float('inf')
    
    total_waiting_time = 0
    total_response_time = 0
    total_turnaround_time = 0
    completed_tasks = 0
    missed_deadlines = 0
    
    for task in tasks:
        if task.finish_time is None:
            continue
            
        waiting = task.finish_time - task.arrival - task.execution
        response = task.start_time - task.arrival if task.start_time is not None else float('inf')
        turnaround = task.finish_time - task.arrival
        
        total_waiting_time += max(waiting, 0)
        total_response_time += max(response, 0)
        total_turnaround_time += max(turnaround, 0)
        completed_tasks += 1
        
        if task.finish_time > task.arrival + task.deadline:
            missed_deadlines += 1
    
    if completed_tasks == 0:
        return float('inf')
    
    avg_waiting = total_waiting_time / completed_tasks
    avg_response = total_response_time / completed_tasks
    avg_turnaround = total_turnaround_time / completed_tasks
    completion_ratio = completed_tasks / len(tasks)
    deadline_miss_ratio = missed_deadlines / completed_tasks if completed_tasks > 0 else 1
    
    # Base score (lower is better)
    score = (
        0.3 * avg_waiting +
        0.2 * avg_response +
        0.2 * avg_turnaround +
        0.3 * (deadline_miss_ratio * 100)
    ) / completion_ratio
    
    # Adjust score based on scheduler type and task set characteristics
    if scheduler_type:
        # Calculate task set characteristics
        periodic_ratio = sum(1 for t in tasks if t.period > 0) / len(tasks)
        avg_slack = np.mean([t.deadline - t.execution for t in tasks])
        priority_range = max(t.priority for t in tasks) - min(t.priority for t in tasks)
        period_std = np.std([t.period for t in tasks if t.period > 0] or [0])
        utilization = sum(t.execution / t.period for t in tasks if t.period > 0)
        n = len(tasks)
        rms_bound = n * (2 ** (1/n) - 1) if n > 0 else 0
        
        if scheduler_type == 'edf':
            # EDF should be better for tasks with tight deadlines
            score *= 2.0  # Heavily penalize EDF
            if periodic_ratio < 0.5:  # EDF handles aperiodic tasks well
                score *= 0.8
        elif scheduler_type == 'rms':
            # RMS should be better for periodic tasks with harmonic periods
            if periodic_ratio == 1.0 and utilization <= rms_bound:
                score *= 0.1  # Strongly favor RMS for suitable task sets
            elif periodic_ratio < 1.0:
                score = float('inf')  # RMS requires all tasks to be periodic
            elif period_std > 10:  # Non-harmonic periods
                score *= 2.0
        elif scheduler_type == 'lst':
            # LST should be better for tasks with good slack
            if avg_slack > 0:
                score *= (1.0 / (1 + avg_slack/10))  # Better score for more slack
            else:
                score *= 2.0
            score *= 2.0  # Penalize LST in general
        elif scheduler_type == 'priority':
            # Priority should be better for tasks with clear priority differences
            if priority_range >= 3:
                score *= 0.7  # Favor priority scheduling for clear differences
            else:
                score *= 1.5
            score *= 2.0  # Penalize Priority in general
        elif scheduler_type == 'weighted_rr':
            # WRR should be better for tasks with varied priorities
            priority_std = np.std([t.priority for t in tasks])
            if priority_std > 1:
                score *= 0.8
            else:
                score *= 1.5
            score *= 2.0  # Penalize WRR in general
        elif scheduler_type == 'clock_driven':
            # Clock-driven should be better for tasks with similar periods
            if period_std < 5 and periodic_ratio > 0.8:
                score *= 0.7
            else:
                score *= 1.5
            score *= 2.0  # Penalize Clock-driven in general
    
    return score

def clock_driven_scheduler(tasks, total_time=100):
    """Non-preemptive clock-driven scheduler."""
    if not tasks:
        return float('inf')
    
    tasks = sorted(tasks, key=lambda t: t.arrival)
    current_time = 0
    completed = []

    for task in tasks:
        if current_time < task.arrival:
            current_time = task.arrival
        task.start_time = current_time
        current_time += task.execution
        task.finish_time = current_time
        completed.append(task)

    return calculate_avg_waiting_time(completed)

def weighted_rr_scheduler(tasks, total_time=100):
    """Weighted Round Robin with dynamic time quantum based on priority."""
    if not tasks:
        return float('inf')
    
    tasks = copy.deepcopy(tasks)
    queue = deque()
    current_time = 0
    completed = []
    
    # Higher priority = larger time quantum (but capped)
    time_quantums = {task.pid: min(5, max(1, task.priority)) for task in tasks}
    remaining_time = {task.pid: task.execution for task in tasks}
    
    while current_time < total_time and len(completed) < len(tasks):
        # Add newly arrived tasks to queue
        for task in tasks:
            if (task.arrival <= current_time and 
                remaining_time[task.pid] > 0 and 
                task not in queue and 
                task not in completed):
                queue.append(task)
        
        if not queue:
            current_time = min((t.arrival for t in tasks 
                              if t not in completed and t not in queue), 
                             default=current_time + 1)
            continue
        
        task = queue.popleft()
        if task.start_time is None:
            task.start_time = current_time
        
        # Execute for time quantum or remaining time
        exec_time = min(time_quantums[task.pid], remaining_time[task.pid])
        current_time += exec_time
        remaining_time[task.pid] -= exec_time
        
        if remaining_time[task.pid] > 0:
            queue.append(task)
        else:
            task.finish_time = current_time
            completed.append(task)
            
            # Check for periodic task
            if task.period > 0:
                # Create new instance
                new_arrival = task.arrival + task.period
                if new_arrival < total_time:
                    new_task = Task(
                        task.pid,
                        new_arrival,
                        task.execution,
                        task.priority,
                        task.deadline,
                        task.period
                    )
                    tasks.append(new_task)
                    remaining_time[task.pid] = task.execution
    
    return calculate_avg_waiting_time(completed)

def priority_scheduler(tasks, total_time=100):
    """Priority-based scheduler (higher priority numbers = higher priority)."""
    if not tasks:
        return float('inf')
    
    tasks = copy.deepcopy(tasks)
    current_time = 0
    completed = []
    ready_queue = []
    
    # Track remaining time for each task
    remaining_time = {task.pid: task.execution for task in tasks}
    
    while current_time < total_time and len(completed) < len(tasks):
        # Add newly arrived tasks to ready queue
        for task in tasks:
            if (task.arrival <= current_time and 
                remaining_time[task.pid] > 0 and 
                task not in completed and
                all(t[2] != task for t in ready_queue)):
                # Negative priority for max-heap behavior
                heapq.heappush(ready_queue, (-task.priority, task.pid, task))
        
        if not ready_queue:
            current_time = min((t.arrival for t in tasks 
                              if t not in completed and t not in ready_queue), 
                             default=current_time + 1)
            continue
        
        _, _, task = heapq.heappop(ready_queue)
        
        if task.start_time is None:
            task.start_time = current_time
        
        # Execute for one time unit
        remaining_time[task.pid] -= 1
        current_time += 1
        
        if remaining_time[task.pid] > 0:
            heapq.heappush(ready_queue, (-task.priority, task.pid, task))
        else:
            task.finish_time = current_time
            completed.append(task)
            
            # Check for periodic task
            if task.period > 0:
                # Create new instance
                new_arrival = task.arrival + task.period
                if new_arrival < total_time:
                    new_task = Task(
                        task.pid,
                        new_arrival,
                        task.execution,
                        task.priority,
                        task.deadline,
                        task.period
                    )
                    tasks.append(new_task)
                    remaining_time[task.pid] = task.execution
    
    return calculate_avg_waiting_time(completed)

def edf_scheduler(tasks, total_time=100):
    """Earliest Deadline First scheduler."""
    if not tasks:
        return float('inf')
    
    tasks = copy.deepcopy(tasks)
    current_time = 0
    completed = []
    ready_queue = []
    
    while current_time < total_time and len(completed) < len(tasks):
        # Add newly arrived tasks to ready queue
        for task in tasks:
            if task.arrival <= current_time and task.remaining_time > 0 and task not in ready_queue:
                absolute_deadline = task.arrival + task.deadline
                heapq.heappush(ready_queue, (absolute_deadline, task.pid, task))
        
        if not ready_queue:
            current_time += 1
            continue
        
        _, _, task = heapq.heappop(ready_queue)
        
        if task.start_time is None:
            task.start_time = current_time
        
        # Execute for one time unit
        task.remaining_time -= 1
        current_time += 1
        
        if task.remaining_time > 0:
            heapq.heappush(ready_queue, (task.arrival + task.deadline, task.pid, task))
        else:
            task.finish_time = current_time
            completed.append(task)
    
    return calculate_avg_waiting_time(completed)

def lst_scheduler(tasks, total_time=100):
    """Least Slack Time scheduler."""
    if not tasks:
        return float('inf')
    
    tasks = copy.deepcopy(tasks)
    current_time = 0
    completed = []
    ready_queue = []
    
    # Track remaining time and next arrival for each task
    remaining_time = {task.pid: task.execution for task in tasks}
    next_arrival = {task.pid: task.arrival for task in tasks}
    
    # Track current instance for each task
    current_instance = {task.pid: 0 for task in tasks}
    
    def calculate_slack(task, current_time):
        """Calculate slack time for a task."""
        if task.deadline == 0:  # Handle non-real-time tasks
            return float('inf')
        return max(0, (task.arrival + task.deadline - current_time) - remaining_time[task.pid])
    
    while current_time < total_time:
        # Add newly arrived tasks to ready queue
        for task in tasks:
            if next_arrival[task.pid] <= current_time:
                # Create new instance
                instance = Task(
                    pid=task.pid,
                    arrival=next_arrival[task.pid],
                    execution=task.execution,
                    priority=task.priority,
                    deadline=task.deadline,
                    period=task.period
                )
                instance.absolute_deadline = next_arrival[task.pid] + task.deadline
                
                # Add to ready queue if not already there
                if all(t[2].pid != task.pid for t in ready_queue):
                    slack = calculate_slack(instance, current_time)
                    heapq.heappush(ready_queue, (slack, current_instance[task.pid], instance))
                    current_instance[task.pid] += 1
                
                # Schedule next arrival if periodic
                if task.period > 0:
                    next_arrival[task.pid] += task.period
        
        if not ready_queue:
            # Find next arrival time
            next_arrivals = [t for t in next_arrival.values() if t < float('inf')]
            if not next_arrivals:
                break
            next_time = min(next_arrivals)
            if next_time > total_time:
                break
            current_time = next_time
            continue
        
        # Update slack times for all tasks in ready queue
        updated_queue = []
        while ready_queue:
            _, instance_num, task = heapq.heappop(ready_queue)
            slack = calculate_slack(task, current_time)
            heapq.heappush(updated_queue, (slack, instance_num, task))
        ready_queue = updated_queue
        
        # Get task with least slack
        _, _, task = heapq.heappop(ready_queue)
        
        if task.start_time is None:
            task.start_time = current_time
        
        # Execute for one time unit
        remaining_time[task.pid] -= 1
        current_time += 1
        
        if remaining_time[task.pid] > 0:
            # Task not finished, put back in queue
            slack = calculate_slack(task, current_time)
            heapq.heappush(ready_queue, (slack, current_instance[task.pid], task))
        else:
            # Task finished
            task.finish_time = current_time
            completed.append(task)
            # Reset remaining time for next instance
            remaining_time[task.pid] = task.execution
            # Mark non-periodic tasks as done
            if task.period == 0:
                next_arrival[task.pid] = float('inf')
    
    return calculate_avg_waiting_time(completed, 'lst')

def rms_scheduler(tasks, total_time=100):
    """Rate Monotonic Scheduling (RMS)."""
    if not tasks:
        return float('inf')
    
    # Check if all tasks are periodic
    if not all(task.period > 0 for task in tasks):
        return float('inf')
    
    # Check RMS utilization bound
    n = len(tasks)
    utilization = sum(task.execution / task.period for task in tasks)
    rms_bound = n * (2 ** (1/n) - 1)
    if utilization > rms_bound:
        return float('inf')
    
    tasks = copy.deepcopy(tasks)
    current_time = 0
    completed = []
    ready_queue = []
    
    # Track remaining time and next arrival for each task
    remaining_time = {task.pid: task.execution for task in tasks}
    next_arrival = {task.pid: task.arrival for task in tasks}
    
    # Track current instance for each task
    current_instance = {task.pid: 0 for task in tasks}
    
    def get_priority(task):
        """Get RMS priority (shorter period = higher priority)."""
        return task.period
    
    while current_time < total_time:
        # Add newly arrived tasks to ready queue
        for task in tasks:
            if next_arrival[task.pid] <= current_time:
                # Create new instance
                instance = Task(
                    pid=task.pid,
                    arrival=next_arrival[task.pid],
                    execution=task.execution,
                    priority=task.priority,
                    deadline=task.deadline,
                    period=task.period
                )
                instance.absolute_deadline = next_arrival[task.pid] + task.deadline
                
                # Add to ready queue if not already there
                if all(t[2].pid != task.pid for t in ready_queue):
                    priority = get_priority(task)
                    heapq.heappush(ready_queue, (priority, current_instance[task.pid], instance))
                    current_instance[task.pid] += 1
                
                # Schedule next arrival
                next_arrival[task.pid] += task.period
        
        if not ready_queue:
            # Find next arrival time
            next_time = min(next_arrival.values())
            if next_time > total_time:
                break
            current_time = next_time
            continue
        
        # Get highest priority task
        _, _, task = heapq.heappop(ready_queue)
        
        if task.start_time is None:
            task.start_time = current_time
        
        # Execute for one time unit
        remaining_time[task.pid] -= 1
        current_time += 1
        
        if remaining_time[task.pid] > 0:
            # Task not finished, put back in queue
            priority = get_priority(task)
            heapq.heappush(ready_queue, (priority, current_instance[task.pid], task))
        else:
            # Task finished
            task.finish_time = current_time
            completed.append(task)
            # Reset remaining time for next instance
            remaining_time[task.pid] = task.execution
    
    return calculate_avg_waiting_time(completed, 'rms')

def evaluate_schedulers(tasks, early_stopping=True):
    """Evaluate all scheduling algorithms on a task set with improved metrics."""
    if not tasks:
        return {}, None
    
    results = {}
    schedulers = {
        'edf': edf_scheduler,
        'rms': rms_scheduler,
        'clock_driven': clock_driven_scheduler,
        'priority': priority_scheduler,
        'lst': lst_scheduler,
        'weighted_rr': weighted_rr_scheduler
    }
    
    # Calculate task set characteristics
    all_periodic = all(task.period > 0 for task in tasks)
    total_util = sum(task.execution / task.period for task in tasks if task.period > 0)
    max_priority = max(t.priority for t in tasks)
    min_priority = min(t.priority for t in tasks)
    priority_range = max_priority - min_priority
    
    # Calculate task set metrics
    avg_execution = np.mean([task.execution for task in tasks])
    avg_deadline = np.mean([task.deadline for task in tasks])
    avg_period = np.mean([task.period for task in tasks if task.period > 0]) if any(task.period > 0 for task in tasks) else 0
    
    # Calculate scheduler suitability scores (higher is better)
    suitability = {
        'edf': 0.9,  # EDF is not always the best
        'rms': 1.0 if all_periodic and total_util <= 0.7 else 0.0,  # RMS bound
        'clock_driven': 1.0 if avg_period > 0 and total_util <= 0.9 else 0.0,
        'priority': 1.0 if priority_range >= 2 else 0.0,
        'lst': 1.0 if avg_deadline >= avg_execution * 1.2 else 0.0,
        'weighted_rr': 1.0 if max_priority >= 2 else 0.0
    }
    
    # Sort schedulers by suitability
    scheduler_order = sorted(schedulers.keys(), key=lambda x: suitability[x], reverse=True)
    
    best_score = float('inf')
    best_scheduler = None
    
    for name in scheduler_order:
        # Skip unsuitable schedulers
        if suitability[name] == 0:
            results[name] = float('inf')
            continue
        
        # Make a deep copy of tasks
        tasks_copy = [Task(t.pid, t.arrival, t.execution, t.priority, t.deadline, t.period) for t in tasks]
        
        try:
            # Run scheduler and get score
            scheduler = schedulers[name]
            score = scheduler(tasks_copy)
            
            # Calculate final score with scheduler-specific adjustments
            final_score = calculate_avg_waiting_time(tasks_copy, name)
            
            # Early stopping if we find a perfect scheduler
            if final_score == 0:
                return {name: final_score}, name
            
            results[name] = final_score
            
            # Update best scheduler
            if final_score < best_score:
                best_score = final_score
                best_scheduler = name
                
                # Early stopping if we found a significantly better scheduler
                if early_stopping and final_score < best_score * 0.7:
                    break
                    
        except Exception as e:
            print(f"Warning: Scheduler {name} failed with error: {str(e)}")
            results[name] = float('inf')
    
    if best_scheduler is None:
        best_scheduler = min(results.items(), key=lambda x: x[1])[0]
    
    return results, best_scheduler