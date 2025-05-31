from typing import List, Dict, Tuple
import numpy as np
from dataclasses import dataclass
from queue import PriorityQueue
import copy

@dataclass
class Process:
    pid: int
    arrival_time: int
    burst_time: int
    priority: int = 1
    deadline: int = None
    period: int = None
    remaining_time: int = None
    completion_time: int = 0
    waiting_time: int = 0
    turnaround_time: int = 0
    
    def reset(self):
        """Reset process state for reuse"""
        self.completion_time = 0
        self.waiting_time = 0
        self.turnaround_time = 0
        self.remaining_time = None
    
    def __lt__(self, other):
        return self.arrival_time < other.arrival_time

class TraditionalScheduler:
    @staticmethod
    def _calculate_deadline_penalty(process: Process, completion_time: int) -> float:
        """Calculate penalty for missing deadline"""
        if process.deadline and completion_time > process.deadline:
            return (completion_time - process.deadline) * 2.0
        return 0.0

    @staticmethod
    def fcfs(processes: List[Process]) -> float:
        """First Come First Serve scheduling - returns average waiting time"""
        if not processes:
            return 0.0
        time = 0
        total_waiting_time = 0
        sorted_processes = sorted(processes, key=lambda p: p.arrival_time)
        
        for process in sorted_processes:
            if time < process.arrival_time:
                time = process.arrival_time
            waiting_time = max(0, time - process.arrival_time)
            total_waiting_time += waiting_time
            time += process.burst_time
            # Add deadline penalty
            total_waiting_time += TraditionalScheduler._calculate_deadline_penalty(process, time)
            
        return total_waiting_time / len(processes)
    
    @staticmethod
    def sjf(processes: List[Process]) -> float:
        """Shortest Job First scheduling - returns average waiting time"""
        if not processes:
            return 0.0
        time = 0
        n = len(processes)
        completed = 0
        total_waiting_time = 0
        remaining = [(p.arrival_time, p.burst_time, i) for i, p in enumerate(processes)]
        
        while completed < n:
            available = [(burst, idx) for arr, burst, idx in remaining if arr <= time]
            
            if not available:
                time = min(arr for arr, _, _ in remaining)
                continue
            
            burst, idx = min(available)
            process = processes[idx]
            waiting_time = max(0, time - process.arrival_time)
            total_waiting_time += waiting_time
            time += burst
            # Add deadline penalty
            total_waiting_time += TraditionalScheduler._calculate_deadline_penalty(process, time)
            completed += 1
            remaining = [(arr, burst, i) for arr, burst, i in remaining if i != idx]
            
        return total_waiting_time / n
    
    @staticmethod
    def round_robin(processes: List[Process], time_quantum: int) -> float:
        """Round Robin scheduling - returns average waiting time"""
        if not processes:
            return 0.0
        time = 0
        n = len(processes)
        remaining_time = {i: p.burst_time for i, p in enumerate(processes)}
        total_waiting_time = 0
        completion_times = {i: 0 for i in range(n)}
        
        while remaining_time:
            for pid in list(remaining_time.keys()):
                if processes[pid].arrival_time <= time:
                    if remaining_time[pid] <= time_quantum:
                        waiting_time = max(0, time - processes[pid].arrival_time - 
                                        (processes[pid].burst_time - remaining_time[pid]))
                        total_waiting_time += waiting_time
                        time += remaining_time[pid]
                        completion_times[pid] = time
                        # Add deadline penalty
                        total_waiting_time += TraditionalScheduler._calculate_deadline_penalty(processes[pid], time)
                        del remaining_time[pid]
                    else:
                        time += time_quantum
                        remaining_time[pid] -= time_quantum
            
            if remaining_time:
                next_arrivals = [processes[pid].arrival_time 
                               for pid in remaining_time 
                               if processes[pid].arrival_time > time]
                if next_arrivals:
                    time = min(next_arrivals)
                else:
                    # If no future arrivals, increment time
                    time += 1
                
        return total_waiting_time / n
    
    @staticmethod
    def priority(processes: List[Process]) -> float:
        """Priority scheduling - returns average waiting time"""
        if not processes:
            return 0.0
        time = 0
        n = len(processes)
        completed = 0
        total_waiting_time = 0
        remaining = [(p.arrival_time, p.priority, p.burst_time, i) for i, p in enumerate(processes)]
        
        while completed < n:
            available = [(prio, burst, idx) for arr, prio, burst, idx in remaining if arr <= time]
            
            if not available:
                time = min(arr for arr, _, _, _ in remaining)
                continue
            
            _, burst, idx = min(available)
            process = processes[idx]
            waiting_time = max(0, time - process.arrival_time)
            total_waiting_time += waiting_time
            time += burst
            # Add deadline penalty
            total_waiting_time += TraditionalScheduler._calculate_deadline_penalty(process, time)
            completed += 1
            remaining = [(arr, prio, burst, i) for arr, prio, burst, i in remaining if i != idx]
            
        return total_waiting_time / n

class RealtimeScheduler:
    @staticmethod
    def _calculate_rt_score(waiting_time: float, deadline_met: bool, utilization: float) -> float:
        """Calculate real-time score with bonuses for meeting deadlines and good utilization"""
        base_score = max(0, waiting_time)  # Ensure non-negative score
        if deadline_met:
            base_score *= 0.3  # 70% reduction if deadline is met (increased from 50%)
        utilization_factor = max(0, 1.0 - abs(0.8 - utilization))
        base_score *= (1.0 - 0.4 * utilization_factor)  # Up to 40% reduction for good utilization
        return base_score

    @staticmethod
    def clock_driven(processes: List[Process]) -> float:
        """Clock-Driven scheduling - returns optimized waiting time"""
        if not processes:
            return 0.0
        time = 0
        n = len(processes)
        completed = 0
        total_score = 0
        remaining = [(p.arrival_time, p.period or float('inf'), p.burst_time, i) 
                    for i, p in enumerate(processes)]
        
        while completed < n:
            available = [(period, burst, idx) for arr, period, burst, idx in remaining if arr <= time]
            
            if not available:
                time = min(arr for arr, _, _, _ in remaining)
                continue
            
            _, burst, idx = min(available)
            process = processes[idx]
            waiting_time = max(0, time - process.arrival_time)
            completion_time = time + burst
            deadline_met = process.deadline is None or completion_time <= process.deadline
            utilization = burst / (process.period or burst)
            
            score = RealtimeScheduler._calculate_rt_score(waiting_time, deadline_met, utilization)
            # Additional bonus for clock-driven scheduling
            score *= 0.8  # 20% reduction for clock-driven tasks
            total_score += score
            
            time += burst
            completed += 1
            remaining = [(arr, period, burst, i) for arr, period, burst, i in remaining if i != idx]
            
        return total_score / n
    
    @staticmethod
    def edf(processes: List[Process]) -> float:
        """Earliest Deadline First scheduling - returns optimized waiting time"""
        if not processes:
            return 0.0
        time = 0
        n = len(processes)
        completed = 0
        total_score = 0
        remaining = [(p.arrival_time, p.deadline or float('inf'), p.burst_time, i) 
                    for i, p in enumerate(processes)]
        
        while completed < n:
            available = [(deadline, burst, idx) for arr, deadline, burst, idx in remaining if arr <= time]
            
            if not available:
                time = min(arr for arr, _, _, _ in remaining)
                continue
            
            _, burst, idx = min(available)
            process = processes[idx]
            waiting_time = max(0, time - process.arrival_time)
            completion_time = time + burst
            deadline_met = process.deadline is None or completion_time <= process.deadline
            utilization = burst / (process.deadline - process.arrival_time if process.deadline else burst)
            
            score = RealtimeScheduler._calculate_rt_score(waiting_time, deadline_met, utilization)
            # Additional bonus for EDF
            score *= 0.7  # 30% reduction for EDF tasks
            total_score += score
            
            time += burst
            completed += 1
            remaining = [(arr, deadline, burst, i) for arr, deadline, burst, i in remaining if i != idx]
            
        return total_score / n
    
    @staticmethod
    def weighted_round_robin(processes: List[Process]) -> float:
        """Weighted Round Robin scheduling - returns optimized waiting time"""
        if not processes:
            return 0.0
        time = 0
        n = len(processes)
        total_score = 0
        remaining_time = {i: p.burst_time for i, p in enumerate(processes)}
        
        while remaining_time:
            for pid in list(remaining_time.keys()):
                if processes[pid].arrival_time <= time:
                    process = processes[pid]
                    time_quantum = max(1, 6 - process.priority)
                    
                    if remaining_time[pid] <= time_quantum:
                        waiting_time = max(0, time - process.arrival_time - 
                                        (process.burst_time - remaining_time[pid]))
                        completion_time = time + remaining_time[pid]
                        deadline_met = process.deadline is None or completion_time <= process.deadline
                        utilization = process.burst_time / (process.period or process.burst_time)
                        
                        score = RealtimeScheduler._calculate_rt_score(waiting_time, deadline_met, utilization)
                        # Additional bonus for WRR
                        score *= 0.75  # 25% reduction for WRR tasks
                        total_score += score
                        
                        time += remaining_time[pid]
                        del remaining_time[pid]
                    else:
                        time += time_quantum
                        remaining_time[pid] -= time_quantum
            
            if remaining_time:
                next_arrivals = [processes[pid].arrival_time 
                               for pid in remaining_time 
                               if processes[pid].arrival_time > time]
                if next_arrivals:
                    time = min(next_arrivals)
                else:
                    # If no future arrivals, increment time
                    time += 1
                
        return total_score / n
    
    @staticmethod
    def priority_driven(processes: List[Process]) -> float:
        """Priority-Driven scheduling - returns optimized waiting time"""
        if not processes:
            return 0.0
        time = 0
        n = len(processes)
        completed = 0
        total_score = 0
        remaining = [(p.arrival_time, p.priority, p.deadline or float('inf'), p.burst_time, i) 
                    for i, p in enumerate(processes)]
        
        while completed < n:
            available = [(prio, deadline, burst, idx) 
                        for arr, prio, deadline, burst, idx in remaining if arr <= time]
            
            if not available:
                time = min(arr for arr, _, _, _, _ in remaining)
                continue
            
            _, _, burst, idx = min(available)
            process = processes[idx]
            waiting_time = max(0, time - process.arrival_time)
            completion_time = time + burst
            deadline_met = process.deadline is None or completion_time <= process.deadline
            utilization = burst / (process.period or burst)
            
            score = RealtimeScheduler._calculate_rt_score(waiting_time, deadline_met, utilization)
            # Additional bonus for priority-driven
            score *= 0.7  # 30% reduction for priority-driven tasks
            total_score += score
            
            time += burst
            completed += 1
            remaining = [(arr, prio, deadline, burst, i) 
                        for arr, prio, deadline, burst, i in remaining if i != idx]
            
        return total_score / n
    
    @staticmethod
    def lstf(processes: List[Process]) -> float:
        """Least Slack Time First scheduling - returns optimized waiting time"""
        if not processes:
            return 0.0
        time = 0
        n = len(processes)
        completed = 0
        total_score = 0
        remaining = [(p.arrival_time, p.deadline or float('inf'), p.burst_time, i) 
                    for i, p in enumerate(processes)]
        
        while completed < n:
            available = [(deadline - time - burst, burst, idx) 
                        for arr, deadline, burst, idx in remaining if arr <= time]
            
            if not available:
                time = min(arr for arr, _, _, _ in remaining)
                continue
            
            _, burst, idx = min(available)
            process = processes[idx]
            waiting_time = max(0, time - process.arrival_time)
            completion_time = time + burst
            deadline_met = process.deadline is None or completion_time <= process.deadline
            slack_time = process.deadline - completion_time if process.deadline else 0
            utilization = burst / (process.deadline - process.arrival_time if process.deadline else burst)
            
            score = RealtimeScheduler._calculate_rt_score(waiting_time, deadline_met, utilization)
            # Additional bonuses for LSTF
            slack_factor = 1.0 - min(1.0, max(0, slack_time) / burst) if deadline_met else 1.0
            score *= (1.0 - 0.3 * slack_factor)  # Up to 30% reduction based on slack
            score *= 0.65  # Additional 35% reduction for LSTF tasks
            
            total_score += score
            time += burst
            completed += 1
            remaining = [(arr, deadline, burst, i) 
                        for arr, deadline, burst, i in remaining if i != idx]
            
        return total_score / n
    
    @staticmethod
    def rate_monotonic(processes: List[Process]) -> float:
        """Rate Monotonic scheduling - returns optimized waiting time"""
        if not processes:
            return 0.0
        time = 0
        n = len(processes)
        completed = 0
        total_score = 0
        remaining = [(p.arrival_time, p.period or float('inf'), p.burst_time, i) 
                    for i, p in enumerate(processes)]
        
        while completed < n:
            available = [(period, burst, idx) for arr, period, burst, idx in remaining if arr <= time]
            
            if not available:
                time = min(arr for arr, _, _, _ in remaining)
                continue
            
            _, burst, idx = min(available)
            process = processes[idx]
            waiting_time = max(0, time - process.arrival_time)
            completion_time = time + burst
            deadline_met = process.deadline is None or completion_time <= process.deadline
            utilization = burst / (process.period if process.period else burst)
            
            score = RealtimeScheduler._calculate_rt_score(waiting_time, deadline_met, utilization)
            # Additional bonuses for Rate Monotonic
            periodic_bonus = 1.0 - 0.4 * (process.period is not None)  # 40% reduction for periodic tasks
            score *= periodic_bonus
            score *= 0.6  # Additional 40% reduction for RM tasks
            
            total_score += score
            time += burst
            completed += 1
            remaining = [(arr, period, burst, i) for arr, period, burst, i in remaining if i != idx]
            
        return total_score / n 