import numpy as np
import pandas as pd
from typing import List, Tuple
from scheduling_algorithms import Process, TraditionalScheduler, RealtimeScheduler

class ProcessGenerator:
    def __init__(self, num_samples: int = 500):  
        self.num_samples = num_samples
        self.traditional_scheduler = TraditionalScheduler()
        self.realtime_scheduler = RealtimeScheduler()
    
    def _generate_traditional_sample(self) -> dict:
        num_processes = np.random.randint(3, 8)
        processes = []
        
        for j in range(num_processes):
            processes.append(Process(
                pid=j,
                burst_time=np.random.randint(5, 20),
                arrival_time=np.random.randint(0, 12), 
                priority=np.random.randint(1, 5)
            ))
        time_quantum = np.random.randint(3, 7)  
        
        fcfs_wt = self.traditional_scheduler.fcfs(processes)
        sjf_wt = self.traditional_scheduler.sjf(processes)
        rr_wt = self.traditional_scheduler.round_robin(processes, time_quantum)
        priority_wt = self.traditional_scheduler.priority(processes)
        
        waiting_times = [fcfs_wt, sjf_wt, rr_wt, priority_wt]
        best_algo = ['FCFS', 'SJF', 'RR', 'Priority'][np.argmin(waiting_times)]
        p = processes[0]
        return {
            'burst_time': p.burst_time,
            'arrival_time': p.arrival_time,
            'priority': p.priority,
            'num_processes': num_processes,
            'time_quantum': time_quantum,
            'fcfs_wt': fcfs_wt,
            'sjf_wt': sjf_wt,
            'rr_wt': rr_wt,
            'priority_wt': priority_wt,
            'best_algorithm': best_algo
        }
    
    def _generate_realtime_sample(self) -> dict:

        num_processes = np.random.randint(3, 8)  # Reduced max processes
        processes = []
        
        for j in range(num_processes):

            burst_time = np.random.randint(1, 12) 
            arrival_time = np.random.randint(0, 6)  
            priority = np.random.randint(1, 5)
            
    
            deadline = burst_time + np.random.randint(2, 8)  
            period = deadline + np.random.randint(4, 12) 
            
        
            if np.random.random() < 0.4: 
                deadline = burst_time + np.random.randint(1, 4)  
                period = deadline + np.random.randint(2, 6) 
            
            processes.append(Process(
                pid=j,
                burst_time=burst_time,
                arrival_time=arrival_time,
                priority=priority,
                deadline=deadline,
                period=period
            ))
        
        # Calculate waiting times for real-time algorithms
        cd_wt = self.realtime_scheduler.clock_driven(processes)
        edf_wt = self.realtime_scheduler.edf(processes)
        wrr_wt = self.realtime_scheduler.weighted_round_robin(processes)
        pd_wt = self.realtime_scheduler.priority_driven(processes)
        lstf_wt = self.realtime_scheduler.lstf(processes)
        rm_wt = self.realtime_scheduler.rate_monotonic(processes)
        
        
        waiting_times = [
            cd_wt * 0.85,   
            edf_wt * 0.8,    
            wrr_wt * 0.9,    
            pd_wt * 0.8,     
            lstf_wt * 0.75,  
            rm_wt * 0.7      
        ]
        
        algorithms = ['Clock-Driven', 'EDF', 'Weighted-RR', 'Priority-Driven', 'LSTF', 'Rate-Monotonic']
        best_algo = algorithms[np.argmin(waiting_times)]
        

        p = processes[0]
        return {
            'burst_time': p.burst_time,
            'arrival_time': p.arrival_time,
            'priority': p.priority,
            'deadline': p.deadline,
            'period': p.period,
            'clock_driven_wt': cd_wt,
            'edf_wt': edf_wt,
            'weighted_rr_wt': wrr_wt,
            'priority_driven_wt': pd_wt,
            'lstf_wt': lstf_wt,
            'rate_monotonic_wt': rm_wt,
            'best_algorithm': best_algo
        }
    
    def generate_traditional_dataset(self) -> pd.DataFrame:
        data = []
        for i in range(self.num_samples):
            try:
                sample = self._generate_traditional_sample()
                data.append(sample)
            except Exception as e:
                print(f"Error generating traditional sample {i}: {str(e)}")
                continue
        return pd.DataFrame(data)
    
    def generate_realtime_dataset(self) -> pd.DataFrame:
        data = []
        for i in range(self.num_samples):
            try:
                sample = self._generate_realtime_sample()
                data.append(sample)
            except Exception as e:
                print(f"Error generating realtime sample {i}: {str(e)}")
                continue
        return pd.DataFrame(data) 