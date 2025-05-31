import random

def calculate_avg_waiting_time(processes, order):
    time = 0
    waiting_times = []
    process_dict = {p['pid']: p for p in processes}
    
    for pid in order:
        p = process_dict[pid]
        arrival = p['arrival']
        burst = p['burst']
        
        if time < arrival:
            time = arrival
        
        waiting_time = time - arrival
        waiting_times.append(waiting_time)
        time += burst
    
    return sum(waiting_times) / len(waiting_times)

class Particle:
    def __init__(self, processes):
        self.processes = processes
        self.position = random.sample([p['pid'] for p in processes], len(processes))
        self.best_position = self.position[:]
        self.best_fitness = float('inf')
        self.velocity = []
    
    def fitness(self):
        return calculate_avg_waiting_time(self.processes, self.position)

    def update_velocity(self, global_best_position, w=0.5, c1=1, c2=1):
        swaps = []
        for i in range(len(self.position)):
            if self.position[i] != self.best_position[i]:
                swaps.append((i, self.position.index(self.best_position[i])))
            if self.position[i] != global_best_position[i]:
                swaps.append((i, self.position.index(global_best_position[i])))
        self.velocity = random.sample(swaps, min(len(swaps), 2))
    
    def apply_velocity(self):
        for i, j in self.velocity:
            self.position[i], self.position[j] = self.position[j], self.position[i]

def pso_scheduler(tasks, num_particles=30, iterations=100):
    # Fix attribute names here:
    processes = [{
        'pid': t.pid,
        'arrival': t.arrival,         # Changed from t.arrival_time
        'burst': t.execution,         # Changed from t.burst_time
        'priority': t.priority,
        'deadline': t.deadline,
        'period': t.period
    } for t in tasks]

    swarm = [Particle(processes) for _ in range(num_particles)]
    global_best_position = swarm[0].position[:]
    global_best_fitness = float('inf')

    for _ in range(iterations):
        for particle in swarm:
            fitness = particle.fitness()
            if fitness < particle.best_fitness:
                particle.best_fitness = fitness
                particle.best_position = particle.position[:]
            if fitness < global_best_fitness:
                global_best_fitness = fitness
                global_best_position = particle.position[:]

        for particle in swarm:
            particle.update_velocity(global_best_position)
            particle.apply_velocity()

    return global_best_fitness
