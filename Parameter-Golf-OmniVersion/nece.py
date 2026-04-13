"""
NeuroEvolutionary Compression Engine (NECE)
Evolves neural network architectures and compresses them under 16MB
Using PyTorch and evolutionary algorithms like NEAT
"""

import torch
import torch.nn as nn
import random
import numpy as np
from typing import List, Tuple, Optional
import math

class NeuralNetworkGenome:
    """Represents a neural network genome with variable architecture"""
    
    def __init__(self, input_size: int = 784, output_size: int = 10):
        self.input_size = input_size
        self.output_size = output_size
        self.layers: List[Tuple[int, int]] = []  # [(in_features, out_features)]
        self.activations: List[str] = []  # activation function names
        self.fitness: float = 0.0
        self.age: int = 0
        
        # Initialize with a simple network
        self.layers.append((input_size, 64))
        self.activations.append("relu")
        self.layers.append((64, 32))
        self.activations.append("relu")
        self.layers.append((32, output_size))
        self.activations.append("softmax")
        
    def mutate(self):
        """Apply mutations to the genome"""
        mutation_type = random.choice(['add_layer', 'remove_layer', 'modify_layer', 'change_activation'])
        
        if mutation_type == 'add_layer' and len(self.layers) < 10:
            # Add a new layer
            idx = random.randint(0, len(self.layers))
            if idx == 0:
                in_features = self.input_size
            else:
                in_features = self.layers[idx-1][1]
                
            if idx == len(self.layers):
                out_features = self.output_size
            else:
                out_features = self.layers[idx][0]
                
            # Insert new layer
            mid_features = random.randint(8, max(in_features, out_features))
            self.layers.insert(idx, (in_features, mid_features))
            self.activations.insert(idx, random.choice(["relu", "tanh", "sigmoid"]))
            
            # Adjust the next layer
            if idx < len(self.layers) - 1:
                self.layers[idx+1] = (mid_features, self.layers[idx+1][1])
                
        elif mutation_type == 'remove_layer' and len(self.layers) > 2:
            # Remove a layer
            idx = random.randint(0, len(self.layers) - 1)
            if idx < len(self.layers) - 1:
                # Adjust previous layer's output
                if idx > 0:
                    self.layers[idx-1] = (self.layers[idx-1][0], self.layers[idx+1][0])
                else:
                    # First layer connects to the layer after the removed one
                    pass
            self.layers.pop(idx)
            self.activations.pop(idx)
            
        elif mutation_type == 'modify_layer':
            # Modify layer dimensions
            if len(self.layers) > 0:
                idx = random.randint(0, len(self.layers) - 1)
                in_features, out_features = self.layers[idx]
                # Change by up to 50%
                change_factor = random.uniform(0.5, 1.5)
                new_out = max(8, int(out_features * change_factor))
                self.layers[idx] = (in_features, new_out)
                
                # Adjust next layer if it exists
                if idx < len(self.layers) - 1:
                    next_in, next_out = self.layers[idx+1]
                    self.layers[idx+1] = (new_out, next_out)
                    
        elif mutation_type == 'change_activation':
            # Change activation function
            if len(self.activations) > 0:
                idx = random.randint(0, len(self.activations) - 1)
                self.activations[idx] = random.choice(["relu", "tanh", "sigmoid", "leaky_relu"])
                
        self.age += 1
                
    def crossover(self, other: 'NeuralNetworkGenome') -> 'NeuralNetworkGenome':
        """Create offspring from two genomes"""
        child = NeuralNetworkGenome(self.input_size, self.output_size)
        
        # Take layers from both parents
        max_layers = max(len(self.layers), len(other.layers))
        child.layers = []
        child.activations = []
        
        for i in range(max_layers):
            if i < len(self.layers) and i < len(other.layers):
                # Choose randomly from either parent
                if random.random() < 0.5:
                    child.layers.append(self.layers[i])
                    child.activations.append(self.activations[i])
                else:
                    child.layers.append(other.layers[i])
                    child.activations.append(other.activations[i])
            elif i < len(self.layers):
                child.layers.append(self.layers[i])
                child.activations.append(self.activations[i])
            else:
                child.layers.append(other.layers[i])
                child.activations.append(other.activations[i])
                
        return child

class CompressedNeuralNetwork(nn.Module):
    """PyTorch neural network with compression capabilities"""
    
    def __init__(self, genome: NeuralNetworkGenome):
        super(CompressedNeuralNetwork, self).__init__()
        self.genome = genome
        
        # Build the network from genome
        layers = []
        for i, (in_features, out_features) in enumerate(genome.layers):
            layers.append(nn.Linear(in_features, out_features))
            if i < len(genome.activations):
                if genome.activations[i] == "relu":
                    layers.append(nn.ReLU())
                elif genome.activations[i] == "tanh":
                    layers.append(nn.Tanh())
                elif genome.activations[i] == "sigmoid":
                    layers.append(nn.Sigmoid())
                elif genome.activations[i] == "leaky_relu":
                    layers.append(nn.LeakyReLU())
                elif genome.activations[i] == "softmax":
                    layers.append(nn.Softmax(dim=1))
                    
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)
        
    def get_model_size(self) -> int:
        """Return model size in bytes"""
        param_size = 0
        for param in self.parameters():
            param_size += param.nelement() * param.element_size()
            
        buffer_size = 0
        for buffer in self.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
            
        return param_size + buffer_size

class Population:
    """Manages a population of neural network genomes"""
    
    def __init__(self, size: int, input_size: int = 784, output_size: int = 10):
        self.size = size
        self.genomes: List[NeuralNetworkGenome] = []
        self.input_size = input_size
        self.output_size = output_size
        
        # Initialize population
        for _ in range(size):
            genome = NeuralNetworkGenome(input_size, output_size)
            # Apply some initial mutations
            for _ in range(random.randint(0, 3)):
                genome.mutate()
            self.genomes.append(genome)
            
    def select_parents(self) -> Tuple[NeuralNetworkGenome, NeuralNetworkGenome]:
        """Select two parents based on fitness (tournament selection)"""
        tournament_size = 3
        participants1 = random.sample(self.genomes, min(tournament_size, len(self.genomes)))
        participants2 = random.sample(self.genomes, min(tournament_size, len(self.genomes)))
        
        parent1 = max(participants1, key=lambda g: g.fitness)
        parent2 = max(participants2, key=lambda g: g.fitness)
        
        return parent1, parent2
        
    def evolve(self):
        """Create next generation through selection, crossover and mutation"""
        # Sort by fitness
        self.genomes.sort(key=lambda g: g.fitness, reverse=True)
        
        # Keep top 20% as elite
        elite_count = self.size // 5
        new_genomes = self.genomes[:elite_count]
        
        # Generate offspring for the rest
        while len(new_genomes) < self.size:
            parent1, parent2 = self.select_parents()
            child = parent1.crossover(parent2)
            
            # Mutate the child
            if random.random() < 0.8:  # 80% chance of mutation
                for _ in range(random.randint(1, 3)):
                    child.mutate()
                    
            new_genomes.append(child)
            
        self.genomes = new_genomes

class NeuroEvolutionaryCompressionEngine:
    """Main NECE class that evolves and compresses neural networks"""
    
    def __init__(self, 
                 input_size: int = 784, 
                 output_size: int = 10,
                 population_size: int = 20,
                 max_generations: int = 50,
                 size_constraint: int = 16 * 1024 * 1024):  # 16MB
        self.input_size = input_size
        self.output_size = output_size
        self.population_size = population_size
        self.max_generations = max_generations
        self.size_constraint = size_constraint  # 16MB in bytes
        
        self.population = Population(population_size, input_size, output_size)
        self.best_genome: Optional[NeuralNetworkGenome] = None
        self.best_fitness: float = 0.0
        
    def evaluate_genome(self, genome: NeuralNetworkGenome) -> float:
        """
        Evaluate a genome's fitness based on accuracy and size
        Returns a fitness score (higher is better)
        """
        try:
            model = CompressedNeuralNetwork(genome)
            model_size = model.get_model_size()
            
            # Size penalty - models over 16MB get heavy penalty
            size_penalty = 0
            if model_size > self.size_constraint:
                size_penalty = (model_size - self.size_constraint) / self.size_constraint
            
            # Simulate accuracy (in real implementation, you would train the model)
            # For now, we'll base accuracy on architectural efficiency
            # Fewer parameters generally mean better compression but potentially lower accuracy
            param_count = sum(p.nelement() for p in model.parameters())
            
            # Simple heuristic: balance between parameter count and depth
            depth_score = 1.0 / (1.0 + len(genome.layers) * 0.1)
            param_score = 1.0 / (1.0 + (param_count / 10000.0))
            
            # Combine scores (in a real implementation, you'd use actual training accuracy)
            accuracy_estimate = (depth_score + param_score) / 2.0
            
            # Fitness is accuracy minus size penalty
            fitness = accuracy_estimate - size_penalty
            
            return max(0.0, fitness)  # Ensure non-negative fitness
        except Exception as e:
            # Return very low fitness for invalid architectures
            return 0.0
    
    def compress_model(self, model: nn.Module) -> nn.Module:
        """
        Apply compression techniques to an existing model
        """
        # In a full implementation, you would apply:
        # 1. Pruning - remove unimportant weights
        # 2. Quantization - reduce precision of weights
        # 3. Knowledge distillation - train smaller student model
        # 4. Weight sharing - share similar weights
        
        # For this prototype, we'll just return the model as-is
        # A full implementation would modify the model here
        return model
    
    def evolve(self) -> NeuralNetworkGenome:
        """
        Run the neuroevolutionary process
        Returns the best genome found
        """
        print(f"Starting evolution with population size {self.population_size} for {self.max_generations} generations")
        
        for generation in range(self.max_generations):
            # Evaluate all genomes
            for genome in self.population.genomes:
                genome.fitness = self.evaluate_genome(genome)
                
                # Update best genome if this one is better
                if genome.fitness > self.best_fitness:
                    self.best_fitness = genome.fitness
                    self.best_genome = genome
                    
            # Check if we have a valid solution under size constraint
            if self.best_genome:
                best_model = CompressedNeuralNetwork(self.best_genome)
                best_size = best_model.get_model_size()
                
                print(f"Generation {generation+1}: Best fitness = {self.best_fitness:.4f}, Size = {best_size/1024/1024:.2f}MB")
                
                if best_size <= self.size_constraint:
                    print(f"Found solution under size constraint: {best_size/1024/1024:.2f}MB")
                    break
                    
            # Evolve to next generation
            self.population.evolve()
            
        return self.best_genome
    
    def get_best_model(self) -> Optional[nn.Module]:
        """
        Get the best evolved model
        """
        if self.best_genome is None:
            return None
            
        model = CompressedNeuralNetwork(self.best_genome)
        compressed_model = self.compress_model(model)
        return compressed_model

# Example usage
if __name__ == "__main__":
    # Create NECE instance
    nece = NeuroEvolutionaryCompressionEngine(
        input_size=784,      # e.g., flattened 28x28 MNIST images
        output_size=10,      # e.g., 10 classes for MNIST
        population_size=20,
        max_generations=30,
        size_constraint=16 * 1024 * 1024  # 16MB
    )
    
    # Run evolution
    best_genome = nece.evolve()
    
    if best_genome:
        best_model = nece.get_best_model()
        model_size = best_model.get_model_size()
        print(f"\nBest model found:")
        print(f"  Size: {model_size/1024/1024:.2f}MB")
        print(f"  Fitness: {best_genome.fitness:.4f}")
        print(f"  Layers: {len(best_genome.layers)}")
        print(f"  Architecture: {' -> '.join([str(layer[1]) for layer in best_genome.layers])}")
    else:
        print("No valid model found under size constraint")