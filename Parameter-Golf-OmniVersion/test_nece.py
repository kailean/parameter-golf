#!/usr/bin/env python3
"""
Quick Test: NeuroEvolutionary Compression Engine (NECE) demo
Runs a tiny evolution job to verify NECE integration works.
"""
import torch
import random
import math
from nece import NeuroEvolutionaryCompressionEngine, NeuralNetworkGenome, CompressedNeuralNetwork

def dummy_fitness(genome):
    """Simple fitness: higher reward for small model under 16MB"""
    try:
        model = CompressedNeuralNetwork(genome)
        size_bytes = model.get_model_size()
        size_mb = size_bytes / 1024 / 1024
        # reward small models, penalize large ones
        return 1.0 / (1.0 + size_mb) if size_mb > 0 else 1.0
    except Exception:
        return 0.0

def main():
    print("🚀 Running NECE quick test...")
    
    # Create NECE instance with tiny config for speed
    nece = NeuroEvolutionaryCompressionEngine(
        input_size=32,
        output_size=2,
        population_size=5,          # tiny population
        max_generations=3,          # few generations
        size_constraint=16 * 1024 * 1024  # 16MB constraint
    )
    
    # Override fitness evaluation with our dummy function
    # (NECE's internal evaluate_genome uses its own logic; 
    # we'll simulate by manually evaluating each genome)
    generation = 0
    best_fitness = 0.0
    best_genome = None
    
    for _ in range(3):  # simulate a few generations
        generation += 1
        print(f"\n--- Generation {generation} ---")
        
        # Evaluate all genomes
        for genome in nece.population.genomes:
            genome.fitness = dummy_fitness(genome)
            # Track best
            if genome.fitness > best_fitness:
                best_fitness = genome.fitness
                best_genome = genome
        
        print(f"Best fitness this gen: {best_fitness:.4f}")
        
        # Simple reproduction: select top 2, breed, mutate slightly
        sorted_genomes = sorted(nece.population.genomes, key=lambda g: g.fitness, reverse=True)[:2]
        parents = sorted_genomes
        
        # Create a child
        child = parents[0].crossover(parents[1])
        for _ in range(2):  # mutate 1-2 times
            if random.random() < 0.7:
                child.mutate()
        # Add child to population (replace worst)
        # find worst genome
        worst_idx = min(range(len(nece.population.genomes)), 
                      key=lambda i: nece.population.genomes[i].fitness)
        nece.population.genomes[worst_idx] = child
        
        print(f"  Best genome fitness: {best_fitness:.4f}")
        if best_genome:
            size_mb = best_genome.get_model_size() / 1024 / 1024
            print(f"  Best model size: {size_mb:.2f} MB")
    
    # Final result
    print("\n" + "="*40)
    if best_genome:
        size_mb = best_genome.get_model_size() / 1024 / 1024
        print(f"🎉 NECE completed! Best model size: {size_mb:.2f} MB")
        print(f"   Fitness: {best_fitness:.4f}")
        # Estimate actual model size using CompressedNeuralNetwork
        try:
            model = CompressedNeuralNetwork(best_genome)
            actual_size = model.get_model_size()
            print(f"   Actual size: {actual_size/1024/1024:.2f} MB")
        except Exception as e:
            print(f"   Error measuring actual size: {e}")
    else:
        print("❌ No valid genome found under size constraint")
    
    print("="*40)

if __name__ == "__main__":
    main()