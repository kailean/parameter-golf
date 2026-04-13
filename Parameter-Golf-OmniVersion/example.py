"""
Example usage of the NeuroEvolutionary Compression Engine (NECE)
"""

from nece import NeuroEvolutionaryCompressionEngine
import torch

def main():
    print("NeuroEvolutionary Compression Engine (NECE) Example")
    print("=" * 50)
    
    # Create NECE instance with sample parameters
    # For MNIST dataset (28x28 images = 784 input features)
    nece = NeuroEvolutionaryCompressionEngine(
        input_size=784,      # Flattened 28x28 image
        output_size=10,      # 10 digit classes
        population_size=15,  # Smaller population for faster demo
        max_generations=20,  # Fewer generations for demo
        size_constraint=16 * 1024 * 1024  # 16MB constraint
    )
    
    print(f"Configured NECE with:")
    print(f"  Input size: {nece.input_size}")
    print(f"  Output size: {nece.output_size}")
    print(f"  Population size: {nece.population_size}")
    print(f"  Max generations: {nece.max_generations}")
    print(f"  Size constraint: {nece.size_constraint / 1024 / 1024} MB")
    print()
    
    # Run the evolution process
    print("Starting evolution process...")
    best_genome = nece.evolve()
    
    # Display results
    if best_genome:
        best_model = nece.get_best_model()
        model_size = best_model.get_model_size()
        
        print("\n" + "=" * 50)
        print("EVOLUTION RESULTS")
        print("=" * 50)
        print(f"Best model found:")
        print(f"  Size: {model_size/1024/1024:.2f}MB")
        print(f"  Fitness: {best_genome.fitness:.4f}")
        print(f"  Age: {best_genome.age} generations")
        print(f"  Layers: {len(best_genome.layers)}")
        print(f"  Architecture: {' -> '.join([str(layer[1]) for layer in best_genome.layers[:-1]] + [str(best_genome.layers[-1][1])])}")
        print(f"  Activations: {best_genome.activations}")
        
        # Test the model with a sample input
        print("\nTesting model with sample input...")
        try:
            # Create a sample input (flattened 28x28 image)
            sample_input = torch.randn(1, nece.input_size)
            with torch.no_grad():
                output = best_model(sample_input)
            print(f"  Sample output shape: {output.shape}")
            print(f"  Sample output values: {output.softmax(dim=1).squeeze().tolist()}")
            print("  Model is functional!")
        except Exception as e:
            print(f"  Error testing model: {e}")
    else:
        print("\nNo valid model found under size constraint")

if __name__ == "__main__":
    main()