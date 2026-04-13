import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

class ModularRoutingNetwork(nn.Module):
    """
    Implements a Modular Routing Network (MRN) in PyTorch.

    This network features trainable weights, softmax-based routing,
    top-K expert selection, and entropy calculation for the routing distribution.
    It includes support for configuration loading and placeholders for future
    distillation hooks.
    """
    def __init__(self, config):
        """
        Initializes the ModularRoutingNetwork.

        Args:
            config (dict or str): A dictionary containing network configuration
                                  or a path to a YAML configuration file.
        """
        super().__init__()

        # --- Configuration Loading ---
        if isinstance(config, str):
            with open(config, 'r') as f:
                self.config = yaml.safe_load(f)
        elif isinstance(config, dict):
            self.config = config
        else:
            raise TypeError("Config must be a dictionary or a path to a YAML file.")

        # Extract configuration parameters with defaults
        self.num_experts = self.config.get('num_experts', 8)
        self.hidden_dim = self.config.get('hidden_dim', 256)
        self.output_dim = self.config.get('output_dim', 10)
        self.top_k = self.config.get('top_k', 2)
        self.use_softmax_routing = self.config.get('use_softmax_routing', True)
        self.use_entropy_regularization = self.config.get('use_entropy_regularization', False)
        self.entropy_coeff = self.config.get('entropy_coeff', 0.01)

        if self.top_k > self.num_experts:
            raise ValueError(f"top_k ({self.top_k}) cannot be greater than num_experts ({self.num_experts}).")

        # --- Network Components ---

        # Routing network: Takes input and outputs Gating weights for experts.
        # This is a simplified example; a more complex router could process
        # the input in a more sophisticated way before generating gating weights.
        # Input dimension needs to be determined or passed explicitly.
        # Let's consider a common scenario where input_dim is needed.
        self.input_dim = self.config.get('input_dim', None)
        if self.input_dim is None:
            print("Warning: input_dim not specified in config. Assuming it will be inferred or set later.")

        if self.input_dim is not None:
            self.gate_layer = nn.Linear(self.input_dim, self.num_experts)
        else:
            # Placeholder if input_dim is not available at init
            self.gate_layer = None

        # Experts: Each expert is a small neural network.
        # Here, using simple feed-forward networks as experts.
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.input_dim if self.input_dim is not None else 1, self.hidden_dim),
                nn.ReLU(),
                # For simplicity, each expert outputs to the final output_dim directly.
                # In some MoE architectures, experts might have intermediate outputs
                # that are then combined.
                nn.Linear(self.hidden_dim, self.output_dim)
            ) for _ in range(self.num_experts)
        ])

        # --- Placeholder for Distillation Hooks ---
        # These would be used to connect this network's outputs or intermediate
        # activations to a teacher model for knowledge distillation.
        # Example: self.distillation_hook_1 = SomeDistillationModule()

    def forward(self, x):
        """
        Performs the forward pass of the Modular Routing Network.

        Args:
            x (torch.Tensor): The input tensor. Shape: (batch_size, input_dim).

        Returns:
            tuple: A tuple containing:
                - output (torch.Tensor): The final output of the network.
                                         Shape: (batch_size, output_dim).
                - routing_weights (torch.Tensor): The raw gating scores before softmax.
                                                  Shape: (batch_size, num_experts).
                - top_k_indices (torch.Tensor): Indices of the selected top-k experts by batch item.
                                                Shape: (batch_size, top_k).
                - entropy (torch.Tensor): The entropy of the routing distribution (if enabled).
                                          Shape: scalar.
        """
        if self.gate_layer is None:
            # Infer input_dim if not provided during initialization
            if self.input_dim is None:
                self.input_dim = x.size(-1)
                # Re-initialize gate_layer and experts when input_dim is known
                self.gate_layer = nn.Linear(self.input_dim, self.num_experts)
                self.experts = nn.ModuleList([
                    nn.Sequential(
                        nn.Linear(self.input_dim, self.hidden_dim),
                        nn.ReLU(),
                        nn.Linear(self.hidden_dim, self.output_dim)
                    ) for _ in range(self.num_experts)
                ])
                print(f"Input dimension inferred: {self.input_dim}. Network re-initialized.")
            else:
                 raise RuntimeError("Input dimension not initialized. Please set input_dim in config or call forward with a tensor.")

        # 1. Generate gating weights
        # raw_gating_scores shape: (batch_size, num_experts)
        raw_gating_scores = self.gate_layer(x)

        # 2. Apply softmax for routing probabilities if enabled
        if self.use_softmax_routing:
            routing_probs = F.softmax(raw_gating_scores, dim=-1)
        else:
            # If not using softmax, treat raw scores as direct weights
            # This might require normalization or a different interpretation
            # depending on the specific non-softmax routing mechanism.
            # For simplicity, we'll use them directly but warn about it.
            print("Warning: Softmax routing disabled. Using raw gating scores directly.")
            routing_probs = raw_gating_scores # This is likely not ideal without further processing

        # 3. Select top-K experts
        # torch.topk returns values and indices. We only need indices here for routing.
        # top_k_values, top_k_indices = torch.topk(routing_probs, self.top_k, dim=-1)
        _, top_k_indices = torch.topk(routing_probs, self.top_k, dim=-1)
        # top_k_indices shape: (batch_size, top_k)

        # 4. Process input through selected experts and combine outputs
        # Initialize output tensor
        batch_size = x.size(0)
        # Initialize output for combined expert results.
        # We will sum the results from the top-k experts, weighted by their softmax probabilities.
        combined_output = torch.zeros(batch_size, self.output_dim, device=x.device)

        # Iterate over the batch
        for i in range(batch_size):
            expert_outputs = []
            expert_weights = []

            # For each item in the batch, pass its data through the selected top-k experts
            for k in range(self.top_k):
                expert_idx = top_k_indices[i, k].item() # Get the index of the k-th expert
                expert = self.experts[expert_idx]
                expert_input = x[i].unsqueeze(0) # Pass single item, shape (1, input_dim)

                # Get the raw output from the expert
                expert_output = expert(expert_input) # Shape (1, output_dim)
                expert_outputs.append(expert_output)

                # Get the corresponding routing probability for this expert for this batch item
                # This is the probability assigned by the gating network to the chosen expert.
                weight = routing_probs[i, expert_idx].unsqueeze(0) # Shape (1,)
                expert_weights.append(weight)

            # Combine outputs from top-k experts for this batch item.
            # Sum weighted outputs: sum(weight_i * output_i)
            # Scale the per-expert weights so they sum to 1 over the selected experts for this item.
            # This is a common approach in sparse MoE models.
            current_expert_weights = torch.stack(expert_weights, dim=0).squeeze(-1) # shape (top_k,)
            normalized_weights = F.softmax(current_expert_weights, dim=0) # Normalize weights for the top-k experts

            current_expert_outputs = torch.stack(expert_outputs, dim=0).squeeze(1) # shape (top_k, output_dim)

            # Weighted sum: Dot product of normalized weights and expert outputs
            combined_output[i, :] = torch.matmul(normalized_weights, current_expert_outputs)


        # 5. Calculate entropy if enabled
        entropy = torch.tensor(0.0, device=x.device)
        if self.use_entropy_regularization:
            # Entropy of the routing distribution across all experts for each batch item
            # H(p) = - sum(p * log(p))
            # We want to encourage sparsity, so we'd want to maximize entropy of *selected* experts,
            # or penalize the gating network for low probability assignments.
            # A common approach is to encourage the gates to assign non-zero probabilities
            # to *different* sets of experts across the batch, or to make the distributions
            # more uniform over the selected k experts. For simplicity here, we calculate
            # the entropy of the full routing_probs distribution.
            # A more advanced strategy might compute entropy over the 'top_k_values' after normalization.

            # For the full distribution:
            # log_probs = torch.log(routing_probs + 1e-9) # Add epsilon to prevent log(0)
            # entropy = -torch.sum(routing_probs * log_probs, dim=-1).mean() # Mean entropy over batch

            # Alternatively, to encourage diverse expert usage across the batch,
            # we can compute the average entropy of the gating probabilities.
            # More nuanced entropy calculation for sparsity might involve
            # the 'top_k_values' or a specific form of load balancing.

            # Let's implement a simple average entropy of the softmax distribution:
            log_routing_probs = torch.log(routing_probs + 1e-9) # Add epsilon for numerical stability
            batch_entropy = -torch.sum(routing_probs * log_routing_probs, dim=-1)
            entropy = torch.mean(batch_entropy) # Average entropy across the batch

        # --- Placeholder for Distillation Output ---
        # If distillation hooks were implemented, they would be called here.
        # Example: distillation_loss = self.distillation_hook_1(output, teacher_outputs)

        return combined_output, raw_gating_scores, top_k_indices, entropy

# --- Example Usage ---

if __name__ == '__main__':
    # Create a dummy configuration
    # Example 1: Basic config dictionary
    config_dict = {
        'num_experts': 8,
        'hidden_dim': 128,
        'output_dim': 10,
        'top_k': 2,
        'use_softmax_routing': True,
        'use_entropy_regularization': True,
        'entropy_coeff': 0.005
    }

    # Example 2: Configuration from a YAML file (create one for testing)
    yaml_config_content = """
num_experts: 6
hidden_dim: 256
output_dim: 10
top_k: 3
use_softmax_routing: true
use_entropy_regularization: false
entropy_coeff: 0.01
input_dim: 768 # Explicitly set input dimension
"""
    with open("mrn_config.yaml", "w") as f:
        f.write(yaml_config_content)
    config_path = "mrn_config.yaml"

    print("--- Testing with dictionary config ---")
    # Instantiate the network with dictionary config
    # We need to provide input_dim here as it's not inferrable from an empty init
    # Or, we can let the forward pass infer it if not specified in the dict config.
    # For this example, let's specify it.
    config_dict_with_input = config_dict.copy()
    config_dict_with_input['input_dim'] = 768 # Assume input dim is 768 (like BERT)
    mrn_net = ModularRoutingNetwork(config_dict_with_input)

    # Create dummy input data
    batch_size = 4
    input_dim = 768
    dummy_input = torch.randn(batch_size, input_dim)

    # Forward pass
    output, routing_weights, top_k_indices, entropy = mrn_net(dummy_input)

    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Routing weights shape: {routing_weights.shape}")
    print(f"Top-k indices shape: {top_k_indices.shape}")
    print(f"Entropy: {entropy.item():.4f}")

    print("\n--- Testing with YAML config file ---")
    # Instantiate the network with YAML config
    mrn_net_yaml = ModularRoutingNetwork(config_path)
    output_yaml, _, _, _ = mrn_net_yaml(dummy_input)
    print(f"Output shape (YAML config): {output_yaml.shape}")