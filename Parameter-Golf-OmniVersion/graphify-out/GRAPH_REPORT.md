# Graph Report - .  (2026-04-09)

## Corpus Check
- 44 files · ~56,510 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 855 nodes · 1318 edges · 25 communities detected
- Extraction: 63% EXTRACTED · 37% INFERRED · 0% AMBIGUOUS · INFERRED: 483 edges (avg confidence: 0.5)
- Token cost: 0 input · 0 output

## God Nodes (most connected - your core abstractions)
1. `train_gpt_mlx_kl.py` - 46 edges
2. `train_gpt_kl.py` - 33 edges
3. `train_gpt_kl_v2.py` - 33 edges
4. `train_gpt_mlx.py` - 31 edges
5. `train_gpt.py` - 26 edges
6. `main()` - 23 edges
7. `hemac_core.py` - 21 edges
8. `main()` - 18 edges
9. `main()` - 18 edges
10. `main()` - 17 edges
11. `.__init__()` - 16 edges
12. `.__init__()` - 16 edges
13. `main()` - 13 edges
14. `AllLayerXSA` - 13 edges
15. `NeuralNetworkGenome` - 12 edges
16. `train_chrysalis_full.py` - 12 edges
17. `GPT` - 12 edges
18. `TestTimeTrainer` - 12 edges
19. `.__init__()` - 11 edges
20. `train_gpt_hyperion_full.py` - 11 edges

## Surprising Connections (you probably didn't know these)
- None detected - all connections are within the same source files.

## Communities

### Community 0 - "eval_mlx.py / evaluate_validation()"
Cohesion: 0.03
Nodes (73): evaluate_validation(), load_checkpoint(), main(), Load model from MLX checkpoint., Run sliding window validation and compute BPB., accumulate_flat_grads(), BackoffNgramMixer, BigramHashEmbedding (+65 more)

### Community 1 - "train_gpt_kl.py / apply_rotary_emb()"
Cohesion: 0.04
Nodes (47): apply_rotary_emb(), BigramHashEmbedding, Block, build_sentencepiece_luts(), CastedLinear, CausalSelfAttention, dequantize_state_dict_int6(), DistributedTokenLoader (+39 more)

### Community 2 - "train_gpt_kl_v2.py / apply_rotary_emb()"
Cohesion: 0.04
Nodes (47): apply_rotary_emb(), BigramHashEmbedding, Block, build_sentencepiece_luts(), CastedLinear, CausalSelfAttention, dequantize_state_dict_int6(), DistributedTokenLoader (+39 more)

### Community 3 - "ttt_module.py / BatchedLoRALayer"
Cohesion: 0.04
Nodes (41): BatchedLoRALayer, BatchedTestTimeAdapter, create_ttt_wrapped_model(), LoRAConfig, LoRALayer, LoRAProjection, ttt_module.py — Test-Time Training (TTT) with LoRA  Online adaptation during eva, Initialize LoRA weights with scaled random values. (+33 more)

### Community 4 - "hemac_core.py / BaseExpert"
Cohesion: 0.05
Nodes (32): BaseExpert, CoreConfig, CoreIO, EntropyGatedExperts, Expert, ExpertResult, export_hemac_model(), GeneralistExpert (+24 more)

### Community 5 - "train_gpt_mlx.py / accumulate_flat_grads()"
Cohesion: 0.07
Nodes (28): accumulate_flat_grads(), Block, build_sentencepiece_luts(), CastedLinear, CausalSelfAttention, clip_grad_tree(), dequantize_state_dict_int8(), eval_val() (+20 more)

### Community 6 - "train_chrysalis_full.py / AdaptiveDepthRouter"
Cohesion: 0.06
Nodes (31): AdaptiveDepthRouter, ChrysalisConfig, ChrysalisGPT, evaluate_with_adaptation(), get_batch(), load_data(), LoRAAdapter, MultiHeadAttention (+23 more)

### Community 7 - "example.py / main()"
Cohesion: 0.06
Nodes (28): Example usage of the NeuroEvolutionary Compression Engine (NECE), CompressedNeuralNetwork, NeuralNetworkGenome, NeuroEvolutionaryCompressionEngine, Population, NeuroEvolutionary Compression Engine (NECE) Evolves neural network architectures, PyTorch neural network with compression capabilities, Represents a neural network genome with variable architecture (+20 more)

### Community 8 - "train_gpt.py / apply_rotary_emb()"
Cohesion: 0.08
Nodes (26): apply_rotary_emb(), Block, build_sentencepiece_luts(), CastedLinear, CausalSelfAttention, dequantize_state_dict_int8(), DistributedTokenLoader, eval_val() (+18 more)

### Community 9 - "xsa_all_layers.py / AllLayerXSA"
Cohesion: 0.06
Nodes (26): AllLayerXSA, CrossLayerAttention, integrate_xsa_into_model(), LayerScalingFactor, xsa_all_layers.py — All-Layer Cross-Scaling Attention (XSA) Extension  Extends c, Initialize weights with small random values., Compute cross-layer attention for current layer.                  Args:, All-Layer Cross-Scaling Attention module.          Extends XSA from partial laye (+18 more)

### Community 10 - "compare_baseline.py / count_params()"
Cohesion: 0.09
Nodes (24): count_params(), main(), quick_train(), Standard dense embedding for baseline, Standard multi-head attention, Standard GPT baseline, StandardAttn, StandardBlock (+16 more)

### Community 11 - "train_gpt_hyperion.py / HierarchicalStateMachine"
Cohesion: 0.07
Nodes (23): HierarchicalStateMachine, HyperionBlock, HyperionConfig, HyperionGPT, ProgramLibrary, Return parameter count for size tracking, 4-level hierarchical state machine.     Each level updates at different timescal, x: (batch, seq_len, atom_dim) - input embeddings         states: dict of current (+15 more)

### Community 12 - "train_gpt_hyperion_v2.py / HierarchicalAttention"
Cohesion: 0.13
Nodes (11): HierarchicalAttention, HyperionBlock, HyperionConfig, HyperionGPT, main(), HYPERION transformer block, Complete HYPERION model, HYPERION configuration (+3 more)

### Community 13 - "train_omniclaw.py / AdaptiveRouter"
Cohesion: 0.15
Nodes (10): AdaptiveRouter, count_params(), OMNIBlock, OMNICLAW, HYPERION: Sparse factorized embedding, CHRYSALIS: Route by difficulty, Hybrid block with hierarchical attention, Master model: HYPERION + CHRYSALIS (+2 more)

### Community 14 - "train_chrysalis.py / AdaptiveBlock"
Cohesion: 0.17
Nodes (9): AdaptiveBlock, AdaptiveRouter, CHRYSALIS, count_params(), CHRYSALIS: Route tokens by predicted difficulty, x: (batch, seq, dim) -> depths: (batch,), Transformer block that can run at different depths, CHRYSALIS with adaptive depth routing (+1 more)

### Community 15 - "train_omniclaw_v2.py / GumbelRouter"
Cohesion: 0.19
Nodes (7): GumbelRouter, OmniBlock, OmniClawV2, OmniConfig, Learnable Router using Gumbel-Softmax for differentiable routing, SparseEmbed, train()

### Community 16 - "kl_innovations.py / BigramHashEmbedding"
Cohesion: 0.2
Nodes (3): BigramHashEmbedding, KaiLean Parameter Golf Innovation Stack Baseline to beat: val_bpb 2.3113 @ 200 s, SWABuffer

### Community 17 - "gptq_calibration.py / auto_calibrate_and_quantize()"
Cohesion: 0.38
Nodes (6): auto_calibrate_and_quantize(), find_5_percentile(), quantize_int4(), Quantizes a tensor to Int4 using a calibration threshold.      Args:         dat, Performs auto-calibration and quantization of a tensor.      Args:         tenso, Finds the 5-percentile value of a given tensor.     This is a placeholder for a

### Community 18 - "mrn.py / ModularRoutingNetwork"
Cohesion: 0.29
Nodes (4): ModularRoutingNetwork, Initializes the ModularRoutingNetwork.          Args:             config (dict o, Implements a Modular Routing Network (MRN) in PyTorch.      This network feature, Performs the forward pass of the Modular Routing Network.          Args:

### Community 19 - "sweep_agent.py / greedy_sweep()"
Cohesion: 1.0
Nodes (2): greedy_sweep(), run_smoke()

### Community 20 - "download_to_external.py"
Cohesion: 1.0
Nodes (0): 

### Community 21 - "quick_train_and_eval.py"
Cohesion: 1.0
Nodes (0): 

### Community 22 - "run_training.py"
Cohesion: 1.0
Nodes (0): 

### Community 23 - "simple_eval.py"
Cohesion: 1.0
Nodes (0): 

### Community 24 - "Compute LoRA scaling factor."
Cohesion: 1.0
Nodes (1): Compute LoRA scaling factor.

## Knowledge Gaps
- **196 isolated node(s):** `Example usage of the NeuroEvolutionary Compression Engine (NECE)`, `Finds the 5-percentile value of a given tensor.     This is a placeholder for a`, `Quantizes a tensor to Int4 using a calibration threshold.      Args:         dat`, `Performs auto-calibration and quantization of a tensor.      Args:         tenso`, `CoreConfig` (+191 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `download_to_external.py`** (1 nodes): `download_to_external.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `quick_train_and_eval.py`** (1 nodes): `quick_train_and_eval.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `run_training.py`** (1 nodes): `run_training.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `simple_eval.py`** (1 nodes): `simple_eval.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Compute LoRA scaling factor.`** (1 nodes): `Compute LoRA scaling factor.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.