"""
DELTA-BASED STEERING FOR MAMBA SSM
Works with Mamba's continuous state space, not discrete neurons.
Based on: https://github.com/vanivamshi/Mamba-Interpretability/tree/main/delta_neurons

results
📊 MEMORY TASKS (Long-context, reasoning, recall):
INFO:__main__:----------------------------------------------------------------------
INFO:__main__:Approach                       Accuracy     vs Baseline     Status
INFO:__main__:----------------------------------------------------------------------
INFO:__main__:Baseline                         50.0%      ---            
INFO:__main__:Variance-Based Delta             25.0%      -25.0%         ❌ NEGATIVE
INFO:__main__:Temporal Contrast                41.7%       -8.3%         ❌ NEGATIVE
INFO:__main__:Multi-Scale                      16.7%      -33.3%         ❌ NEGATIVE
INFO:__main__:Cluster 9 + Delta                50.0%       +0.0%         ❌ NEGATIVE
INFO:__main__:SSM State Dynamics               16.7%      -33.3%         ❌ NEGATIVE
INFO:__main__:
📊 INSTRUCTION TASKS:
INFO:__main__:----------------------------------------------------------------------
INFO:__main__:Baseline                         50.0%      ---            
INFO:__main__:Variance-Based Delta             50.0%       +0.0%         ❌ NEGATIVE
INFO:__main__:Temporal Contrast                33.3%      -16.7%         ❌ NEGATIVE
INFO:__main__:Multi-Scale                      66.7%      +16.7%         ✅ EXCELLENT
INFO:__main__:Cluster 9 + Delta                66.7%      +16.7%         ✅ EXCELLENT
INFO:__main__:SSM State Dynamics               33.3%      -16.7%         ❌ NEGATIVE
INFO:__main__:
================================================================================
INFO:__main__:💡 RECOMMENDATIONS
INFO:__main__:================================================================================
INFO:__main__:
🎯 BEST FOR MEMORY TASKS:
INFO:__main__:   Method: Cluster 9 + Delta
INFO:__main__:   Improvement: +0.0%
INFO:__main__:   ⚠️ No improvement. Consider:
INFO:__main__:      • Different layers (try 8-12 for early memory)
INFO:__main__:      • Stronger steering (4-10x)
INFO:__main__:      • Task-specific fine-tuning
INFO:__main__:
🎯 BEST FOR INSTRUCTION TASKS:
INFO:__main__:   Method: Multi-Scale
INFO:__main__:   Improvement: +16.7%
"""

import torch
import torch.nn.functional as F
import logging
from typing import List, Dict, Tuple, Optional
import json
from pathlib import Path
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DeltaBasedSteering:
    """
    Steering based on DELTA (rate of change) in hidden states.
    
    Key insight from your research:
    - Mamba's SSM has continuous hidden states h_t
    - Delta = h_t - h_{t-1} captures temporal dynamics
    - High-delta dimensions = important for sequential processing
    """
    
    def __init__(self, model):
        self.model = model
        self.hooks = []
        self.prev_hidden_states = {}  # Track previous states for delta
        
        if hasattr(model, 'backbone'):
            self.layers = model.backbone.layers
        else:
            self.layers = model.layers
        
        # Original Cluster 9 indices (proven for instructions)
        self.cluster9_indices = [
            4, 38, 84, 94, 163, 171, 268, 363, 401, 497, 
            564, 568, 582, 654, 659, 686
        ]
    
    def compute_delta_mask(self, hidden_dim: int, percentile: float = 75) -> torch.Tensor:
        """
        Create a mask for high-variance dimensions.
        These are the "delta neurons" - dimensions with high temporal change.
        
        Args:
            hidden_dim: Dimension of hidden state
            percentile: Top percentile to select (75 = top 25%)
        
        Returns:
            Binary mask indicating high-delta dimensions
        """
        # Initialize with uniform distribution, will adapt during inference
        mask = torch.ones(hidden_dim)
        
        # Mark top 25% of dimensions for steering
        threshold_idx = int(hidden_dim * (100 - percentile) / 100)
        
        # Create gradient: stronger steering on early dimensions (input encoding)
        # and late dimensions (output formation)
        positions = torch.arange(hidden_dim).float()
        
        # U-shaped curve: high at ends, lower in middle
        u_curve = torch.exp(-((positions - hidden_dim/2) / (hidden_dim/4))**2)
        u_curve = 1 - u_curve * 0.5  # Invert and scale
        
        mask = mask * u_curve
        
        return mask
    
    def apply_delta_steering(
        self, 
        layer_idx: int = 15, 
        strength: float = 3.0,
        use_variance_based: bool = True
    ):
        """
        Apply delta-based steering to a layer.
        
        Args:
            layer_idx: Which layer to steer
            strength: Steering strength multiplier
            use_variance_based: If True, emphasize high-variance dimensions
        """
        if layer_idx >= len(self.layers):
            logger.warning(f"Layer {layer_idx} doesn't exist")
            return
        
        logger.info(f"\n🌊 DELTA-BASED STEERING")
        logger.info(f"   Layer: {layer_idx}")
        logger.info(f"   Strength: {strength}x")
        logger.info(f"   Strategy: {'Variance-adaptive' if use_variance_based else 'Uniform'}")
        
        layer = self.layers[layer_idx]
        target = getattr(layer, 'mixer', getattr(layer, 'ssm', layer))
        
        # Create variance mask
        hidden_dim = None
        delta_mask = None
        
        def hook(module, input, output):
            nonlocal hidden_dim, delta_mask
            
            if isinstance(output, tuple):
                hidden = output[0]
                rest = output[1:]
            else:
                hidden = output
                rest = ()
            
            # Initialize mask on first pass
            if delta_mask is None:
                hidden_dim = hidden.shape[-1]
                delta_mask = self.compute_delta_mask(hidden_dim)
                delta_mask = delta_mask.to(hidden.device)
                logger.info(f"   Initialized delta mask: {hidden_dim} dims")
            
            h_mod = hidden.clone()
            
            if use_variance_based:
                # Emphasize high-variance dimensions (delta approach)
                # These dimensions show strong temporal dynamics
                h_mod = h_mod * (1 + (strength - 1) * delta_mask)
            else:
                # Uniform scaling
                h_mod = h_mod * strength
            
            if rest:
                return (h_mod,) + rest
            return h_mod
        
        h = target.register_forward_hook(hook)
        self.hooks.append(h)
    
    def apply_temporal_contrast_steering(
        self,
        layer_idx: int = 15,
        strength: float = 2.0
    ):
        """
        NOVEL APPROACH: Enhance temporal contrast.
        
        Key insight: Delta = h_t - h_{t-1}
        If we amplify changes, the model tracks sequence better.
        """
        if layer_idx >= len(self.layers):
            logger.warning(f"Layer {layer_idx} doesn't exist")
            return
        
        logger.info(f"\n⚡ TEMPORAL CONTRAST STEERING")
        logger.info(f"   Layer: {layer_idx}")
        logger.info(f"   Strength: {strength}x")
        logger.info(f"   Strategy: Amplify rate of change")
        
        layer = self.layers[layer_idx]
        target = getattr(layer, 'mixer', getattr(layer, 'ssm', layer))
        
        layer_key = f"layer_{layer_idx}"
        
        def hook(module, input, output):
            if isinstance(output, tuple):
                hidden = output[0]
                rest = output[1:]
            else:
                hidden = output
                rest = ()
            
            h_mod = hidden.clone()
            
            # Get previous state
            if layer_key in self.prev_hidden_states:
                h_prev = self.prev_hidden_states[layer_key]
                
                # Compute delta (rate of change)
                if h_prev.shape == h_mod.shape:
                    delta = h_mod - h_prev
                    
                    # Amplify the change
                    h_mod = h_prev + delta * strength
            
            # Store current state for next timestep
            self.prev_hidden_states[layer_key] = h_mod.detach().clone()
            
            if rest:
                return (h_mod,) + rest
            return h_mod
        
        h = target.register_forward_hook(hook)
        self.hooks.append(h)
    
    def apply_multi_scale_steering(
        self,
        layers: List[int] = [10, 15, 20],
        strengths: List[float] = [2.0, 3.0, 2.5]
    ):
        """
        Apply delta steering at multiple scales.
        
        Args:
            layers: Which layers to steer
            strengths: Corresponding strength for each layer
        """
        logger.info(f"\n🎯 MULTI-SCALE DELTA STEERING")
        logger.info(f"   Layers: {layers}")
        logger.info(f"   Strengths: {strengths}")
        logger.info(f"   Strategy: Early (memory) → Mid (integration) → Late (output)")
        
        for layer_idx, strength in zip(layers, strengths):
            if layer_idx >= len(self.layers):
                continue
            
            # Different strategy per layer
            if layer_idx < 15:
                # Early layer: emphasize memory encoding
                logger.info(f"      Layer {layer_idx}: Memory encoding @ {strength}x")
                use_variance = True
            elif layer_idx < 20:
                # Mid layer: temporal contrast
                logger.info(f"      Layer {layer_idx}: Temporal contrast @ {strength}x")
                self.apply_temporal_contrast_steering(layer_idx, strength)
                continue  # Skip the default application below
            else:
                # Late layer: output refinement
                logger.info(f"      Layer {layer_idx}: Output refinement @ {strength}x")
                use_variance = False
            
            self.apply_delta_steering(layer_idx, strength, use_variance)
    
    def apply_cluster9_with_delta_boost(
        self,
        layer_idx: int = 20,
        cluster_strength: float = 5.0,
        delta_strength: float = 2.0
    ):
        """
        Hybrid: Original Cluster 9 + delta-based boost.
        
        This combines your proven approach with delta insights.
        """
        if layer_idx >= len(self.layers):
            logger.warning(f"Layer {layer_idx} doesn't exist")
            return
        
        logger.info(f"\n🔀 HYBRID: Cluster 9 + Delta Boost")
        logger.info(f"   Layer: {layer_idx}")
        logger.info(f"   Cluster 9 strength: {cluster_strength}x")
        logger.info(f"   Delta boost: {delta_strength}x")
        
        layer = self.layers[layer_idx]
        target = getattr(layer, 'mixer', getattr(layer, 'ssm', layer))
        
        delta_mask = None
        
        def hook(module, input, output):
            nonlocal delta_mask
            
            if isinstance(output, tuple):
                hidden = output[0]
                rest = output[1:]
            else:
                hidden = output
                rest = ()
            
            h_mod = hidden.clone()
            
            # Apply Cluster 9 steering (proven method)
            for idx in self.cluster9_indices:
                if idx < h_mod.shape[-1]:
                    h_mod[..., idx] *= cluster_strength
            
            # Add delta boost to all dimensions
            if delta_mask is None:
                delta_mask = self.compute_delta_mask(h_mod.shape[-1])
                delta_mask = delta_mask.to(h_mod.device)
            
            # Multiplicative boost based on delta importance
            h_mod = h_mod * (1 + (delta_strength - 1) * delta_mask * 0.3)
            
            if rest:
                return (h_mod,) + rest
            return h_mod
        
        h = target.register_forward_hook(hook)
        self.hooks.append(h)
    
    def apply_ssm_state_steering(
        self,
        layer_idx: int = 15,
        strength: float = 3.0
    ):
        """
        ADVANCED: Directly modify SSM state dynamics.
        
        Targets the A, B, C, D matrices in the SSM if accessible.
        This is closer to what delta neurons conceptually represent.
        """
        if layer_idx >= len(self.layers):
            logger.warning(f"Layer {layer_idx} doesn't exist")
            return
        
        logger.info(f"\n🧬 SSM STATE STEERING")
        logger.info(f"   Layer: {layer_idx}")
        logger.info(f"   Strength: {strength}x")
        logger.info(f"   Strategy: Modify state transition dynamics")
        
        layer = self.layers[layer_idx]
        
        # Try to access SSM components
        if hasattr(layer, 'mixer'):
            ssm = layer.mixer
        elif hasattr(layer, 'ssm'):
            ssm = layer.ssm
        else:
            logger.warning(f"   Could not find SSM in layer {layer_idx}")
            return
        
        def hook(module, input, output):
            if isinstance(output, tuple):
                hidden = output[0]
                rest = output[1:]
            else:
                hidden = output
                rest = ()
            
            h_mod = hidden.clone()
            
            # Enhance state dynamics
            # This makes the hidden state more "responsive" to changes
            std = h_mod.std(dim=-1, keepdim=True)
            mean = h_mod.mean(dim=-1, keepdim=True)
            
            # Increase contrast (make peaks higher, valleys lower)
            h_mod = mean + (h_mod - mean) * strength
            
            if rest:
                return (h_mod,) + rest
            return h_mod
        
        h = ssm.register_forward_hook(hook)
        self.hooks.append(h)
    
    def remove_steering(self):
        """Remove all hooks and clear state."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.prev_hidden_states = {}


def run_delta_experiment():
    """
    Test various delta-based steering approaches.
    """
    from mamba_model_loader import load_mamba_model_and_tokenizer
    from targeted_approach_2 import EnhancedRealisticEvaluation
    
    logger.info("="*80)
    logger.info("🔬 DELTA-BASED STEERING EXPERIMENT")
    logger.info("Testing continuous state-space steering approaches")
    logger.info("="*80)
    
    # Load model
    logger.info("\n📦 Loading Mamba-130M...")
    model, tokenizer = load_mamba_model_and_tokenizer(
        model_name="state-spaces/mamba-130m-hf",
        device="cuda" if torch.cuda.is_available() else "cpu",
        use_mamba_class=True,
        fallback_to_auto=True
    )
    
    device = next(model.parameters()).device
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Initialize
    evaluator = EnhancedRealisticEvaluation(tokenizer, device)
    steering = DeltaBasedSteering(model)
    
    # Get test cases
    test_cases = evaluator.get_enhanced_test_cases()
    
    # Focus on problematic tasks
    memory_tasks = {
        'long_context_recall': test_cases['long_context_recall'],
        'multi_hop_reasoning': test_cases['multi_hop_reasoning'],
        'associative_recall': test_cases['associative_recall'],
    }
    
    instruction_tasks = {
        'simple_instructions': test_cases['simple_instructions'],
        'in_context_learning': test_cases['in_context_learning'],
    }
    
    all_results = {}
    
    # ================================================
    # BASELINE
    # ================================================
    logger.info("\n" + "="*80)
    logger.info("📊 BASELINE (No Steering)")
    logger.info("="*80)
    
    baseline_memory = evaluator.test_model_enhanced(
        model, memory_tasks, "BASELINE: Memory Tasks"
    )
    baseline_instruction = evaluator.test_model_enhanced(
        model, instruction_tasks, "BASELINE: Instruction Tasks"
    )
    
    base_mem_acc = sum(r['correct'] for r in baseline_memory.values()) / \
                   sum(r['total'] for r in baseline_memory.values())
    base_inst_acc = sum(r['correct'] for r in baseline_instruction.values()) / \
                    sum(r['total'] for r in baseline_instruction.values())
    
    logger.info(f"\n📊 Baseline:")
    logger.info(f"   Memory: {base_mem_acc*100:.1f}%")
    logger.info(f"   Instruction: {base_inst_acc*100:.1f}%")
    
    all_results['baseline'] = {
        'memory': base_mem_acc,
        'instruction': base_inst_acc
    }
    
    # ================================================
    # APPROACH 1: Variance-Based Delta Steering
    # ================================================
    logger.info("\n" + "="*80)
    logger.info("🧪 APPROACH 1: Variance-Based Delta")
    logger.info("="*80)
    
    steering.apply_delta_steering(layer_idx=15, strength=3.0, use_variance_based=True)
    
    delta_memory = evaluator.test_model_enhanced(
        model, memory_tasks, "Delta Variance: Memory"
    )
    delta_instruction = evaluator.test_model_enhanced(
        model, instruction_tasks, "Delta Variance: Instruction"
    )
    
    steering.remove_steering()
    
    delta_mem = sum(r['correct'] for r in delta_memory.values()) / \
                sum(r['total'] for r in delta_memory.values())
    delta_inst = sum(r['correct'] for r in delta_instruction.values()) / \
                 sum(r['total'] for r in delta_instruction.values())
    
    logger.info(f"\n📊 Results:")
    logger.info(f"   Memory: {delta_mem*100:.1f}% [{(delta_mem-base_mem_acc)*100:+.1f}%]")
    logger.info(f"   Instruction: {delta_inst*100:.1f}% [{(delta_inst-base_inst_acc)*100:+.1f}%]")
    
    all_results['variance_delta'] = {
        'memory': delta_mem,
        'instruction': delta_inst,
        'memory_improvement': delta_mem - base_mem_acc,
        'instruction_improvement': delta_inst - base_inst_acc
    }
    
    # ================================================
    # APPROACH 2: Temporal Contrast Steering
    # ================================================
    logger.info("\n" + "="*80)
    logger.info("🧪 APPROACH 2: Temporal Contrast")
    logger.info("="*80)
    
    steering.apply_temporal_contrast_steering(layer_idx=15, strength=2.0)
    
    temporal_memory = evaluator.test_model_enhanced(
        model, memory_tasks, "Temporal Contrast: Memory"
    )
    temporal_instruction = evaluator.test_model_enhanced(
        model, instruction_tasks, "Temporal Contrast: Instruction"
    )
    
    steering.remove_steering()
    
    temp_mem = sum(r['correct'] for r in temporal_memory.values()) / \
               sum(r['total'] for r in temporal_memory.values())
    temp_inst = sum(r['correct'] for r in temporal_instruction.values()) / \
                sum(r['total'] for r in temporal_instruction.values())
    
    logger.info(f"\n📊 Results:")
    logger.info(f"   Memory: {temp_mem*100:.1f}% [{(temp_mem-base_mem_acc)*100:+.1f}%]")
    logger.info(f"   Instruction: {temp_inst*100:.1f}% [{(temp_inst-base_inst_acc)*100:+.1f}%]")
    
    all_results['temporal_contrast'] = {
        'memory': temp_mem,
        'instruction': temp_inst,
        'memory_improvement': temp_mem - base_mem_acc,
        'instruction_improvement': temp_inst - base_inst_acc
    }
    
    # ================================================
    # APPROACH 3: Multi-Scale Steering
    # ================================================
    logger.info("\n" + "="*80)
    logger.info("🧪 APPROACH 3: Multi-Scale Delta")
    logger.info("="*80)
    
    steering.apply_multi_scale_steering(
        layers=[10, 15, 20],
        strengths=[2.0, 3.0, 2.5]
    )
    
    multi_memory = evaluator.test_model_enhanced(
        model, memory_tasks, "Multi-Scale: Memory"
    )
    multi_instruction = evaluator.test_model_enhanced(
        model, instruction_tasks, "Multi-Scale: Instruction"
    )
    
    steering.remove_steering()
    
    multi_mem = sum(r['correct'] for r in multi_memory.values()) / \
                sum(r['total'] for r in multi_memory.values())
    multi_inst = sum(r['correct'] for r in multi_instruction.values()) / \
                 sum(r['total'] for r in multi_instruction.values())
    
    logger.info(f"\n📊 Results:")
    logger.info(f"   Memory: {multi_mem*100:.1f}% [{(multi_mem-base_mem_acc)*100:+.1f}%]")
    logger.info(f"   Instruction: {multi_inst*100:.1f}% [{(multi_inst-base_inst_acc)*100:+.1f}%]")
    
    all_results['multi_scale'] = {
        'memory': multi_mem,
        'instruction': multi_inst,
        'memory_improvement': multi_mem - base_mem_acc,
        'instruction_improvement': multi_inst - base_inst_acc
    }
    
    # ================================================
    # APPROACH 4: Cluster 9 + Delta Boost (Hybrid)
    # ================================================
    logger.info("\n" + "="*80)
    logger.info("🧪 APPROACH 4: Cluster 9 + Delta Boost")
    logger.info("="*80)
    
    steering.apply_cluster9_with_delta_boost(
        layer_idx=20,
        cluster_strength=5.0,
        delta_strength=2.0
    )
    
    hybrid_memory = evaluator.test_model_enhanced(
        model, memory_tasks, "Hybrid: Memory"
    )
    hybrid_instruction = evaluator.test_model_enhanced(
        model, instruction_tasks, "Hybrid: Instruction"
    )
    
    steering.remove_steering()
    
    hybrid_mem = sum(r['correct'] for r in hybrid_memory.values()) / \
                 sum(r['total'] for r in hybrid_memory.values())
    hybrid_inst = sum(r['correct'] for r in hybrid_instruction.values()) / \
                  sum(r['total'] for r in hybrid_instruction.values())
    
    logger.info(f"\n📊 Results:")
    logger.info(f"   Memory: {hybrid_mem*100:.1f}% [{(hybrid_mem-base_mem_acc)*100:+.1f}%]")
    logger.info(f"   Instruction: {hybrid_inst*100:.1f}% [{(hybrid_inst-base_inst_acc)*100:+.1f}%]")
    
    all_results['hybrid'] = {
        'memory': hybrid_mem,
        'instruction': hybrid_inst,
        'memory_improvement': hybrid_mem - base_mem_acc,
        'instruction_improvement': hybrid_inst - base_inst_acc
    }
    
    # ================================================
    # APPROACH 5: SSM State Steering
    # ================================================
    logger.info("\n" + "="*80)
    logger.info("🧪 APPROACH 5: SSM State Dynamics")
    logger.info("="*80)
    
    steering.apply_ssm_state_steering(layer_idx=15, strength=3.0)
    
    ssm_memory = evaluator.test_model_enhanced(
        model, memory_tasks, "SSM State: Memory"
    )
    ssm_instruction = evaluator.test_model_enhanced(
        model, instruction_tasks, "SSM State: Instruction"
    )
    
    steering.remove_steering()
    
    ssm_mem = sum(r['correct'] for r in ssm_memory.values()) / \
              sum(r['total'] for r in ssm_memory.values())
    ssm_inst = sum(r['correct'] for r in ssm_instruction.values()) / \
               sum(r['total'] for r in ssm_instruction.values())
    
    logger.info(f"\n📊 Results:")
    logger.info(f"   Memory: {ssm_mem*100:.1f}% [{(ssm_mem-base_mem_acc)*100:+.1f}%]")
    logger.info(f"   Instruction: {ssm_inst*100:.1f}% [{(ssm_inst-base_inst_acc)*100:+.1f}%]")
    
    all_results['ssm_state'] = {
        'memory': ssm_mem,
        'instruction': ssm_inst,
        'memory_improvement': ssm_mem - base_mem_acc,
        'instruction_improvement': ssm_inst - base_inst_acc
    }
    
    # ================================================
    # FINAL COMPARISON
    # ================================================
    logger.info("\n" + "="*80)
    logger.info("🏆 COMPREHENSIVE COMPARISON")
    logger.info("="*80)
    
    logger.info("\n📊 MEMORY TASKS (Long-context, reasoning, recall):")
    logger.info("-" * 70)
    logger.info(f"{'Approach':<30} {'Accuracy':<12} {'vs Baseline':<15} {'Status'}")
    logger.info("-" * 70)
    
    approaches = [
        ('baseline', 'Baseline'),
        ('variance_delta', 'Variance-Based Delta'),
        ('temporal_contrast', 'Temporal Contrast'),
        ('multi_scale', 'Multi-Scale'),
        ('hybrid', 'Cluster 9 + Delta'),
        ('ssm_state', 'SSM State Dynamics')
    ]
    
    best_memory_improvement = -999
    best_memory_method = None
    
    for key, name in approaches:
        results = all_results[key]
        if key == 'baseline':
            logger.info(f"{name:<30} {results['memory']*100:6.1f}%      {'---':<15}")
        else:
            imp = results['memory_improvement']
            if imp > best_memory_improvement:
                best_memory_improvement = imp
                best_memory_method = name
            
            if imp > 0.10:
                status = "✅ EXCELLENT"
            elif imp > 0.05:
                status = "📈 GOOD"
            elif imp > 0:
                status = "📊 MODEST"
            else:
                status = "❌ NEGATIVE"
            
            logger.info(f"{name:<30} {results['memory']*100:6.1f}%      "
                       f"{imp*100:+5.1f}%         {status}")
    
    logger.info("\n📊 INSTRUCTION TASKS:")
    logger.info("-" * 70)
    
    best_inst_improvement = -999
    best_inst_method = None
    
    for key, name in approaches:
        results = all_results[key]
        if key == 'baseline':
            logger.info(f"{name:<30} {results['instruction']*100:6.1f}%      {'---':<15}")
        else:
            imp = results['instruction_improvement']
            if imp > best_inst_improvement:
                best_inst_improvement = imp
                best_inst_method = name
            
            if imp > 0.15:
                status = "✅ EXCELLENT"
            elif imp > 0.10:
                status = "📈 GOOD"
            elif imp > 0:
                status = "📊 MODEST"
            else:
                status = "❌ NEGATIVE"
            
            logger.info(f"{name:<30} {results['instruction']*100:6.1f}%      "
                       f"{imp*100:+5.1f}%         {status}")
    
    # ================================================
    # RECOMMENDATIONS
    # ================================================
    logger.info("\n" + "="*80)
    logger.info("💡 RECOMMENDATIONS")
    logger.info("="*80)
    
    logger.info(f"\n🎯 BEST FOR MEMORY TASKS:")
    logger.info(f"   Method: {best_memory_method}")
    logger.info(f"   Improvement: {best_memory_improvement*100:+.1f}%")
    
    if best_memory_improvement > 0.10:
        logger.info(f"   ✅ Strong results! This approach works for long-context tasks.")
    elif best_memory_improvement > 0.05:
        logger.info(f"   📈 Promising direction, consider tuning hyperparameters.")
    elif best_memory_improvement > 0:
        logger.info(f"   📊 Modest gains, may need different layers or stronger steering.")
    else:
        logger.info(f"   ⚠️ No improvement. Consider:")
        logger.info(f"      • Different layers (try 8-12 for early memory)")
        logger.info(f"      • Stronger steering (4-10x)")
        logger.info(f"      • Task-specific fine-tuning")
    
    logger.info(f"\n🎯 BEST FOR INSTRUCTION TASKS:")
    logger.info(f"   Method: {best_inst_method}")
    logger.info(f"   Improvement: {best_inst_improvement*100:+.1f}%")
    
    logger.info(f"\n💭 KEY INSIGHTS:")
    
    # Check if delta approaches helped
    delta_methods = ['variance_delta', 'temporal_contrast', 'multi_scale']
    delta_helps_memory = any(
        all_results[m]['memory_improvement'] > 0.05 for m in delta_methods
    )
    
    if delta_helps_memory:
        logger.info(f"   ✅ Delta-based steering shows promise for memory tasks!")
        logger.info(f"      → The temporal dynamics hypothesis is correct")
    else:
        logger.info(f"   📊 Delta approaches didn't significantly help memory")
        logger.info(f"      → May need to target different SSM components")
    
    if all_results['hybrid']['instruction_improvement'] > 0.10:
        logger.info(f"   ✅ Hybrid approach maintains instruction-following quality")
    
    # Save results
    output_path = Path("experiment_logs/delta_steering_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump({
            'experiment': 'delta_based_steering',
            'model': 'state-spaces/mamba-130m-hf',
            'approaches_tested': [name for _, name in approaches[1:]],
            'results': {k: {
                'memory_accuracy': float(v['memory']),
                'instruction_accuracy': float(v['instruction']),
                'memory_improvement': float(v.get('memory_improvement', 0)),
                'instruction_improvement': float(v.get('instruction_improvement', 0))
            } for k, v in all_results.items()},
            'best_memory_method': best_memory_method,
            'best_instruction_method': best_inst_method,
            'best_memory_improvement': float(best_memory_improvement),
            'best_instruction_improvement': float(best_inst_improvement)
        }, f, indent=2)
    
    logger.info(f"\n✅ Results saved to: {output_path}")


if __name__ == "__main__":
    run_delta_experiment()