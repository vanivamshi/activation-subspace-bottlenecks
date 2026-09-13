"""
MEMORY-FOCUSED STEERING FOR MAMBA SSM
Addresses Mamba's specific weakness on in-context learning and long-context recall.

Key insight from research:
- Mamba struggles with: copying, in-context learning, associative recall
- Root cause: SSM state decay (HiPPO) forgets information too quickly
- Solution: Prevent premature forgetting by stabilizing the hidden state

result
📊 MEMORY TASKS (The hard ones!):
Includes queries from: squad_queries
----------------------------------------------------------------------
Method                         Accuracy     Change       Status
----------------------------------------------------------------------
Baseline                         31.6%      ---
Memory Retention                 34.2%       +2.6%      📊
Anti Decay                       34.2%       +2.6%      📊
Associative                      28.9%       -2.6%      ❌
Cascade                          23.7%       -7.9%      ❌
Minimal                          36.8%       +5.3%      📈

📊 INSTRUCTION TASKS (Make sure we don't break these):
----------------------------------------------------------------------
Baseline                         33.3%      ---
Memory Retention                 33.3%       +0.0%      📊
Anti Decay                       66.7%      +33.3%      ✅
Associative                      66.7%      +33.3%      ✅
Cascade                          33.3%       +0.0%      📊
Minimal                          66.7%      +33.3%      ✅
"""

import torch
import torch.nn.functional as F
import logging
from typing import List, Dict, Tuple, Optional
import json
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MemoryEnhancedSteering:
    """
    Steering specifically designed to improve Mamba's memory retention.
    
    Approach:
    1. Reduce state decay (combat HiPPO forgetting)
    2. Enhance early layers (encode memory better)
    3. Stabilize recurrent dynamics (prevent information loss)
    """
    
    def __init__(self, model):
        self.model = model
        self.hooks = []
        
        if hasattr(model, 'backbone'):
            self.layers = model.backbone.layers
        else:
            self.layers = model.layers
        
        # Cluster 9 (proven for instructions)
        self.cluster9_indices = [
            4, 38, 84, 94, 163, 171, 268, 363, 401, 497, 
            564, 568, 582, 654, 659, 686
        ]
    
    def apply_memory_retention_steering(
        self,
        layer_idx: int = 8,  # EARLY layer for memory encoding
        strength: float = 1.5,  # Gentle boost (not aggressive)
        stabilize: bool = True
    ):
        """
        APPROACH 1: Memory Retention
        Prevent information from decaying too quickly in the SSM state.
        
        Early layers encode input → we want them to retain more.
        """
        if layer_idx >= len(self.layers):
            logger.warning(f"Layer {layer_idx} doesn't exist")
            return
        
        logger.info(f"\n💾 MEMORY RETENTION STEERING")
        logger.info(f"   Layer: {layer_idx} (early - for encoding)")
        logger.info(f"   Strength: {strength}x")
        logger.info(f"   Stabilize: {stabilize}")
        
        layer = self.layers[layer_idx]
        target = getattr(layer, 'mixer', getattr(layer, 'ssm', layer))
        
        def hook(module, input, output):
            if isinstance(output, tuple):
                hidden = output[0]
                rest = output[1:]
            else:
                hidden = output
                rest = ()
            
            h_mod = hidden.clone()
            
            if stabilize:
                # Reduce variance to prevent information loss
                std = h_mod.std(dim=-1, keepdim=True)
                mean = h_mod.mean(dim=-1, keepdim=True)
                
                # Normalize but keep information
                h_mod = mean + (h_mod - mean) / (std + 1e-5) * std * 0.8
            
            # Gentle amplification
            h_mod = h_mod * strength
            
            if rest:
                return (h_mod,) + rest
            return h_mod
        
        h = target.register_forward_hook(hook)
        self.hooks.append(h)
    
    def apply_anti_decay_steering(
        self,
        layers: List[int] = [5, 8, 12],  # Multiple early layers
        decay_prevention: float = 1.3
    ):
        """
        APPROACH 2: Anti-Decay (Multi-Layer)
        Combat HiPPO's aggressive forgetting by boosting early layers.
        
        Logic: If early layers preserve more, later layers have more to work with.
        """
        logger.info(f"\n🛡️ ANTI-DECAY STEERING (Multi-Layer)")
        logger.info(f"   Layers: {layers}")
        logger.info(f"   Strategy: Preserve information in early encoding")
        
        for layer_idx in layers:
            if layer_idx >= len(self.layers):
                continue
            
            logger.info(f"      Layer {layer_idx}: Prevent decay @ {decay_prevention}x")
            
            layer = self.layers[layer_idx]
            target = getattr(layer, 'mixer', getattr(layer, 'ssm', layer))
            
            def make_hook(strength):
                def hook(module, input, output):
                    if isinstance(output, tuple):
                        hidden = output[0]
                        rest = output[1:]
                    else:
                        hidden = output
                        rest = ()
                    
                    # Amplify all dimensions uniformly (prevent selective decay)
                    h_mod = hidden * strength
                    
                    if rest:
                        return (h_mod,) + rest
                    return h_mod
                return hook
            
            h = target.register_forward_hook(make_hook(decay_prevention))
            self.hooks.append(h)
    
    def apply_associative_recall_boost(
        self,
        early_layer: int = 6,
        late_layer: int = 18,
        early_strength: float = 2.0,
        late_strength: float = 3.0
    ):
        """
        APPROACH 3: Associative Recall Enhancement
        
        Strategy:
        - Early layer: Encode associations (key-value pairs)
        - Late layer: Retrieve associations (recall)
        
        This mimics how Transformers do in-context learning.
        """
        logger.info(f"\n🔗 ASSOCIATIVE RECALL BOOST")
        logger.info(f"   Early (encode): Layer {early_layer} @ {early_strength}x")
        logger.info(f"   Late (retrieve): Layer {late_layer} @ {late_strength}x")
        
        # Early layer: encoding boost
        if early_layer < len(self.layers):
            layer = self.layers[early_layer]
            target = getattr(layer, 'mixer', getattr(layer, 'ssm', layer))
            
            def early_hook(module, input, output):
                if isinstance(output, tuple):
                    hidden = output[0]
                    rest = output[1:]
                else:
                    hidden = output
                    rest = ()
                
                # Boost signal strength for encoding
                h_mod = hidden * early_strength
                
                if rest:
                    return (h_mod,) + rest
                return h_mod
            
            h = target.register_forward_hook(early_hook)
            self.hooks.append(h)
        
        # Late layer: retrieval boost with Cluster 9
        if late_layer < len(self.layers):
            layer = self.layers[late_layer]
            target = getattr(layer, 'mixer', getattr(layer, 'ssm', layer))
            
            def late_hook(module, input, output):
                if isinstance(output, tuple):
                    hidden = output[0]
                    rest = output[1:]
                else:
                    hidden = output
                    rest = ()
                
                h_mod = hidden.clone()
                
                # Strong boost on Cluster 9 for output
                for idx in self.cluster9_indices:
                    if idx < h_mod.shape[-1]:
                        h_mod[..., idx] *= late_strength
                
                if rest:
                    return (h_mod,) + rest
                return h_mod
            
            h = target.register_forward_hook(late_hook)
            self.hooks.append(h)
    
    def apply_cascade_memory_steering(
        self,
        layers: List[int] = [4, 8, 12, 16, 20],
        strengths: List[float] = [1.5, 1.8, 2.0, 2.5, 3.0]
    ):
        """
        APPROACH 4: Cascade Memory
        Progressive amplification: weak early, strong late.
        
        Logic: 
        - Early: gentle encoding preservation
        - Mid: stronger integration
        - Late: aggressive retrieval (with Cluster 9)
        """
        logger.info(f"\n🌊 CASCADE MEMORY STEERING")
        logger.info(f"   Layers: {layers}")
        logger.info(f"   Strengths: {strengths}")
        logger.info(f"   Strategy: Gradual amplification through network")
        
        for layer_idx, strength in zip(layers, strengths):
            if layer_idx >= len(self.layers):
                continue
            
            is_late = layer_idx >= 18
            logger.info(f"      Layer {layer_idx}: {strength}x {'+ Cluster 9' if is_late else ''}")
            
            layer = self.layers[layer_idx]
            target = getattr(layer, 'mixer', getattr(layer, 'ssm', layer))
            
            def make_hook(s, use_cluster9):
                def hook(module, input, output):
                    if isinstance(output, tuple):
                        hidden = output[0]
                        rest = output[1:]
                    else:
                        hidden = output
                        rest = ()
                    
                    h_mod = hidden.clone()
                    
                    if use_cluster9:
                        # Late layers: targeted boost
                        for idx in self.cluster9_indices:
                            if idx < h_mod.shape[-1]:
                                h_mod[..., idx] *= s
                    else:
                        # Early/mid layers: uniform boost
                        h_mod = h_mod * s
                    
                    if rest:
                        return (h_mod,) + rest
                    return h_mod
                return hook
            
            h = target.register_forward_hook(make_hook(strength, is_late))
            self.hooks.append(h)
    
    def apply_minimal_interference_steering(
        self,
        layer_idx: int = 20,
        strength: float = 4.0
    ):
        """
        APPROACH 5: Minimal Interference
        Only boost Cluster 9 at the output layer.
        
        Rationale: Don't disrupt SSM dynamics, just enhance final output.
        """
        if layer_idx >= len(self.layers):
            logger.warning(f"Layer {layer_idx} doesn't exist")
            return
        
        logger.info(f"\n🎯 MINIMAL INTERFERENCE (Output Only)")
        logger.info(f"   Layer: {layer_idx}")
        logger.info(f"   Strength: {strength}x on Cluster 9 only")
        
        layer = self.layers[layer_idx]
        target = getattr(layer, 'mixer', getattr(layer, 'ssm', layer))
        
        def hook(module, input, output):
            if isinstance(output, tuple):
                hidden = output[0]
                rest = output[1:]
            else:
                hidden = output
                rest = ()
            
            h_mod = hidden.clone()
            
            # Only Cluster 9
            for idx in self.cluster9_indices:
                if idx < h_mod.shape[-1]:
                    h_mod[..., idx] *= strength
            
            if rest:
                return (h_mod,) + rest
            return h_mod
        
        h = target.register_forward_hook(hook)
        self.hooks.append(h)
    
    def remove_steering(self):
        """Remove all hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


def run_memory_focused_experiment(
    trained_model_path: str = None,
    query_datasets: List[str] = ['squad']
):
    """
    Test memory-specific steering approaches.
    Focus on the tasks that are hardest: long-context recall, associative recall.
    
    Now uses:
    - Trained model (trained on The Pile) if provided, else pretrained
    - Query datasets (SQuAD, Natural Questions, etc.) for testing
    
    Args:
        trained_model_path: Path to model trained on The Pile (optional)
        query_datasets: List of query datasets to use for testing
    """
    from mamba_model_loader import load_mamba_model_and_tokenizer
    from targeted_approach_2 import EnhancedRealisticEvaluation
    from query_dataset_loader import get_query_tasks_for_evaluation
    
    logger.info("="*80)
    logger.info("🧠 MEMORY-FOCUSED STEERING EXPERIMENT")
    logger.info("Targeting Mamba's known weakness: in-context learning & recall")
    if trained_model_path:
        logger.info(f"Using model trained on The Pile: {trained_model_path}")
    logger.info(f"Testing on query datasets: {query_datasets}")
    logger.info("="*80)
    
    # Load model (trained or pretrained)
    model_name = trained_model_path if trained_model_path else "state-spaces/mamba-130m-hf"
    logger.info(f"\n📦 Loading model: {model_name}")
    model, tokenizer = load_mamba_model_and_tokenizer(
        model_name=model_name,
        device="cuda" if torch.cuda.is_available() else "cpu",
        use_mamba_class=True,
        fallback_to_auto=True
    )
    
    device = next(model.parameters()).device
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Initialize
    evaluator = EnhancedRealisticEvaluation(tokenizer, device)
    steering = MemoryEnhancedSteering(model)
    
    # Get original test cases
    test_cases = evaluator.get_enhanced_test_cases()
    
    # Load query datasets for testing
    logger.info("\n📚 Loading query datasets for testing...")
    try:
        query_tasks = get_query_tasks_for_evaluation(
            datasets=query_datasets,
            num_per_dataset=30
        )
        logger.info(f"✅ Loaded {sum(len(v) for v in query_tasks.values())} queries from test datasets")
        
        # Add query tasks to test cases
        test_cases.update(query_tasks)
    except Exception as e:
        logger.warning(f"⚠️ Could not load query datasets: {e}")
        logger.info("   Continuing with original test cases only")
        query_tasks = {}
    
    # CRITICAL: Focus only on memory tasks
    memory_tasks = {
        'long_context_recall': test_cases['long_context_recall'],
        'associative_recall': test_cases['associative_recall'],
    }
    
    # Add query tasks to memory tasks
    if query_tasks:
        memory_tasks.update(query_tasks)
    
    # Also test on instruction tasks to ensure we don't break them
    instruction_tasks = {
        'simple_instructions': test_cases['simple_instructions'],
    }
    
    all_results = {}
    
    # ================================================
    # BASELINE
    # ================================================
    logger.info("\n" + "="*80)
    logger.info("📊 BASELINE")
    logger.info("="*80)
    
    # Initialize I/O recording for baseline
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    baseline_io_records = []
    
    baseline_memory = evaluator.test_model_enhanced(
        model, memory_tasks, "BASELINE: Memory"
    )
    baseline_instruction = evaluator.test_model_enhanced(
        model, instruction_tasks, "BASELINE: Instructions"
    )
    
    # Record baseline I/O from results
    for task_name, result in {**baseline_memory, **baseline_instruction}.items():
        for i, resp in enumerate(result.get('responses', [])):
            # Get full prompt from test cases
            full_prompt = None
            if task_name in memory_tasks:
                cases = memory_tasks[task_name]
            elif task_name in instruction_tasks:
                cases = instruction_tasks[task_name]
            else:
                cases = []
            
            if i < len(cases):
                full_prompt = cases[i].get('prompt', '')
            
            baseline_io_records.append({
                'method': 'baseline',
                'task_type': task_name,
                'case_index': i + 1,
                'input_prompt': full_prompt or resp.get('prompt_short', ''),
                'expected_output': resp.get('expected', ''),
                'actual_output': resp.get('response', ''),
                'is_correct': resp.get('correct', False),
                'task': resp.get('task', 'unknown')
            })
    
    base_mem_acc = sum(r['correct'] for r in baseline_memory.values()) / \
                   sum(r['total'] for r in baseline_memory.values())
    base_inst_acc = sum(r['correct'] for r in baseline_instruction.values()) / \
                    sum(r['total'] for r in baseline_instruction.values())
    
    logger.info(f"\n📊 Baseline:")
    logger.info(f"   Memory: {base_mem_acc*100:.1f}%")
    logger.info(f"   Instructions: {base_inst_acc*100:.1f}%")
    
    all_results['baseline'] = {'memory': base_mem_acc, 'instruction': base_inst_acc}
    
    # ================================================
    # TEST 1: Memory Retention (Early Layer)
    # ================================================
    logger.info("\n" + "="*80)
    logger.info("🧪 TEST 1: Memory Retention")
    logger.info("="*80)
    
    steering.apply_memory_retention_steering(layer_idx=8, strength=1.5, stabilize=True)
    
    test1_mem = evaluator.test_model_enhanced(model, memory_tasks, "Memory Retention: Memory")
    test1_inst = evaluator.test_model_enhanced(model, instruction_tasks, "Memory Retention: Instructions")
    
    steering.remove_steering()
    
    t1_mem = sum(r['correct'] for r in test1_mem.values()) / sum(r['total'] for r in test1_mem.values())
    t1_inst = sum(r['correct'] for r in test1_inst.values()) / sum(r['total'] for r in test1_inst.values())
    
    logger.info(f"\n📊 Results:")
    logger.info(f"   Memory: {t1_mem*100:.1f}% [{(t1_mem-base_mem_acc)*100:+.1f}%]")
    logger.info(f"   Instructions: {t1_inst*100:.1f}% [{(t1_inst-base_inst_acc)*100:+.1f}%]")
    
    all_results['memory_retention'] = {
        'memory': t1_mem, 'instruction': t1_inst,
        'memory_improvement': t1_mem - base_mem_acc,
        'instruction_improvement': t1_inst - base_inst_acc
    }
    
    # ================================================
    # TEST 2: Anti-Decay (Multiple Early Layers)
    # ================================================
    logger.info("\n" + "="*80)
    logger.info("🧪 TEST 2: Anti-Decay")
    logger.info("="*80)
    
    steering.apply_anti_decay_steering(layers=[5, 8, 12], decay_prevention=1.3)
    
    test2_mem = evaluator.test_model_enhanced(model, memory_tasks, "Anti-Decay: Memory")
    test2_inst = evaluator.test_model_enhanced(model, instruction_tasks, "Anti-Decay: Instructions")
    
    steering.remove_steering()
    
    t2_mem = sum(r['correct'] for r in test2_mem.values()) / sum(r['total'] for r in test2_mem.values())
    t2_inst = sum(r['correct'] for r in test2_inst.values()) / sum(r['total'] for r in test2_inst.values())
    
    logger.info(f"\n📊 Results:")
    logger.info(f"   Memory: {t2_mem*100:.1f}% [{(t2_mem-base_mem_acc)*100:+.1f}%]")
    logger.info(f"   Instructions: {t2_inst*100:.1f}% [{(t2_inst-base_inst_acc)*100:+.1f}%]")
    
    all_results['anti_decay'] = {
        'memory': t2_mem, 'instruction': t2_inst,
        'memory_improvement': t2_mem - base_mem_acc,
        'instruction_improvement': t2_inst - base_inst_acc
    }
    
    # ================================================
    # TEST 3: Associative Recall Boost
    # ================================================
    logger.info("\n" + "="*80)
    logger.info("🧪 TEST 3: Associative Recall Boost")
    logger.info("="*80)
    
    steering.apply_associative_recall_boost(
        early_layer=6, late_layer=18,
        early_strength=2.0, late_strength=3.0
    )
    
    test3_mem = evaluator.test_model_enhanced(model, memory_tasks, "Associative: Memory")
    test3_inst = evaluator.test_model_enhanced(model, instruction_tasks, "Associative: Instructions")
    
    steering.remove_steering()
    
    t3_mem = sum(r['correct'] for r in test3_mem.values()) / sum(r['total'] for r in test3_mem.values())
    t3_inst = sum(r['correct'] for r in test3_inst.values()) / sum(r['total'] for r in test3_inst.values())
    
    logger.info(f"\n📊 Results:")
    logger.info(f"   Memory: {t3_mem*100:.1f}% [{(t3_mem-base_mem_acc)*100:+.1f}%]")
    logger.info(f"   Instructions: {t3_inst*100:.1f}% [{(t3_inst-base_inst_acc)*100:+.1f}%]")
    
    all_results['associative'] = {
        'memory': t3_mem, 'instruction': t3_inst,
        'memory_improvement': t3_mem - base_mem_acc,
        'instruction_improvement': t3_inst - base_inst_acc
    }
    
    # ================================================
    # TEST 4: Cascade Memory
    # ================================================
    logger.info("\n" + "="*80)
    logger.info("🧪 TEST 4: Cascade Memory")
    logger.info("="*80)
    
    steering.apply_cascade_memory_steering(
        layers=[4, 8, 12, 16, 20],
        strengths=[1.5, 1.8, 2.0, 2.5, 3.0]
    )
    
    test4_mem = evaluator.test_model_enhanced(model, memory_tasks, "Cascade: Memory")
    test4_inst = evaluator.test_model_enhanced(model, instruction_tasks, "Cascade: Instructions")
    
    steering.remove_steering()
    
    t4_mem = sum(r['correct'] for r in test4_mem.values()) / sum(r['total'] for r in test4_mem.values())
    t4_inst = sum(r['correct'] for r in test4_inst.values()) / sum(r['total'] for r in test4_inst.values())
    
    logger.info(f"\n📊 Results:")
    logger.info(f"   Memory: {t4_mem*100:.1f}% [{(t4_mem-base_mem_acc)*100:+.1f}%]")
    logger.info(f"   Instructions: {t4_inst*100:.1f}% [{(t4_inst-base_inst_acc)*100:+.1f}%]")
    
    all_results['cascade'] = {
        'memory': t4_mem, 'instruction': t4_inst,
        'memory_improvement': t4_mem - base_mem_acc,
        'instruction_improvement': t4_inst - base_inst_acc
    }
    
    # ================================================
    # TEST 5: Minimal Interference (Control)
    # ================================================
    logger.info("\n" + "="*80)
    logger.info("🧪 TEST 5: Minimal Interference (Control)")
    logger.info("="*80)
    
    steering.apply_minimal_interference_steering(layer_idx=20, strength=4.0)
    
    # Initialize I/O recording for minimal steering
    minimal_io_records = []
    
    test5_mem = evaluator.test_model_enhanced(model, memory_tasks, "Minimal: Memory")
    test5_inst = evaluator.test_model_enhanced(model, instruction_tasks, "Minimal: Instructions")
    
    steering.remove_steering()
    
    # Record minimal steering I/O from results
    for task_name, result in {**test5_mem, **test5_inst}.items():
        for i, resp in enumerate(result.get('responses', [])):
            # Get full prompt from test cases
            full_prompt = None
            if task_name in memory_tasks:
                cases = memory_tasks[task_name]
            elif task_name in instruction_tasks:
                cases = instruction_tasks[task_name]
            else:
                cases = []
            
            if i < len(cases):
                full_prompt = cases[i].get('prompt', '')
            
            minimal_io_records.append({
                'method': 'minimal_steering',
                'task_type': task_name,
                'case_index': i + 1,
                'input_prompt': resp.get('prompt_full', full_prompt or resp.get('prompt_short', '')),
                'expected_output': resp.get('expected', ''),
                'actual_output': resp.get('response_full', resp.get('response', '')),
                'is_correct': resp.get('correct', False),
                'task': resp.get('task', 'unknown')
            })
    
    t5_mem = sum(r['correct'] for r in test5_mem.values()) / sum(r['total'] for r in test5_mem.values())
    t5_inst = sum(r['correct'] for r in test5_inst.values()) / sum(r['total'] for r in test5_inst.values())
    
    logger.info(f"\n📊 Results:")
    logger.info(f"   Memory: {t5_mem*100:.1f}% [{(t5_mem-base_mem_acc)*100:+.1f}%]")
    logger.info(f"   Instructions: {t5_inst*100:.1f}% [{(t5_inst-base_inst_acc)*100:+.1f}%]")
    
    all_results['minimal'] = {
        'memory': t5_mem, 'instruction': t5_inst,
        'memory_improvement': t5_mem - base_mem_acc,
        'instruction_improvement': t5_inst - base_inst_acc
    }
    
    # Create combined I/O file with baseline and minimal steering
    logger.info("\n📝 Creating combined I/O record file...")
    combined_io_file = f"experiment_logs/io_minimal_combined_{timestamp}.json"
    
    # Combine records by matching cases
    combined_records = []
    baseline_dict = {r['task_type'] + '_' + str(r['case_index']): r for r in baseline_io_records}
    minimal_dict = {r['task_type'] + '_' + str(r['case_index']): r for r in minimal_io_records}
    
    for key in baseline_dict:
        baseline_record = baseline_dict[key]
        minimal_record = minimal_dict.get(key, {})
        
        combined_records.append({
            'task_type': baseline_record['task_type'],
            'case_index': baseline_record['case_index'],
            'task': baseline_record['task'],
            'input_prompt': baseline_record['input_prompt'],
            'expected_output': baseline_record['expected_output'],
            'baseline': {
                'output': baseline_record['actual_output'],
                'is_correct': baseline_record['is_correct']
            },
            'minimal_steering': {
                'output': minimal_record.get('actual_output', ''),
                'is_correct': minimal_record.get('is_correct', False)
            } if minimal_record else {}
        })
    
    Path(combined_io_file).parent.mkdir(parents=True, exist_ok=True)
    with open(combined_io_file, 'w') as f:
        json.dump({
            'experiment': 'memory_focused_minimal_steering',
            'timestamp': timestamp,
            'total_cases': len(combined_records),
            'records': combined_records
        }, f, indent=2)
    
    logger.info(f"💾 Combined I/O records saved to: {combined_io_file}")
    
    # ================================================
    # FINAL COMPARISON
    # ================================================
    logger.info("\n" + "="*80)
    logger.info("🏆 FINAL RESULTS")
    logger.info("="*80)
    
    logger.info("\n📊 MEMORY TASKS (The hard ones!):")
    if query_tasks:
        logger.info(f"   Includes queries from: {', '.join(query_tasks.keys())}")
    logger.info("-" * 70)
    logger.info(f"{'Method':<30} {'Accuracy':<12} {'Change':<12} {'Status'}")
    logger.info("-" * 70)
    
    best_mem = -999
    best_mem_method = None
    
    for method, results in all_results.items():
        if method == 'baseline':
            logger.info(f"{'Baseline':<30} {results['memory']*100:6.1f}%      {'---'}")
        else:
            imp = results['memory_improvement']
            if imp > best_mem:
                best_mem = imp
                best_mem_method = method
            
            status = "✅" if imp > 0.10 else "📈" if imp > 0.05 else "📊" if imp > 0 else "❌"
            logger.info(f"{method.replace('_', ' ').title():<30} "
                       f"{results['memory']*100:6.1f}%      "
                       f"{imp*100:+5.1f}%      {status}")
    
    logger.info("\n📊 INSTRUCTION TASKS (Make sure we don't break these):")
    logger.info("-" * 70)
    
    for method, results in all_results.items():
        if method == 'baseline':
            logger.info(f"{'Baseline':<30} {results['instruction']*100:6.1f}%      {'---'}")
        else:
            imp = results['instruction_improvement']
            status = "✅" if imp > 0.10 else "📊" if imp >= 0 else "⚠️"
            logger.info(f"{method.replace('_', ' ').title():<30} "
                       f"{results['instruction']*100:6.1f}%      "
                       f"{imp*100:+5.1f}%      {status}")
    
    # ================================================
    # RECOMMENDATIONS
    # ================================================
    logger.info("\n" + "="*80)
    logger.info("💡 KEY FINDINGS")
    logger.info("="*80)
    
    logger.info(f"\n🎯 BEST FOR MEMORY: {best_mem_method.upper()}")
    logger.info(f"   Improvement: {best_mem*100:+.1f}%")
    
    if best_mem > 0.15:
        logger.info(f"   🎉 BREAKTHROUGH! Significant memory improvement!")
    elif best_mem > 0.10:
        logger.info(f"   ✅ Strong improvement! This approach works.")
    elif best_mem > 0.05:
        logger.info(f"   📈 Moderate improvement, worth exploring further.")
    elif best_mem > 0:
        logger.info(f"   📊 Modest gains, may need parameter tuning.")
    else:
        logger.info(f"   ⚠️ No improvement yet. Consider:")
        logger.info(f"      • Different layer combinations")
        logger.info(f"      • Stronger/weaker strengths")
        logger.info(f"      • Task-specific prompting")
    
    # Save
    output_path = Path("experiment_logs/memory_focused_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump({
            'experiment': 'memory_focused_steering',
            'model': trained_model_path if trained_model_path else 'state-spaces/mamba-130m-hf',
            'trained_on_pile': trained_model_path is not None,
            'focus': 'long_context_recall_and_associative_memory',
            'test_datasets': query_datasets,
            'query_tasks_included': len(query_tasks) > 0,
            'results': {k: {
                'memory_accuracy': float(v['memory']),
                'instruction_accuracy': float(v['instruction']),
                'memory_improvement': float(v.get('memory_improvement', 0)),
                'instruction_improvement': float(v.get('instruction_improvement', 0))
            } for k, v in all_results.items()},
            'best_method': best_mem_method,
            'best_improvement': float(best_mem)
        }, f, indent=2)
    
    logger.info(f"\n💾 Results saved to: {output_path}")
    
    # Also save baseline I/O records separately if they exist
    if 'baseline_io_records' in locals() and baseline_io_records:
        baseline_io_file = f"experiment_logs/io_baseline_minimal_{timestamp}.json"
        Path(baseline_io_file).parent.mkdir(parents=True, exist_ok=True)
        with open(baseline_io_file, 'w') as f:
            json.dump({
                'experiment': 'baseline_only',
                'timestamp': timestamp,
                'total_cases': len(baseline_io_records),
                'records': baseline_io_records
            }, f, indent=2)
        logger.info(f"💾 Baseline I/O records saved to: {baseline_io_file}")
    
    logger.info("="*80)
    
    return all_results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Memory-focused steering experiment")
    parser.add_argument("--trained_model", type=str, default=None,
                       help="Path to model trained on The Pile")
    parser.add_argument("--query_datasets", type=str, nargs='+', 
                       default=['squad'],
                       help="Query datasets to test on (squad, natural_questions, triviaqa)")
    
    args = parser.parse_args()
    
    results = run_memory_focused_experiment(
        trained_model_path=args.trained_model,
        query_datasets=args.query_datasets
    )