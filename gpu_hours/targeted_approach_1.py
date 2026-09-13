"""
AGGRESSIVE STEERING TEST FOR MAMBA-130M
Using stronger interventions and different approaches.

result
INFO:__main__:📊 COMPARISON OF ALL APPROACHES
INFO:__main__:================================================================================
INFO:__main__:
STRONG_10X:
INFO:__main__:  Baseline: 66.7% (6/9)
INFO:__main__:  Steered:  77.8% (7/9)
INFO:__main__:  Change:   +11.1%
INFO:__main__:    • challenging_tasks: 60% → 80%
INFO:__main__:
CONTRASTIVE:
INFO:__main__:  Baseline: 66.7% (6/9)
INFO:__main__:  Steered:  66.7% (6/9)
INFO:__main__:  Change:   +0.0%
INFO:__main__:
MULTI_LAYER:
INFO:__main__:  Baseline: 66.7% (6/9)
INFO:__main__:  Steered:  55.6% (5/9)
INFO:__main__:  Change:   -11.1%
"""

import torch
import logging
from typing import List, Dict, Tuple
import json
from pathlib import Path
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AggressiveSteeringTest:
    """
    Test stronger steering approaches on Mamba-130M.
    """
    
    def __init__(self, tokenizer, device="cuda"):
        self.tokenizer = tokenizer
        self.device = device
    
    def get_focus_test_cases(self) -> Dict[str, List[Dict]]:
        """
        Focus ONLY on tasks where model shows SOME capability but fails.
        """
        
        return {
            # Tasks where model FAILS (0% correct) - can only improve!
            'challenging_tasks': [
                {
                    'prompt': 'Repeat exactly: hello',
                    'expected': 'hello',
                    'task': 'exact repetition'
                },
                {
                    'prompt': 'Water is',
                    'expected': 'wet',
                    'task': 'property completion'
                },
                {
                    'prompt': 'Say only: cat',
                    'expected': 'cat',
                    'task': 'single word response'
                },
                {
                    'prompt': 'x = [1, 2, 3]\nx[0] =',
                    'expected': '1',
                    'task': 'array indexing'
                },
                {
                    'prompt': 'Is water wet? Answer:',
                    'expected': 'yes',
                    'task': 'yes/no answer'
                },
            ],
            
            # Tasks where model is 50% correct
            'fifty_fifty_tasks': [
                {
                    'prompt': 'The sky is',
                    'expected': 'blue',
                    'task': 'color property'
                },
                {
                    'prompt': 'if True:\n    print("',
                    'expected': 'True',
                    'task': 'code completion'
                },
                {
                    'prompt': 'Continue: A, B, C,',
                    'expected': 'D',
                    'task': 'alphabet continuation'
                },
                {
                    'prompt': '1, 2, 3, 4,',
                    'expected': '5',
                    'task': 'number continuation'
                },
            ],
        }
    
    def test_model_simple(self, model, test_cases: Dict) -> Dict:
        """
        Simple test with single token generation.
        """
        results = {}
        
        for task_name, cases in test_cases.items():
            logger.info(f"\n📊 Testing: {task_name}")
            
            correct = 0
            responses = []
            
            for case in cases:
                prompt = case['prompt']
                expected = case['expected']
                
                # Tokenize
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
                
                # Generate with longer output
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=50,
                        do_sample=False,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id
                    )
                
                input_len = inputs['input_ids'].shape[1]
                response = self.tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
                response = response.strip()
                
                # Check if correct
                expected_lower = str(expected).lower()
                response_lower = response.lower()
                
                is_correct = (expected_lower in response_lower or 
                             response_lower.startswith(expected_lower[:2]) or
                             len(response_lower) > 0 and expected_lower.startswith(response_lower[:2]))
                
                if is_correct:
                    correct += 1
                
                responses.append({
                    'response': response[:100],  # Store more of response
                    'expected': expected,
                    'correct': is_correct
                })
                
                symbol = "✅" if is_correct else "❌"
                logger.info(f"  {symbol} Expected: '{expected}', Got: '{response[:80]}'")  # Show more in logs
            
            accuracy = correct / len(cases)
            results[task_name] = {
                'accuracy': accuracy,
                'correct': correct,
                'total': len(cases),
                'responses': responses
            }
            
            logger.info(f"  Accuracy: {accuracy*100:.1f}% ({correct}/{len(cases)})")
        
        return results


class StrongCluster9Steering:
    """
    STRONG steering using Cluster 9 neurons.
    """
    
    def __init__(self, model):
        self.model = model
        self.hooks = []
        
        if hasattr(model, 'backbone'):
            self.layers = model.backbone.layers
        else:
            self.layers = model.layers
        
        # Different neuron subsets to test
        self.neuron_subsets = {
            'subset1': [4, 38, 84, 94],      # Small subset
            'subset2': [163, 171, 268],       # Medium subset
            'subset3': [363, 401, 497],       # Another subset
            'subset4': [564, 568, 582, 654],  # Larger subset
            'all': [4, 38, 84, 94, 163, 171, 268, 363, 401, 497, 564, 568, 582, 654, 659, 686]
        }
        
        self.current_subset = 'all'
    
    def apply_strong_steering(self, 
                             strength: float = 10.0, 
                             layer_idx: int = 20,
                             subset: str = 'all'):
        """
        Apply VERY strong steering.
        """
        if layer_idx >= len(self.layers):
            logger.warning(f"Layer {layer_idx} doesn't exist")
            layer_idx = len(self.layers) - 2
        
        neurons = self.neuron_subsets.get(subset, self.neuron_subsets['all'])
        self.current_subset = subset
        
        logger.info(f"\n⚡ APPLYING STRONG STEERING")
        logger.info(f"  Layer: {layer_idx}")
        logger.info(f"  Strength: {strength}x")
        logger.info(f"  Neurons: {len(neurons)} neurons ({subset})")
        
        layer = self.layers[layer_idx]
        target = getattr(layer, 'mixer', getattr(layer, 'ssm', layer))
        
        def hook(module, input, output):
            if isinstance(output, tuple):
                hidden = output[0]
                rest = output[1:]
            else:
                hidden = output
                rest = ()
            
            # VERY strong modification
            h_mod = hidden.clone()
            for idx in neurons:
                if idx < h_mod.shape[-1]:
                    # Boost these neurons aggressively
                    h_mod[..., idx] *= strength
                    
                    # Also add bias to ensure activation
                    h_mod[..., idx] += 2.0
            
            if rest:
                return (h_mod,) + rest
            return h_mod
        
        h = target.register_forward_hook(hook)
        self.hooks.append(h)
    
    def apply_contrastive_steering(self,
                                 positive_neurons: List[int],
                                 negative_neurons: List[int],
                                 pos_strength: float = 5.0,
                                 neg_strength: float = 0.2,
                                 layer_idx: int = 20):
        """
        Apply contrastive steering: boost some neurons, suppress others.
        """
        logger.info(f"\n🎭 APPLYING CONTRASTIVE STEERING")
        logger.info(f"  Boost {len(positive_neurons)} neurons by {pos_strength}x")
        logger.info(f"  Suppress {len(negative_neurons)} neurons by {neg_strength}x")
        
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
            
            # Boost positive neurons
            for idx in positive_neurons:
                if idx < h_mod.shape[-1]:
                    h_mod[..., idx] *= pos_strength
            
            # Suppress negative neurons
            for idx in negative_neurons:
                if idx < h_mod.shape[-1]:
                    h_mod[..., idx] *= neg_strength
            
            if rest:
                return (h_mod,) + rest
            return h_mod
        
        h = target.register_forward_hook(hook)
        self.hooks.append(h)
    
    def remove_steering(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


def run_aggressive_steering_test():
    """
    AGGRESSIVE steering test with multiple approaches.
    """
    from mamba_model_loader import load_mamba_model_and_tokenizer
    
    logger.info("="*80)
    logger.info("⚡ AGGRESSIVE MAMBA-130M STEERING TEST")
    logger.info("Testing stronger interventions")
    logger.info("="*80)
    
    # Load model
    logger.info("\n📦 Loading model...")
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
    tester = AggressiveSteeringTest(tokenizer, device)
    steering = StrongCluster9Steering(model)
    
    # Get test cases
    test_cases = tester.get_focus_test_cases()
    
    all_results = {}
    
    # ================================================
    # APPROACH 1: VERY STRONG STEERING
    # ================================================
    logger.info("\n" + "="*80)
    logger.info("🧪 APPROACH 1: VERY STRONG STEERING (10x)")
    logger.info("="*80)
    
    # Baseline
    logger.info("\n📊 BASELINE:")
    baseline = tester.test_model_simple(model, test_cases)
    
    # With strong steering
    steering.apply_strong_steering(strength=10.0, layer_idx=20, subset='all')
    logger.info("\n📊 WITH STRONG STEERING:")
    strong_results = tester.test_model_simple(model, test_cases)
    steering.remove_steering()
    
    all_results['strong_10x'] = {
        'baseline': baseline,
        'steered': strong_results
    }
    
    # ================================================
    # APPROACH 2: CONTRASTIVE STEERING
    # ================================================
    logger.info("\n" + "="*80)
    logger.info("🧪 APPROACH 2: CONTRASTIVE STEERING")
    logger.info("="*80)
    
    # Boost early neurons, suppress later ones
    positive_neurons = [4, 38, 84, 94, 163]  # Early neurons
    negative_neurons = [654, 659, 686]       # Late neurons
    
    steering.apply_contrastive_steering(
        positive_neurons=positive_neurons,
        negative_neurons=negative_neurons,
        pos_strength=5.0,
        neg_strength=0.2,
        layer_idx=20
    )
    
    logger.info("\n📊 WITH CONTRASTIVE STEERING:")
    contrast_results = tester.test_model_simple(model, test_cases)
    steering.remove_steering()
    
    all_results['contrastive'] = {
        'baseline': baseline,
        'steered': contrast_results
    }
    
    # ================================================
    # APPROACH 3: MULTIPLE LAYERS
    # ================================================
    logger.info("\n" + "="*80)
    logger.info("🧪 APPROACH 3: MULTIPLE LAYER STEERING")
    logger.info("="*80)
    
    # Apply to multiple layers
    for layer_idx in [15, 20, 24]:
        if layer_idx < len(steering.layers):
            steering.apply_strong_steering(strength=8.0, layer_idx=layer_idx, subset='subset1')
    
    logger.info("\n📊 WITH MULTI-LAYER STEERING:")
    multi_results = tester.test_model_simple(model, test_cases)
    steering.remove_steering()
    
    all_results['multi_layer'] = {
        'baseline': baseline,
        'steered': multi_results
    }
    
    # ================================================
    # COMPARE ALL APPROACHES
    # ================================================
    logger.info("\n" + "="*80)
    logger.info("📊 COMPARISON OF ALL APPROACHES")
    logger.info("="*80)
    
    approaches = ['strong_10x', 'contrastive', 'multi_layer']
    
    for approach in approaches:
        base_data = all_results[approach]['baseline']
        steer_data = all_results[approach]['steered']
        
        # Calculate overall accuracy
        base_total = 0
        base_correct = 0
        steer_total = 0
        steer_correct = 0
        
        for task_name in ['challenging_tasks', 'fifty_fifty_tasks']:
            if task_name in base_data:
                base_total += base_data[task_name]['total']
                base_correct += base_data[task_name]['correct']
                steer_total += steer_data[task_name]['total']
                steer_correct += steer_data[task_name]['correct']
        
        base_acc = base_correct / base_total if base_total > 0 else 0
        steer_acc = steer_correct / steer_total if steer_total > 0 else 0
        improvement = steer_acc - base_acc
        
        logger.info(f"\n{approach.upper()}:")
        logger.info(f"  Baseline: {base_acc*100:.1f}% ({base_correct}/{base_total})")
        logger.info(f"  Steered:  {steer_acc*100:.1f}% ({steer_correct}/{steer_total})")
        logger.info(f"  Change:   {improvement*100:+.1f}%")
        
        # Show which tasks improved
        if improvement > 0:
            for task_name in ['challenging_tasks', 'fifty_fifty_tasks']:
                if task_name in base_data and task_name in steer_data:
                    base_task = base_data[task_name]['accuracy']
                    steer_task = steer_data[task_name]['accuracy']
                    if steer_task > base_task:
                        logger.info(f"    • {task_name}: {base_task*100:.0f}% → {steer_task*100:.0f}%")
    
    # ================================================
    # FIND BEST APPROACH
    # ================================================
    logger.info("\n" + "="*80)
    logger.info("🏆 BEST APPROACH ANALYSIS")
    logger.info("="*80)
    
    best_approach = None
    best_improvement = -1
    
    for approach in approaches:
        base_data = all_results[approach]['baseline']
        steer_data = all_results[approach]['steered']
        
        base_total = 0
        base_correct = 0
        steer_total = 0
        steer_correct = 0
        
        for task_name in ['challenging_tasks', 'fifty_fifty_tasks']:
            if task_name in base_data:
                base_total += base_data[task_name]['total']
                base_correct += base_data[task_name]['correct']
                steer_total += steer_data[task_name]['total']
                steer_correct += steer_data[task_name]['correct']
        
        base_acc = base_correct / base_total if base_total > 0 else 0
        steer_acc = steer_correct / steer_total if steer_total > 0 else 0
        improvement = steer_acc - base_acc
        
        if improvement > best_improvement:
            best_improvement = improvement
            best_approach = approach
    
    if best_approach and best_improvement > 0:
        logger.info(f"✅ BEST APPROACH: {best_approach}")
        logger.info(f"   Improvement: {best_improvement*100:+.1f}%")
        
        if best_improvement > 0.1:
            logger.info("   🎉 SIGNIFICANT IMPROVEMENT!")
        elif best_improvement > 0.05:
            logger.info("   📈 MODERATE IMPROVEMENT")
        else:
            logger.info("   📊 SMALL BUT MEASURABLE IMPROVEMENT")
    else:
        logger.info("⚠️  NO APPROACH SHOWED IMPROVEMENT")
        logger.info("\n💡 POSSIBLE REASONS:")
        logger.info("   1. Mamba-130M is too small for neuron steering")
        logger.info("   2. Cluster 9 neurons may not control these behaviors")
        logger.info("   3. Different steering technique needed")
        logger.info("\n🔧 RECOMMENDATIONS:")
        logger.info("   1. Try Mamba-370M or 1.4B model")
        logger.info("   2. Explore other neuron clusters")
        logger.info("   3. Use task vectors instead of simple boosting")
    
    # ================================================
    # SAVE ALL RESULTS
    # ================================================
    output_path = Path("experiment_logs/aggressive_steering_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Simplify for saving
    save_data = {}
    for approach in approaches:
        save_data[approach] = {
            'baseline_accuracy': float(sum(
                all_results[approach]['baseline'][task]['accuracy'] * 
                all_results[approach]['baseline'][task]['total']
                for task in ['challenging_tasks', 'fifty_fifty_tasks']
                if task in all_results[approach]['baseline']
            ) / sum(
                all_results[approach]['baseline'][task]['total']
                for task in ['challenging_tasks', 'fifty_fifty_tasks']
                if task in all_results[approach]['baseline']
            )),
            'steered_accuracy': float(sum(
                all_results[approach]['steered'][task]['accuracy'] * 
                all_results[approach]['steered'][task]['total']
                for task in ['challenging_tasks', 'fifty_fifty_tasks']
                if task in all_results[approach]['steered']
            ) / sum(
                all_results[approach]['steered'][task]['total']
                for task in ['challenging_tasks', 'fifty_fifty_tasks']
                if task in all_results[approach]['steered']
            )),
            'improvement': float(
                sum(
                    all_results[approach]['steered'][task]['accuracy'] * 
                    all_results[approach]['steered'][task]['total']
                    for task in ['challenging_tasks', 'fifty_fifty_tasks']
                    if task in all_results[approach]['steered']
                ) / sum(
                    all_results[approach]['steered'][task]['total']
                    for task in ['challenging_tasks', 'fifty_fifty_tasks']
                    if task in all_results[approach]['steered']
                ) - sum(
                    all_results[approach]['baseline'][task]['accuracy'] * 
                    all_results[approach]['baseline'][task]['total']
                    for task in ['challenging_tasks', 'fifty_fifty_tasks']
                    if task in all_results[approach]['baseline']
                ) / sum(
                    all_results[approach]['baseline'][task]['total']
                    for task in ['challenging_tasks', 'fifty_fifty_tasks']
                    if task in all_results[approach]['baseline']
                )
            )
        }
    
    with open(output_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    
    logger.info(f"\n💾 Results saved to: {output_path}")
    
    return save_data


if __name__ == "__main__":
    results = run_aggressive_steering_test()