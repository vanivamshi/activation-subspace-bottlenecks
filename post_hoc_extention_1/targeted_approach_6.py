"""
COMPLEX REASONING & LONG-CONTEXT TEST FOR MAMBA-130M

Tests the limits of Mamba-130M with query-focus prompting:
1. Simple recall (proven: 100%)
2. Multi-hop reasoning (2-3 steps)
3. Long-context recall (5-10 facts)
4. Complex reasoning (logic + memory)

Goal: Find the capability ceiling of Mamba-130M

results
PERFORMANCE COMPARISON: BASELINE vs STEERING
================================================================================
   Includes queries from: squad_queries, triviaqa_queries
--------------------------------------------------------------------------------
Level                          Tasks    Baseline     With Steering   Change       Status
--------------------------------------------------------------------------------
Simple Recall                  3        100.0%        100.0%           +0.0%        ➖ NEUTRAL
Two-Hop Reasoning              4         75.0%         75.0%           +0.0%        ➖ NEUTRAL
Three-Hop Reasoning            3        100.0%         66.7%          -33.3%        ❌ NEGATIVE
Long Context (5-7 facts)       3         66.7%        100.0%          +33.3%        ✅ EXCELLENT
Combined Reasoning + Memory    3         66.7%         66.7%           +0.0%        ➖ NEUTRAL
Stress Test (10+ facts)        2        100.0%        100.0%           +0.0%        ➖ NEUTRAL
Query Dataset Tasks            40        32.5%         35.0%           +2.5%        📊 MODEST
"""

import torch
import numpy as np
import logging
from typing import List, Dict, Tuple
import json
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleSteering:
    """
    Simple steering using Cluster 9 neurons (proven approach).
    """
    
    def __init__(self, model):
        self.model = model
        self.hooks = []
        
        if hasattr(model, 'backbone'):
            self.layers = model.backbone.layers
        else:
            self.layers = model.layers
        
        # Cluster 9 neurons from original research
        self.cluster9_neurons = [
            4, 38, 84, 94, 163, 171, 268, 363, 401, 497, 
            564, 568, 582, 654, 659, 686
        ]
    
    def apply_steering(self, strength: float = 5.0, layer_idx: int = 20):
        """
        Apply Cluster 9 steering at specified layer.
        """
        if layer_idx >= len(self.layers):
            logger.warning(f"Layer {layer_idx} doesn't exist")
            return
        
        logger.info(f"🎯 Applying steering: Layer {layer_idx}, Strength {strength}x")
        
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
            for idx in self.cluster9_neurons:
                if idx < h_mod.shape[-1]:
                    h_mod[..., idx] *= strength
            
            if rest:
                return (h_mod,) + rest
            return h_mod
        
        h = target.register_forward_hook(hook)
        self.hooks.append(h)
    
    def remove_steering(self):
        """Remove all steering hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


class ComplexReasoningEvaluator:
    """
    Progressive difficulty testing for Mamba-130M.
    """
    
    def __init__(self, tokenizer, device="cuda"):
        self.tokenizer = tokenizer
        self.device = device
    
    def get_progressive_test_suite(self) -> Dict[str, List[Dict]]:
        """
        Test suite with increasing difficulty.
        All use query-focus prompting (proven to work).
        """
        
        return {
            # ============================================================
            # LEVEL 1: SIMPLE RECALL (Baseline - should get 100%)
            # ============================================================
            'level1_simple_recall': [
                {
                    'prompt': 'Question: What is my name?\nAnswer: My name is Alice.\nQuestion: What is my name?\nAnswer:',
                    'expected': 'Alice',
                    'alternatives': ['Alice', 'alice', 'My name is Alice'],
                    'difficulty': 'easy',
                    'task': 'single fact recall'
                },
                {
                    'prompt': 'Question: What is the code?\nAnswer: The code is BLUE42.\nQuestion: What is the code?\nAnswer:',
                    'expected': 'BLUE42',
                    'alternatives': ['BLUE42', 'blue42', 'The code is BLUE42'],
                    'difficulty': 'easy',
                    'task': 'exact recall'
                },
                {
                    'prompt': 'Question: What is 2+2?\nAnswer: 2+2 equals 4.\nQuestion: What is 2+2?\nAnswer:',
                    'expected': '4',
                    'alternatives': ['4', 'four', '2+2 equals 4'],
                    'difficulty': 'easy',
                    'task': 'arithmetic recall'
                },
            ],
            
            # ============================================================
            # LEVEL 2: TWO-HOP REASONING (Moderate)
            # ============================================================
            'level2_two_hop': [
                {
                    'prompt': 'Question: Who is taller?\nFacts: Alice is taller than Bob. Bob is taller than Carol.\nQuestion: Who is the tallest?\nAnswer:',
                    'expected': 'Alice',
                    'alternatives': ['Alice', 'alice'],
                    'difficulty': 'moderate',
                    'task': 'transitive comparison'
                },
                {
                    'prompt': 'Question: What happens to the ground?\nFacts: If it rains, the ground gets wet. It is raining.\nQuestion: What happens to the ground?\nAnswer:',
                    'expected': 'wet',
                    'alternatives': ['wet', 'gets wet', 'the ground gets wet'],
                    'difficulty': 'moderate',
                    'task': 'logical implication'
                },
                {
                    'prompt': 'Question: What color is the car?\nFacts: Alice drives a red car. Bob drives Alice to work.\nQuestion: What color is the car Bob drives?\nAnswer:',
                    'expected': 'red',
                    'alternatives': ['red', 'Red'],
                    'difficulty': 'moderate',
                    'task': 'indirect reference'
                },
                {
                    'prompt': 'Question: How much total?\nFacts: Apple costs 2 dollars. Orange costs 3 dollars.\nQuestion: If I buy one apple and one orange, how much total?\nAnswer:',
                    'expected': '5',
                    'alternatives': ['5', 'five', '5 dollars', '$5'],
                    'difficulty': 'moderate',
                    'task': 'arithmetic reasoning'
                },
            ],
            
            # ============================================================
            # LEVEL 3: THREE-HOP REASONING (Hard)
            # ============================================================
            'level3_three_hop': [
                {
                    'prompt': 'Question: Who is the shortest?\nFacts: Tom is taller than Jim. Jim is taller than Bob. Bob is taller than Sam.\nQuestion: Who is the shortest person?\nAnswer:',
                    'expected': 'Sam',
                    'alternatives': ['Sam', 'sam'],
                    'difficulty': 'hard',
                    'task': 'multi-step comparison'
                },
                {
                    'prompt': 'Question: What is Rex?\nFacts: All dogs are animals. All animals need food. Rex is a dog.\nQuestion: Does Rex need food?\nAnswer:',
                    'expected': 'yes',
                    'alternatives': ['yes', 'Yes', 'YES', 'Rex needs food'],
                    'difficulty': 'hard',
                    'task': 'syllogistic reasoning'
                },
                {
                    'prompt': 'Question: Where is the book?\nFacts: The book is on the table. The table is in the kitchen. The kitchen is in the house.\nQuestion: Is the book in the house?\nAnswer:',
                    'expected': 'yes',
                    'alternatives': ['yes', 'Yes', 'YES'],
                    'difficulty': 'hard',
                    'task': 'spatial reasoning chain'
                },
            ],
            
            # ============================================================
            # LEVEL 4: LONG-CONTEXT RECALL (5-7 facts)
            # ============================================================
            'level4_long_context': [
                {
                    'prompt': '''Question: What is Alice's favorite color?
Facts:
- Alice is 25 years old
- Alice lives in Paris
- Alice likes cats
- Alice's favorite color is blue
- Alice works as a teacher
- Alice speaks French

Question: What is Alice's favorite color?
Answer:''',
                    'expected': 'blue',
                    'alternatives': ['blue', 'Blue'],
                    'difficulty': 'hard',
                    'task': '6-fact recall'
                },
                {
                    'prompt': '''Question: What does Carol study?
Facts:
- Alice studies math
- Bob studies physics
- Carol studies chemistry
- David studies biology
- Emma studies history

Question: What does Carol study?
Answer:''',
                    'expected': 'chemistry',
                    'alternatives': ['chemistry', 'Chemistry'],
                    'difficulty': 'hard',
                    'task': '5-person association'
                },
                {
                    'prompt': '''Question: What is the 4th item?
List:
1. apple
2. banana
3. cherry
4. date
5. elderberry

Question: What is the 4th item in the list?
Answer:''',
                    'expected': 'date',
                    'alternatives': ['date', 'Date'],
                    'difficulty': 'hard',
                    'task': 'position in long list'
                },
            ],
            
            # ============================================================
            # LEVEL 5: COMBINED REASONING + MEMORY (Very Hard)
            # ============================================================
            'level5_combined': [
                {
                    'prompt': '''Question: How old is the oldest person?
Facts:
- Alice is 25 years old
- Bob is older than Alice
- Bob is 30 years old
- Carol is younger than Alice

Question: Who is the oldest person and how old are they?
Answer:''',
                    'expected': 'Bob',
                    'alternatives': ['Bob', 'bob', '30', 'Bob is 30', 'Bob, 30'],
                    'difficulty': 'very_hard',
                    'task': 'comparison + recall'
                },
                {
                    'prompt': '''Question: What is the total cost?
Shopping list:
- Apples: $3 (bought by Alice)
- Bread: $2 (bought by Bob)
- Cheese: $5 (bought by Alice)

Question: How much did Alice spend in total?
Answer:''',
                    'expected': '8',
                    'alternatives': ['8', '$8', '8 dollars', 'eight'],
                    'difficulty': 'very_hard',
                    'task': 'selective arithmetic'
                },
                {
                    'prompt': '''Question: Can Alice reach the top shelf?
Facts:
- Top shelf is 6 feet high
- Alice is 5 feet tall
- Bob is 6.5 feet tall
- Alice can reach 1 foot above her height

Question: Can Alice reach the top shelf?
Answer:''',
                    'expected': 'yes',
                    'alternatives': ['yes', 'Yes', 'YES'],
                    'difficulty': 'very_hard',
                    'task': 'multi-step arithmetic reasoning'
                },
            ],
            
            # ============================================================
            # LEVEL 6: VERY LONG CONTEXT (10+ facts) - Stress Test
            # ============================================================
            'level6_stress_test': [
                {
                    'prompt': '''Question: What is person E's occupation?
Database:
- Person A: Age 25, City Paris, Occupation Engineer
- Person B: Age 30, City London, Occupation Doctor
- Person C: Age 35, City Berlin, Occupation Teacher
- Person D: Age 28, City Madrid, Occupation Nurse
- Person E: Age 32, City Rome, Occupation Architect
- Person F: Age 27, City Vienna, Occupation Lawyer

Question: What is person E's occupation?
Answer:''',
                    'expected': 'Architect',
                    'alternatives': ['Architect', 'architect'],
                    'difficulty': 'extreme',
                    'task': '10+ fact database'
                },
                {
                    'prompt': '''Question: Who has the blue car?
Garage inventory:
- Slot 1: Red car owned by Alice
- Slot 2: Blue car owned by Bob
- Slot 3: Green car owned by Carol
- Slot 4: Yellow car owned by David
- Slot 5: Black car owned by Emma

Question: Who owns the blue car?
Answer:''',
                    'expected': 'Bob',
                    'alternatives': ['Bob', 'bob'],
                    'difficulty': 'extreme',
                    'task': 'structured long recall'
                },
            ],
        }
    
    def evaluate_progressive(self, model, test_suite: Dict) -> Dict:
        """
        Evaluate model on progressive difficulty levels.
        """
        all_results = {}
        
        logger.info("\n" + "="*80)
        logger.info("🧪 PROGRESSIVE DIFFICULTY EVALUATION")
        logger.info("="*80)
        
        for level_name, cases in test_suite.items():
            logger.info(f"\n{'='*70}")
            logger.info(f"Testing: {level_name.upper()}")
            logger.info(f"{'='*70}")
            
            correct = 0
            total = len(cases)
            details = []
            
            for i, case in enumerate(cases):
                prompt = case['prompt']
                expected = case['expected']
                alternatives = case.get('alternatives', [])
                difficulty = case.get('difficulty', 'unknown')  # Default for query datasets
                task = case.get('task', 'unknown')
                
                # Tokenize
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
                
                # Generate with appropriate length
                max_tokens = 100 if 'level6' in level_name or 'level5' in level_name else 50
                
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=max_tokens,
                        do_sample=False,
                        temperature=0.1,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id
                    )
                
                input_len = inputs['input_ids'].shape[1]
                response = self.tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
                response = response.strip()
                
                # Smart matching
                is_correct = self._smart_match(response, expected, alternatives)
                
                if is_correct:
                    correct += 1
                
                details.append({
                    'task': task,
                    'difficulty': difficulty,
                    'expected': expected,
                    'response': response[:80],
                    'correct': is_correct
                })
                
                # Show examples
                symbol = "✅" if is_correct else "❌"
                logger.info(f"  {i+1}. {symbol} {task} [{difficulty}]")
                logger.info(f"      Expected: '{expected}'")
                logger.info(f"      Got: '{response[:60]}...'")
            
            accuracy = correct / total if total > 0 else 0
            all_results[level_name] = {
                'accuracy': accuracy,
                'correct': correct,
                'total': total,
                'details': details
            }
            
            # Level summary
            if accuracy >= 0.8:
                status = "🟢 EXCELLENT"
            elif accuracy >= 0.6:
                status = "🟡 GOOD"
            elif accuracy >= 0.4:
                status = "🟠 MODERATE"
            else:
                status = "🔴 STRUGGLING"
            
            logger.info(f"\n  {status}: {accuracy*100:.1f}% ({correct}/{total})")
        
        return all_results
    
    def _smart_match(self, response: str, expected: str, alternatives: List[str]) -> bool:
        """Smart matching with flexible criteria."""
        response_lower = response.lower().strip()
        expected_lower = expected.lower().strip()
        
        # Exact match
        if expected_lower in response_lower:
            return True
        
        # Check alternatives
        if alternatives:
            for alt in alternatives:
                if alt.lower() in response_lower:
                    return True
        
        # First word match
        response_words = response_lower.split()
        if response_words and response_words[0] == expected_lower:
            return True
        
        # Number extraction for arithmetic
        import re
        if expected.isdigit():
            numbers = re.findall(r'\d+', response)
            if numbers and numbers[0] == expected:
                return True
        
        return False


class SteeringDiagnostics:
    """
    Diagnostic tools to understand steering effects.
    """
    
    def __init__(self, model, tokenizer, device="cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        
        if hasattr(model, 'backbone'):
            self.layers = model.backbone.layers
        else:
            self.layers = model.layers
        
        self.cluster9_neurons = [
            4, 38, 84, 94, 163, 171, 268, 363, 401, 497, 
            564, 568, 582, 654, 659, 686
        ]
    
    def analyze_neuron_activations(self, prompts: List[Tuple[str, str]], layer_idx: int = 20):
        """
        Analyze how neurons activate for different types of prompts.
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"🔬 NEURON ACTIVATION ANALYSIS - Layer {layer_idx}")
        logger.info(f"{'='*80}")
        
        activations = {}
        
        for prompt_type, prompt in prompts:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            # Capture activations
            captured = {}
            def capture_hook(name):
                def hook(module, input, output):
                    if isinstance(output, tuple):
                        hidden = output[0]
                    else:
                        hidden = output
                    captured[name] = hidden.detach().cpu()
                return hook
            
            layer = self.layers[layer_idx]
            target = getattr(layer, 'mixer', getattr(layer, 'ssm', layer))
            hook = target.register_forward_hook(capture_hook(f'layer_{layer_idx}'))
            
            with torch.no_grad():
                _ = self.model(**inputs)
            
            hook.remove()
            
            if f'layer_{layer_idx}' in captured:
                activations[prompt_type] = captured[f'layer_{layer_idx}']
        
        # Analyze Cluster 9 neurons specifically
        logger.info(f"\n📊 Cluster 9 Neuron Statistics:")
        logger.info(f"{'Prompt Type':<20} {'Mean Act':<12} {'Std Act':<12} {'Max Act':<12}")
        logger.info("-"*60)
        
        for prompt_type, act in activations.items():
            cluster_acts = act[..., self.cluster9_neurons].numpy()
            mean_act = np.mean(cluster_acts)
            std_act = np.std(cluster_acts)
            max_act = np.max(cluster_acts)
            
            logger.info(f"{prompt_type:<20} {mean_act:11.4f} {std_act:11.4f} {max_act:11.4f}")
        
        return activations
    
    def test_steering_strength_sweep(
        self, 
        test_prompts: List[Tuple[str, str, str]],  # (prompt, expected, task_type)
        layer_idx: int = 20,
        strengths: List[float] = [1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0]
    ):
        """
        Test different steering strengths to find optimal range.
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"🎚️ STEERING STRENGTH SWEEP - Layer {layer_idx}")
        logger.info(f"{'='*80}")
        
        results = {strength: {'correct': 0, 'total': 0} for strength in strengths}
        results[1.0] = {'correct': 0, 'total': 0}  # Baseline (no steering)
        
        for strength in strengths:
            logger.info(f"\n🔧 Testing strength: {strength}x")
            
            # Apply steering
            hooks = []
            if strength != 1.0:
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
                    for idx in self.cluster9_neurons:
                        if idx < h_mod.shape[-1]:
                            h_mod[..., idx] *= strength
                    
                    if rest:
                        return (h_mod,) + rest
                    return h_mod
                
                h = target.register_forward_hook(hook)
                hooks.append(h)
            
            # Test on prompts
            for prompt, expected, task_type in test_prompts:
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=50,
                        do_sample=False,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id
                    )
                
                input_len = inputs['input_ids'].shape[1]
                response = self.tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
                
                if expected.lower() in response.lower():
                    results[strength]['correct'] += 1
                results[strength]['total'] += 1
            
            # Remove hooks
            for h in hooks:
                h.remove()
            
            accuracy = results[strength]['correct'] / results[strength]['total']
            logger.info(f"   Accuracy: {accuracy*100:.1f}%")
        
        # Summary
        logger.info(f"\n📊 STRENGTH SWEEP SUMMARY:")
        logger.info(f"{'Strength':<12} {'Accuracy':<12} {'vs Baseline':<12}")
        logger.info("-"*40)
        
        baseline_acc = results[1.0]['correct'] / results[1.0]['total']
        best_strength = 1.0
        best_acc = baseline_acc
        
        for strength in sorted(strengths):
            acc = results[strength]['correct'] / results[strength]['total']
            diff = acc - baseline_acc
            logger.info(f"{strength:11.1f}x {acc*100:5.1f}%       {diff*100:+5.1f}%")
            
            if acc > best_acc:
                best_acc = acc
                best_strength = strength
        
        logger.info(f"\n🏆 Best strength: {best_strength}x ({best_acc*100:.1f}%)")
        
        return results, best_strength
    
    def test_layer_sweep(
        self,
        test_prompts: List[Tuple[str, str, str]],
        strength: float = 5.0,
        layers_to_test: List[int] = None
    ):
        """
        Test steering at different layers.
        """
        if layers_to_test is None:
            layers_to_test = [5, 10, 15, 18, 20, 22, 24]
        
        logger.info(f"\n{'='*80}")
        logger.info(f"🔍 LAYER SWEEP - Strength {strength}x")
        logger.info(f"{'='*80}")
        
        results = {}
        
        for layer_idx in layers_to_test:
            if layer_idx >= len(self.layers):
                continue
            
            logger.info(f"\n🔧 Testing layer: {layer_idx}")
            
            # Apply steering
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
                for idx in self.cluster9_neurons:
                    if idx < h_mod.shape[-1]:
                        h_mod[..., idx] *= strength
                
                if rest:
                    return (h_mod,) + rest
                return h_mod
            
            h = target.register_forward_hook(hook)
            
            # Test
            correct = 0
            total = 0
            
            for prompt, expected, task_type in test_prompts:
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=50,
                        do_sample=False,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id
                    )
                
                input_len = inputs['input_ids'].shape[1]
                response = self.tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
                
                if expected.lower() in response.lower():
                    correct += 1
                total += 1
            
            h.remove()
            
            accuracy = correct / total if total > 0 else 0
            results[layer_idx] = {'accuracy': accuracy, 'correct': correct, 'total': total}
            
            logger.info(f"   Accuracy: {accuracy*100:.1f}%")
        
        # Summary
        logger.info(f"\n📊 LAYER SWEEP SUMMARY:")
        logger.info(f"{'Layer':<12} {'Accuracy':<12}")
        logger.info("-"*30)
        
        best_layer = max(results.keys(), key=lambda k: results[k]['accuracy'])
        
        for layer_idx in sorted(results.keys()):
            acc = results[layer_idx]['accuracy']
            marker = "🏆" if layer_idx == best_layer else "  "
            logger.info(f"{marker} {layer_idx:<10} {acc*100:5.1f}%")
        
        logger.info(f"\n🏆 Best layer: {best_layer} ({results[best_layer]['accuracy']*100:.1f}%)")
        
        return results, best_layer
    
    def analyze_failure_cases(
        self,
        test_cases: List[Dict],
        layer_idx: int = 20,
        strength: float = 5.0
    ):
        """
        Analyze specific cases where steering hurts performance.
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"🔍 FAILURE CASE ANALYSIS")
        logger.info(f"{'='*80}")
        
        failures = {'without_steering': [], 'with_steering': []}
        
        # Test without steering
        logger.info(f"\n📊 Testing WITHOUT steering...")
        for case in test_cases:
            inputs = self.tokenizer(case['prompt'], return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=50,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            input_len = inputs['input_ids'].shape[1]
            response = self.tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
            
            is_correct = case['expected'].lower() in response.lower()
            if not is_correct:
                failures['without_steering'].append(case)
        
        # Test with steering
        logger.info(f"\n📊 Testing WITH steering (layer={layer_idx}, strength={strength}x)...")
        
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
            for idx in self.cluster9_neurons:
                if idx < h_mod.shape[-1]:
                    h_mod[..., idx] *= strength
            
            if rest:
                return (h_mod,) + rest
            return h_mod
        
        h = target.register_forward_hook(hook)
        
        for case in test_cases:
            inputs = self.tokenizer(case['prompt'], return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=50,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            input_len = inputs['input_ids'].shape[1]
            response = self.tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
            
            is_correct = case['expected'].lower() in response.lower()
            if not is_correct:
                failures['with_steering'].append(case)
        
        h.remove()
        
        # Analysis
        logger.info(f"\n📊 FAILURE ANALYSIS:")
        logger.info(f"   Without steering: {len(failures['without_steering'])} failures")
        logger.info(f"   With steering:    {len(failures['with_steering'])} failures")
        
        # Find cases that BROKE due to steering
        broke_cases = []
        for case in test_cases:
            failed_without = case in failures['without_steering']
            failed_with = case in failures['with_steering']
            
            if not failed_without and failed_with:
                broke_cases.append(case)
        
        if broke_cases:
            logger.info(f"\n⚠️ Cases BROKEN by steering ({len(broke_cases)}):")
            for case in broke_cases[:5]:  # Show first 5
                logger.info(f"   - {case['task']}: Expected '{case['expected']}'")
        
        return failures, broke_cases


def run_comprehensive_diagnostics():
    """
    Run all diagnostic tests.
    """
    from mamba_model_loader import load_mamba_model_and_tokenizer
    
    logger.info("="*80)
    logger.info("🔬 COMPREHENSIVE STEERING DIAGNOSTICS")
    logger.info("="*80)
    
    # Load model
    model, tokenizer = load_mamba_model_and_tokenizer(
        model_name="state-spaces/mamba-130m-hf",
        device="cuda" if torch.cuda.is_available() else "cpu",
        use_mamba_class=True,
        fallback_to_auto=True
    )
    
    device = next(model.parameters()).device
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    diagnostics = SteeringDiagnostics(model, tokenizer, device)
    
    # Get test cases
    evaluator = ComplexReasoningEvaluator(tokenizer, device)
    test_suite = evaluator.get_progressive_test_suite()
    
    # Prepare test prompts
    test_prompts = []
    for level_name, cases in test_suite.items():
        if 'level' in level_name:
            for case in cases[:2]:  # Take 2 from each level
                test_prompts.append((case['prompt'], case['expected'], case['task']))
    
    # ================================================
    # DIAGNOSTIC 1: Neuron Activation Analysis
    # ================================================
    analysis_prompts = [
        ('simple_recall', 'Question: What is my name?\nAnswer: My name is Alice.\nQuestion: What is my name?\nAnswer:'),
        ('two_hop', 'Question: Who is taller?\nFacts: Alice is taller than Bob. Bob is taller than Carol.\nQuestion: Who is the tallest?\nAnswer:'),
        ('long_context', '''Question: What is Alice's favorite color?\nFacts:\n- Alice is 25 years old\n- Alice lives in Paris\n- Alice likes cats\n- Alice's favorite color is blue\n- Alice works as a teacher\n- Alice speaks French\n\nQuestion: What is Alice's favorite color?\nAnswer:'''),
    ]
    
    diagnostics.analyze_neuron_activations(analysis_prompts, layer_idx=20)
    
    # ================================================
    # DIAGNOSTIC 2: Strength Sweep
    # ================================================
    results, best_strength = diagnostics.test_steering_strength_sweep(
        test_prompts=test_prompts[:10],  # Test on 10 prompts
        layer_idx=20,
        strengths=[1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0]
    )
    
    # ================================================
    # DIAGNOSTIC 3: Layer Sweep
    # ================================================
    layer_results, best_layer = diagnostics.test_layer_sweep(
        test_prompts=test_prompts[:10],
        strength=best_strength,
        layers_to_test=[5, 10, 15, 18, 20, 22, 24]
    )
    
    # ================================================
    # DIAGNOSTIC 4: Failure Case Analysis
    # ================================================
    three_hop_cases = test_suite['level3_three_hop']
    failures, broke_cases = diagnostics.analyze_failure_cases(
        test_cases=three_hop_cases,
        layer_idx=20,
        strength=5.0
    )
    
    # ================================================
    # FINAL RECOMMENDATIONS
    # ================================================
    logger.info(f"\n{'='*80}")
    logger.info(f"💡 FINAL RECOMMENDATIONS")
    logger.info(f"{'='*80}")
    
    logger.info(f"\n🔧 Optimal Configuration Found:")
    logger.info(f"   Best Layer:    {best_layer}")
    logger.info(f"   Best Strength: {best_strength}x")
    
    if len(broke_cases) > 2:
        logger.info(f"\n⚠️ WARNING: Steering breaks {len(broke_cases)} cases that work without it")
        logger.info(f"   This suggests Cluster 9 neurons may not be optimal for all tasks")
        logger.info(f"\n💡 Recommendations:")
        logger.info(f"   1. Re-run mechanistic interpretability to find task-specific neurons")
        logger.info(f"   2. Try neuron ablation studies to identify harmful neurons")
        logger.info(f"   3. Consider using different neurons for different task types")
        logger.info(f"   4. Test with lower strengths (1.5-2.5x) for better balance")
    else:
        logger.info(f"\n✅ Steering appears beneficial with optimal settings")
        logger.info(f"   Try these settings in your main experiment:")
        logger.info(f"   - Layer: {best_layer}")
        logger.info(f"   - Strength: {best_strength}x")
    
    logger.info(f"\n📚 Next Steps:")
    logger.info(f"   1. Test optimal configuration on full test suite")
    logger.info(f"   2. Consider ensemble of multiple steering strategies")
    logger.info(f"   3. Investigate why query dataset tasks remain low (~35%)")
    logger.info(f"      - May need fine-tuning or different approach")
    
    return {
        'best_strength': best_strength,
        'best_layer': best_layer,
        'broke_cases': len(broke_cases)
    }


def run_capability_assessment(
    trained_model_path: str = None,
    query_datasets: List[str] = ['squad']
):
    """
    Comprehensive assessment of Mamba-130M capabilities.
    
    Now uses:
    - Trained model (trained on The Pile) if provided
    - Query datasets for testing
    
    Args:
        trained_model_path: Path to model trained on The Pile (optional)
        query_datasets: List of query datasets to use for testing
    """
    from mamba_model_loader import load_mamba_model_and_tokenizer
    from query_dataset_loader import get_query_tasks_for_evaluation
    
    logger.info("="*80)
    logger.info("🔬 MAMBA-130M CAPABILITY ASSESSMENT")
    logger.info("Testing: Complex reasoning, long context, multi-step problems")
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
    
    # Initialize evaluator
    evaluator = ComplexReasoningEvaluator(tokenizer, device)
    
    # Get test suite
    test_suite = evaluator.get_progressive_test_suite()
    
    # Load query datasets for testing
    logger.info("\n📚 Loading query datasets for testing...")
    try:
        query_tasks = get_query_tasks_for_evaluation(
            datasets=query_datasets,
            num_per_dataset=30
        )
        logger.info(f"✅ Loaded {sum(len(v) for v in query_tasks.values())} queries from test datasets")
        
        # Add query tasks as a new level
        all_query_tasks = []
        for task_list in query_tasks.values():
            all_query_tasks.extend(task_list[:20])  # Take first 20 from each
        test_suite['query_dataset_tasks'] = all_query_tasks
    except Exception as e:
        logger.warning(f"⚠️ Could not load query datasets: {e}")
        logger.info("   Continuing with original test suite only")
        query_tasks = {}
        test_suite['query_dataset_tasks'] = []
    
    # ================================================
    # BASELINE: WITHOUT STEERING
    # ================================================
    logger.info("\n" + "="*80)
    logger.info("📊 BASELINE EVALUATION (No Steering)")
    logger.info("="*80)
    
    baseline_results = evaluator.evaluate_progressive(model, test_suite)
    
    # ================================================
    # WITH STEERING
    # ================================================
    logger.info("\n" + "="*80)
    logger.info("🎯 EVALUATION WITH STEERING")
    logger.info("="*80)
    
    steering = SimpleSteering(model)
    steering.apply_steering(strength=5.0, layer_idx=20)
    
    steering_results = evaluator.evaluate_progressive(model, test_suite)
    
    # Remove steering
    steering.remove_steering()
    
    # Use steering results for main analysis
    results = steering_results
    
    # ================================================
    # COMPREHENSIVE ANALYSIS WITH COMPARISON
    # ================================================
    logger.info("\n" + "="*80)
    logger.info("📊 PERFORMANCE COMPARISON: BASELINE vs STEERING")
    logger.info("="*80)
    if query_tasks:
        logger.info(f"   Includes queries from: {', '.join(query_tasks.keys())}")
    
    logger.info("\n" + "-"*80)
    logger.info(f"{'Level':<30} {'Tasks':<8} {'Baseline':<12} {'With Steering':<15} {'Change':<12} {'Status'}")
    logger.info("-"*80)
    
    capability_map = {
        'level1_simple_recall': 'Simple Recall',
        'level2_two_hop': 'Two-Hop Reasoning',
        'level3_three_hop': 'Three-Hop Reasoning',
        'level4_long_context': 'Long Context (5-7 facts)',
        'level5_combined': 'Combined Reasoning + Memory',
        'level6_stress_test': 'Stress Test (10+ facts)',
        'query_dataset_tasks': 'Query Dataset Tasks'
    }
    
    for level_key, level_name in capability_map.items():
        if level_key in results and level_key in baseline_results:
            baseline_acc = baseline_results[level_key]['accuracy']
            steering_acc = results[level_key]['accuracy']
            change = steering_acc - baseline_acc
            change_pct = change * 100
            
            if change > 0.10:
                status = "✅ EXCELLENT"
            elif change > 0.05:
                status = "📈 GOOD"
            elif change > 0:
                status = "📊 MODEST"
            elif change > -0.05:
                status = "➖ NEUTRAL"
            else:
                status = "❌ NEGATIVE"
            
            logger.info(f"{level_name:<30} {results[level_key]['total']:<8} "
                       f"{baseline_acc*100:5.1f}%        {steering_acc*100:5.1f}%          "
                       f"{change_pct:+5.1f}%        {status}")
    
    # ================================================
    # SUMMARY STATISTICS
    # ================================================
    logger.info("\n" + "="*80)
    logger.info("📊 SUMMARY STATISTICS")
    logger.info("="*80)
    
    def calc_avg(results_dict, level_keys):
        accs = [results_dict[k]['accuracy'] for k in level_keys if k in results_dict]
        return sum(accs) / len(accs) if accs else 0.0
    
    baseline_simple = calc_avg(baseline_results, ['level1_simple_recall'])
    baseline_moderate = calc_avg(baseline_results, ['level2_two_hop', 'level3_three_hop'])
    baseline_hard = calc_avg(baseline_results, ['level4_long_context', 'level5_combined'])
    baseline_extreme = calc_avg(baseline_results, ['level6_stress_test'])
    
    steering_simple = calc_avg(results, ['level1_simple_recall'])
    steering_moderate = calc_avg(results, ['level2_two_hop', 'level3_three_hop'])
    steering_hard = calc_avg(results, ['level4_long_context', 'level5_combined'])
    steering_extreme = calc_avg(results, ['level6_stress_test'])
    
    logger.info(f"\n{'Category':<30} {'Baseline':<12} {'With Steering':<15} {'Improvement':<12}")
    logger.info("-"*80)
    logger.info(f"{'Simple Recall':<30} {baseline_simple*100:5.1f}%        {steering_simple*100:5.1f}%          {(steering_simple-baseline_simple)*100:+5.1f}%")
    logger.info(f"{'Moderate (2-3 hops)':<30} {baseline_moderate*100:5.1f}%        {steering_moderate*100:5.1f}%          {(steering_moderate-baseline_moderate)*100:+5.1f}%")
    logger.info(f"{'Hard (long context)':<30} {baseline_hard*100:5.1f}%        {steering_hard*100:5.1f}%          {(steering_hard-baseline_hard)*100:+5.1f}%")
    logger.info(f"{'Extreme (10+ facts)':<30} {baseline_extreme*100:5.1f}%        {steering_extreme*100:5.1f}%          {(steering_extreme-baseline_extreme)*100:+5.1f}%")
    
    overall_baseline = (baseline_simple + baseline_moderate + baseline_hard + baseline_extreme) / 4
    overall_steering = (steering_simple + steering_moderate + steering_hard + steering_extreme) / 4
    overall_improvement = overall_steering - overall_baseline
    
    logger.info("-"*80)
    logger.info(f"{'OVERALL AVERAGE':<30} {overall_baseline*100:5.1f}%        {overall_steering*100:5.1f}%          {overall_improvement*100:+5.1f}%")
    
    # ================================================
    # CAPABILITY ANALYSIS (Using steering results)
    # ================================================
    logger.info("\n" + "="*80)
    logger.info("📊 CAPABILITY ANALYSIS (With Steering)")
    logger.info("="*80)
    
    logger.info("\n" + "-"*80)
    logger.info(f"{'Level':<30} {'Tasks':<8} {'Accuracy':<12} {'Assessment'}")
    logger.info("-"*80)
    
    for level_key, level_name in capability_map.items():
        if level_key in results:
            res = results[level_key]
            acc = res['accuracy']
            
            if acc >= 0.8:
                assessment = "✅ Strong capability"
            elif acc >= 0.6:
                assessment = "📈 Moderate capability"
            elif acc >= 0.4:
                assessment = "⚠️ Limited capability"
            else:
                assessment = "❌ Struggles significantly"
            
            logger.info(f"{level_name:<30} {res['total']:<8} "
                       f"{acc*100:5.1f}%       {assessment}")
    
    # ================================================
    # CAPABILITY CEILING ANALYSIS
    # ================================================
    logger.info("\n" + "="*80)
    logger.info("🎯 CAPABILITY CEILING")
    logger.info("="*80)
    
    # Find where performance drops
    level_order = [
        'level1_simple_recall',
        'level2_two_hop',
        'level3_three_hop',
        'level4_long_context',
        'level5_combined',
        'level6_stress_test',
        'query_dataset_tasks'
    ]
    
    ceiling_found = False
    for level_key in level_order:
        if level_key in results:
            acc = results[level_key]['accuracy']
            level_name = capability_map[level_key]
            
            if acc >= 0.8 and not ceiling_found:
                logger.info(f"✅ {level_name}: Strong performance ({acc*100:.1f}%)")
            elif acc >= 0.5 and not ceiling_found:
                logger.info(f"⚠️ {level_name}: Performance degrading ({acc*100:.1f}%)")
                ceiling_found = True
            elif not ceiling_found:
                logger.info(f"❌ {level_name}: Capability ceiling reached ({acc*100:.1f}%)")
                ceiling_found = True
            else:
                logger.info(f"❌ {level_name}: Below threshold ({acc*100:.1f}%)")
    
    # ================================================
    # RECOMMENDATIONS
    # ================================================
    logger.info("\n" + "="*80)
    logger.info("💡 RECOMMENDATIONS")
    logger.info("="*80)
    
    # Use the calculated averages from summary section
    avg_simple = steering_simple
    avg_moderate = steering_moderate
    avg_hard = steering_hard
    avg_extreme = steering_extreme
    
    logger.info(f"\n📊 Performance Summary (With Steering):")
    logger.info(f"   Simple tasks (recall):        {avg_simple*100:.1f}% (baseline: {baseline_simple*100:.1f}%)")
    logger.info(f"   Moderate tasks (2-3 hops):    {avg_moderate*100:.1f}% (baseline: {baseline_moderate*100:.1f}%)")
    logger.info(f"   Hard tasks (long context):     {avg_hard*100:.1f}% (baseline: {baseline_hard*100:.1f}%)")
    logger.info(f"   Extreme tasks (10+ facts):   {avg_extreme*100:.1f}% (baseline: {baseline_extreme*100:.1f}%)")
    logger.info(f"   Overall improvement:           {overall_improvement*100:+.1f}%")
    
    logger.info(f"\n💭 What Mamba-130M CAN do:")
    if avg_simple >= 0.8:
        logger.info(f"   ✅ Simple recall with query-focus prompting")
    if avg_moderate >= 0.6:
        logger.info(f"   ✅ Basic multi-hop reasoning (2-3 steps)")
    if avg_hard >= 0.5:
        logger.info(f"   ✅ Moderate long-context tasks (5-7 facts)")
    
    logger.info(f"\n💭 What Mamba-130M STRUGGLES with:")
    if avg_moderate < 0.6:
        logger.info(f"   ⚠️ Multi-hop reasoning beyond 2 steps")
    if avg_hard < 0.5:
        logger.info(f"   ⚠️ Long-context recall (5+ facts)")
    if avg_extreme < 0.4:
        logger.info(f"   ⚠️ Very long contexts (10+ facts)")
    
    logger.info(f"\n🎯 Recommendations for your research:")
    
    if avg_simple >= 0.8 and avg_moderate >= 0.6:
        logger.info(f"   ✅ Mamba-130M is capable with right prompting!")
        logger.info(f"   → Focus on query-focus patterns for papers")
        logger.info(f"   → Document the 2-3 hop reasoning capability")
    
    if avg_hard < 0.5:
        logger.info(f"   📊 For longer contexts, consider:")
        logger.info(f"   → Testing Mamba-370M or Mamba-790M")
        logger.info(f"   → Breaking tasks into smaller sub-queries")
        logger.info(f"   → Using retrieval-augmented approaches")
    
    if avg_extreme < 0.4:
        logger.info(f"   ⚠️ 10+ fact contexts exceed 130M capacity")
        logger.info(f"   → This is expected for a small model")
        logger.info(f"   → Larger models or different architectures needed")
    
    # ================================================
    # SAVE RESULTS
    # ================================================
    output_path = Path("experiment_logs/capability_assessment.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump({
            'model': trained_model_path if trained_model_path else 'state-spaces/mamba-130m-hf',
            'trained_on_pile': trained_model_path is not None,
            'prompting_strategy': 'query_focus',
            'test_datasets': query_datasets,
            'query_tasks_included': len(query_tasks) > 0,
            'steering_config': {
                'method': 'cluster9_neurons',
                'strength': 5.0,
                'layer': 20
            },
            'baseline_results': {k: {
                'accuracy': float(v['accuracy']),
                'correct': v['correct'],
                'total': v['total']
            } for k, v in baseline_results.items()},
            'steering_results': {k: {
                'accuracy': float(v['accuracy']),
                'correct': v['correct'],
                'total': v['total']
            } for k, v in results.items()},
            'comparison': {
                'baseline': {
                    'simple_recall': float(baseline_simple),
                    'moderate_reasoning': float(baseline_moderate),
                    'hard_tasks': float(baseline_hard),
                    'extreme_tasks': float(baseline_extreme),
                    'overall': float(overall_baseline)
                },
                'steering': {
                    'simple_recall': float(avg_simple),
                    'moderate_reasoning': float(avg_moderate),
                    'hard_tasks': float(avg_hard),
                    'extreme_tasks': float(avg_extreme),
                    'overall': float(overall_steering)
                },
                'improvements': {
                    'simple_recall': float(steering_simple - baseline_simple),
                    'moderate_reasoning': float(steering_moderate - baseline_moderate),
                    'hard_tasks': float(steering_hard - baseline_hard),
                    'extreme_tasks': float(steering_extreme - baseline_extreme),
                    'overall': float(overall_improvement)
                }
            },
            'capability_ceiling': 'moderate_reasoning' if avg_moderate >= 0.6 else 'simple_recall'
        }, f, indent=2)
    
    logger.info(f"\n💾 Results saved to: {output_path}")
    logger.info("="*80)
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Capability assessment or diagnostics")
    parser.add_argument("--trained_model", type=str, default=None,
                       help="Path to model trained on The Pile")
    parser.add_argument("--query_datasets", type=str, nargs='+', 
                       default=['squad'],
                       help="Query datasets to test on (squad, natural_questions, triviaqa)")
    parser.add_argument("--diagnostics", action="store_true",
                       help="Run comprehensive steering diagnostics instead of assessment")
    
    args = parser.parse_args()
    
    if args.diagnostics:
        results = run_comprehensive_diagnostics()
    else:
        results = run_capability_assessment(
            trained_model_path=args.trained_model,
            query_datasets=args.query_datasets
        )