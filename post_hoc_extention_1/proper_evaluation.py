"""
FIXED EVALUATION: Proper metrics for natural text continuation

Problem: Current evaluation is too lenient (accepts empty strings, common words)
Solution: Measure ACTUAL recall capability with proper metrics
"""

import torch
import logging
from typing import List, Dict, Tuple, Optional
import re
from difflib import SequenceMatcher

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProperEvaluationMetrics:
    """
    Evaluation that distinguishes between:
    1. Exact recall (model outputs the target entity)
    2. Contextual recall (model discusses the entity naturally)
    3. Failure (model doesn't recall at all)
    """
    
    @staticmethod
    def evaluate_recall(
        response: str,
        expected: str,
        context_keywords: List[str],
        task_type: str = "entity_recall"
    ) -> Dict:
        """
        Multi-level evaluation:
        - exact: Target appears verbatim in first 10 tokens
        - contextual: Target appears anywhere in response
        - semantic: Response discusses the right topic
        - failed: No relevant recall
        """
        response_lower = response.lower().strip()
        expected_lower = expected.lower().strip()
        
        # Skip if expected is empty (continuation tasks)
        if not expected_lower:
            # For continuation tasks, check if response is relevant
            if context_keywords:
                keyword_matches = sum(1 for kw in context_keywords if kw.lower() in response_lower)
                if keyword_matches >= len(context_keywords) * 0.5:
                    return {
                        'response': response,
                        'expected': expected,
                        'exact_match': False,
                        'immediate_recall': False,
                        'contextual_recall': True,
                        'semantic_relevance': True,
                        'off_topic': False,
                        'score': 0.6,
                        'details': f"📝 CONTINUATION: Relevant continuation ({keyword_matches}/{len(context_keywords)} keywords)"
                    }
            return {
                'response': response,
                'expected': expected,
                'exact_match': False,
                'immediate_recall': False,
                'contextual_recall': False,
                'semantic_relevance': False,
                'off_topic': True,
                'score': 0.0,
                'details': f"❌ EMPTY: No expected value to check"
            }
        
        # Get first 20 tokens for "immediate recall" check
        first_tokens = ' '.join(response.split()[:20]).lower()
        
        result = {
            'response': response,
            'expected': expected,
            'exact_match': False,
            'immediate_recall': False,
            'contextual_recall': False,
            'semantic_relevance': False,
            'off_topic': False,
            'score': 0.0,
            'details': ''
        }
        
        # 1. Exact match in first few tokens (BEST - what we really want)
        if expected_lower in first_tokens:
            result['exact_match'] = True
            result['immediate_recall'] = True
            result['contextual_recall'] = True
            result['semantic_relevance'] = True
            result['score'] = 1.0
            result['details'] = f"✅ EXACT: '{expected}' appears immediately"
            return result
        
        # 2. Exact match anywhere in response (GOOD - recalls correctly)
        if expected_lower in response_lower:
            result['contextual_recall'] = True
            result['semantic_relevance'] = True
            result['score'] = 0.7
            result['details'] = f"📝 CONTEXTUAL: '{expected}' appears later in response"
            return result
        
        # 3. Semantic relevance (OK - discusses related concepts)
        if context_keywords:
            keyword_matches = sum(1 for kw in context_keywords if kw.lower() in response_lower)
            if keyword_matches >= len(context_keywords) * 0.5:
                result['semantic_relevance'] = True
                result['score'] = 0.3
                result['details'] = f"🔍 SEMANTIC: Discusses related concepts ({keyword_matches}/{len(context_keywords)} keywords)"
                return result
        
        # 4. Partial match (fuzzy string matching)
        similarity = SequenceMatcher(None, expected_lower, response_lower).ratio()
        if similarity > 0.6:
            result['score'] = similarity * 0.5
            result['details'] = f"📊 PARTIAL: {similarity*100:.1f}% similar"
            return result
        
        # 5. Complete failure
        result['off_topic'] = True
        result['score'] = 0.0
        result['details'] = f"❌ OFF-TOPIC: No recall of '{expected}'"
        return result
    
    @staticmethod
    def evaluate_qa_task(
        response: str,
        expected_answer: str,
        question: str,
        context: str
    ) -> Dict:
        """
        Special evaluation for Q&A tasks.
        
        For Q&A, we want:
        1. Answer extraction: Does response contain the answer?
        2. Answer position: Is answer in first 10 tokens? (immediate)
        3. Context awareness: Does it reference the context?
        """
        response_clean = response.strip()
        answer_lower = expected_answer.lower().strip()
        
        if not answer_lower:
            return {
                'response': response,
                'expected': expected_answer,
                'answer_in_first_sentence': False,
                'answer_anywhere': False,
                'answer_position': -1,
                'score': 0.0,
                'details': '❌ No expected answer provided'
            }
        
        # Split into sentences
        sentences = [s.strip() for s in response_clean.split('.') if s.strip()]
        first_sentence = sentences[0] if sentences else response_clean
        
        result = {
            'response': response,
            'expected': expected_answer,
            'answer_in_first_sentence': False,
            'answer_anywhere': False,
            'answer_position': -1,
            'score': 0.0,
            'details': ''
        }
        
        # Check if answer is in first sentence (ideal for Q&A)
        if answer_lower in first_sentence.lower():
            result['answer_in_first_sentence'] = True
            result['answer_anywhere'] = True
            result['answer_position'] = first_sentence.lower().find(answer_lower)
            result['score'] = 1.0
            result['details'] = f"✅ CORRECT: Answer in first sentence"
            return result
        
        # Check if answer appears anywhere
        if answer_lower in response_clean.lower():
            position = response_clean.lower().find(answer_lower)
            words_before = len(response_clean[:position].split())
            result['answer_anywhere'] = True
            result['answer_position'] = position
            result['score'] = 0.7
            result['details'] = f"📝 FOUND: Answer after {words_before} words"
            return result
        
        # Check for partial matches (important for long answers)
        answer_words = answer_lower.split()
        if len(answer_words) > 1:
            matches = sum(1 for word in answer_words if word in response_clean.lower())
            if matches >= len(answer_words) * 0.7:
                result['score'] = 0.4
                result['details'] = f"📊 PARTIAL: {matches}/{len(answer_words)} answer words present"
                return result
        
        result['score'] = 0.0
        result['details'] = f"❌ WRONG: Expected '{expected_answer}', not found"
        return result


class ImprovedDiagnostic:
    """
    Run the diagnostic with PROPER evaluation.
    """
    
    def __init__(self, tokenizer, device="cuda"):
        self.tokenizer = tokenizer
        self.device = device
        self.evaluator = ProperEvaluationMetrics()
    
    def test_with_full_output_analysis(self, model, test_cases: Dict) -> Dict:
        """
        Test with detailed output analysis (no truncation in logging).
        """
        results = {}
        
        for strategy_name, cases in test_cases.items():
            logger.info(f"\n{'='*80}")
            logger.info(f"Testing: {strategy_name}")
            logger.info(f"{'='*80}")
            
            strategy_results = {
                'cases': [],
                'exact_matches': 0,
                'immediate_recalls': 0,
                'contextual_recalls': 0,
                'total': len(cases),
                'average_score': 0.0
            }
            
            for case in cases:
                prompt = case['prompt']
                expected = case['expected']
                
                # Tokenize
                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    max_length=1024,
                    truncation=True
                ).to(self.device)
                
                # Generate with NO sampling for reproducibility
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=50,  # Enough to see what model generates
                        do_sample=False,
                        temperature=1.0,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id
                    )
                
                # Decode FULL output (no truncation)
                input_len = inputs['input_ids'].shape[1]
                full_response = self.tokenizer.decode(
                    outputs[0][input_len:], 
                    skip_special_tokens=True
                ).strip()
                
                # Evaluate based on task type
                if 'qa' in case.get('task', '').lower() or 'question' in prompt.lower():
                    eval_result = self.evaluator.evaluate_qa_task(
                        response=full_response,
                        expected_answer=expected,
                        question=case.get('question', ''),
                        context=prompt
                    )
                else:
                    eval_result = self.evaluator.evaluate_recall(
                        response=full_response,
                        expected=expected,
                        context_keywords=case.get('context_keywords', []),
                        task_type=case.get('task', 'unknown')
                    )
                
                # Update counts
                if eval_result['score'] >= 0.95:
                    strategy_results['exact_matches'] += 1
                if eval_result.get('immediate_recall', False):
                    strategy_results['immediate_recalls'] += 1
                if eval_result.get('contextual_recall', False):
                    strategy_results['contextual_recalls'] += 1
                
                strategy_results['cases'].append({
                    'prompt': prompt,
                    'expected': expected,
                    'response': full_response,  # FULL response
                    'evaluation': eval_result,
                    'task': case.get('task', 'unknown')
                })
                
                # Log with full output
                logger.info(f"\n📝 Task: {case.get('task', 'unknown')}")
                logger.info(f"   Prompt: {prompt[:80]}{'...' if len(prompt) > 80 else ''}")
                logger.info(f"   Expected: '{expected}'")
                logger.info(f"   Response: '{full_response}'")  # FULL OUTPUT
                logger.info(f"   {eval_result['details']} (score: {eval_result['score']:.2f})")
                
                # Clear cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Calculate averages
            strategy_results['average_score'] = sum(
                c['evaluation']['score'] for c in strategy_results['cases']
            ) / len(strategy_results['cases'])
            
            results[strategy_name] = strategy_results
            
            # Summary
            logger.info(f"\n{'='*80}")
            logger.info(f"Summary for {strategy_name}:")
            logger.info(f"  Exact matches: {strategy_results['exact_matches']}/{strategy_results['total']}")
            logger.info(f"  Immediate recalls: {strategy_results['immediate_recalls']}/{strategy_results['total']}")
            logger.info(f"  Contextual recalls: {strategy_results['contextual_recalls']}/{strategy_results['total']}")
            logger.info(f"  Average score: {strategy_results['average_score']:.2f}")
            logger.info(f"{'='*80}")
        
        return results
    
    def get_better_test_cases(self) -> Dict[str, List[Dict]]:
        """
        Test cases with proper expected values and evaluation criteria.
        """
        return {
            'narrative_continuation': [
                {
                    'prompt': 'Alice lived in Paris with her cats. She loved her life there. Alice',
                    'expected': '',  # We expect natural continuation about Alice
                    'context_keywords': ['Alice', 'Paris', 'cats', 'loved', 'lived'],
                    'task': 'name_in_narrative',
                    'evaluation_note': 'Should continue naturally mentioning Alice or her activities'
                },
                {
                    'prompt': 'The secret code was BLUE42. Nobody knew it except the agents. The code',
                    'expected': 'BLUE42',
                    'context_keywords': ['BLUE42', 'code', 'secret', 'agents'],
                    'task': 'code_recall',
                    'evaluation_note': 'Should mention BLUE42 explicitly'
                },
            ],
            
            'explicit_qa': [
                {
                    'prompt': '''Context: Newton's law states that gravitational force is inversely proportional to the square of the distance. This means the force decreases at larger distances.
Question: The force decreases at what?
Answer:''',
                    'expected': 'larger distances',
                    'context_keywords': ['force', 'distance', 'decreases', 'larger'],
                    'task': 'extract_answer',
                    'evaluation_note': 'Should extract "larger distances" as answer'
                },
            ],
            
            'implicit_qa': [
                {
                    'prompt': '''Newton's law states that gravitational force is inversely proportional to the square of the distance. This means the force decreases at larger distances. In other words, the gravitational force decreases at''',
                    'expected': 'larger distances',
                    'context_keywords': ['larger', 'greater', 'increased', 'distance'],
                    'task': 'complete_sentence',
                    'evaluation_note': 'Should complete with "larger distances" or equivalent'
                },
            ],
        }


def run_proper_diagnostic(model_name: str = "state-spaces/mamba-130m-hf"):
    """
    Run diagnostic with PROPER evaluation metrics.
    """
    from mamba_model_loader import load_mamba_model_and_tokenizer
    
    logger.info("="*80)
    logger.info("🔬 PROPER DIAGNOSTIC WITH FULL OUTPUT ANALYSIS")
    logger.info("="*80)
    logger.info("\nGoal: See EXACTLY what the model generates (no truncation)")
    logger.info("      Use proper evaluation metrics (not just substring matching)")
    logger.info("="*80)
    
    # Load model
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
    diagnostic = ImprovedDiagnostic(tokenizer, device)
    test_cases = diagnostic.get_better_test_cases()
    
    # Run tests
    results = diagnostic.test_with_full_output_analysis(model, test_cases)
    
    # Final summary
    logger.info("\n" + "="*80)
    logger.info("🏆 FINAL ANALYSIS")
    logger.info("="*80)
    
    for strategy, data in results.items():
        logger.info(f"\n{strategy}:")
        logger.info(f"  Average score: {data['average_score']:.2f}")
        logger.info(f"  Exact matches: {data['exact_matches']}/{data['total']}")
        logger.info(f"  Immediate recalls: {data['immediate_recalls']}/{data['total']}")
        
        if data['average_score'] >= 0.8:
            logger.info(f"  ✅ STRONG PERFORMANCE")
        elif data['average_score'] >= 0.5:
            logger.info(f"  📊 MODERATE PERFORMANCE")
        else:
            logger.info(f"  ❌ POOR PERFORMANCE")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="state-spaces/mamba-130m-hf")
    args = parser.parse_args()
    
    run_proper_diagnostic(model_name=args.model)

