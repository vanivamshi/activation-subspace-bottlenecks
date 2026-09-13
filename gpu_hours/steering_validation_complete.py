#!/usr/bin/env python3
"""
Option B: All-in-One Steering Validation Script
Runs both discovery (Stage 1) and validation (Stage 2) in a single run.
"""

import argparse
import sys
from pathlib import Path
import torch
from ablation_3 import (
    SteeringValidator,
    StructuredTaskGenerator,
    save_discovered_neurons,
    generate_protocol_summary,
    create_neuron_importance_visualization,
    logger
)
from mamba_model_loader import load_mamba_model_and_tokenizer
from datetime import datetime
import json
import numpy as np
from collections import defaultdict


def main():
    parser = argparse.ArgumentParser(
        description="Complete steering validation: discovery + validation (All-in-One)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="state-spaces/mamba-130m-hf",
        help="Model to evaluate (default: state-spaces/mamba-130m-hf)"
    )
    parser.add_argument(
        "--run_discovery",
        action="store_true",
        help="Run neuron discovery phase (Stage 1) - this is the default behavior"
    )
    parser.add_argument(
        "--validation_size",
        type=int,
        default=200,
        help="Number of validation examples per task (default: 200)"
    )
    parser.add_argument(
        "--test_size",
        type=int,
        default=200,
        help="Number of test examples per task (default: 200)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run on (default: cuda)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="ablation_3_results",
        help="Directory to save results (default: ablation_3_results)"
    )
    parser.add_argument(
        "--save_discovered",
        type=str,
        default=None,
        help="Optional: Path to save discovered neurons JSON file"
    )
    
    args = parser.parse_args()
    
    # Auto-correct model name: add state-spaces/ prefix and -hf suffix if needed
    model_name = args.model
    if 'mamba' in model_name.lower():
        # Add state-spaces/ prefix if not present
        if not model_name.startswith('state-spaces/'):
            model_name = f"state-spaces/{model_name}"
        # Add -hf suffix if not present
        if not model_name.endswith('-hf'):
            model_name = model_name + '-hf'
        if model_name != args.model:
            logger.info(f"Auto-correcting model name: {args.model} → {model_name}")
    
    # Load model and tokenizer
    logger.info(f"Loading model: {model_name}")
    try:
        model, tokenizer = load_mamba_model_and_tokenizer(
            model_name=model_name,
            device=args.device if torch.cuda.is_available() else "cpu",
            use_mamba_class=True,
            fallback_to_auto=True
        )
    except Exception as e:
        if "tiktoken" in str(e).lower():
            logger.error(f"Error: Model requires tiktoken package. Either:")
            logger.error(f"  1. Install tiktoken: pip install tiktoken")
            logger.error(f"  2. Use the -hf version: --model {args.model}-hf")
            sys.exit(1)
        else:
            raise
    
    device = next(model.parameters()).device
    
    # Set padding token if needed
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model.eval()
    
    # Generate tasks
    logger.info("Generating structured tasks...")
    generator = StructuredTaskGenerator(seed=42)
    validation_tasks = generator.generate_validation_set(
        size_per_task=args.validation_size // 4
    )
    test_tasks = generator.generate_test_set(
        size_per_task=args.test_size // 4
    )
    
    # Initialize validator
    validator = SteeringValidator(model, tokenizer, device)
    
    # Run discovery (Stage 1)
    logger.info("\n" + "="*80)
    logger.info("STAGE 1: NEURON DISCOVERY")
    logger.info("="*80)
    discovery_results, discovered_neurons = validator.run_comprehensive_neuron_discovery(validation_tasks)
    
    # Save discovered neurons if requested
    if args.save_discovered:
        save_path = Path(args.save_discovered)
        save_path.parent.mkdir(exist_ok=True, parents=True)
        save_discovered_neurons(discovery_results, discovered_neurons, save_path)
    else:
        # Default save location
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        default_save_path = output_dir / "discovered_neurons.json"
        save_discovered_neurons(discovery_results, discovered_neurons, default_save_path)
    
    # Run validation (Stage 2)
    logger.info("\n" + "="*80)
    logger.info("STAGE 2: VALIDATION OF DISCOVERED NEURONS")
    logger.info("="*80)
    results = validator.run_complete_validation(
        validation_tasks,
        test_tasks,
        discovered_neurons=discovered_neurons
    )
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Save results as JSON
    def json_serializable(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, torch.Tensor):
            return obj.cpu().numpy().tolist()
        elif isinstance(obj, defaultdict):
            return dict(obj)
        elif isinstance(obj, (list, dict, str, int, float, bool, type(None))):
            return obj
        else:
            return str(obj)
    
    results_path = output_dir / "steering_validation_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=json_serializable)
    
    # Save summary report
    summary_path = output_dir / "validation_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("STEERING VALIDATION PROTOCOL - RESULTS SUMMARY\n")
        f.write("="*80 + "\n\n")
        
        baseline_val = results['neuron_selection']['baseline']['validation']
        baseline_test = results['neuron_selection']['baseline']['test']
        best_config = results['summary']['best_config']
        best_test = results['strength_selection'][best_config['strength']]['test']
        
        f.write(f"MODEL: {args.model}\n")
        f.write(f"DATE: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("BEST CONFIGURATION:\n")
        f.write(f"  Neuron Selection: {best_config['method']}\n")
        f.write(f"  Layer: {best_config['layer']}\n")
        f.write(f"  Strength: {best_config['strength']}x\n")
        f.write(f"  Neurons: {best_config['neurons'][:5]}...\n\n")
        
        f.write("OVERALL RESULTS:\n")
        f.write(f"  Baseline Accuracy: {baseline_test['accuracy']*100:.1f}%\n")
        f.write(f"  Steered Accuracy: {best_test['accuracy']*100:.1f}%\n")
        f.write(f"  Improvement: +{(best_test['accuracy']-baseline_test['accuracy'])*100:.1f}%\n\n")
        
        f.write("PER-TASK IMPROVEMENTS:\n")
        for task_type, imp in results['summary']['improvements'].items():
            baseline = baseline_test['task_accuracies'][task_type]
            steered = best_test['task_accuracies'][task_type]
            f.write(f"  {task_type.replace('_', ' ').title():<20}: {baseline*100:>5.1f}% → {steered*100:>5.1f}% (+{imp:>5.1f}%)\n")
        
        f.write(f"\nTransfer Ratio (test/val): {results['summary']['transfer_ratio']:.2f}\n")
    
    logger.info(f"\nResults saved to: {output_dir}")
    logger.info(f"  Detailed results: {results_path}")
    logger.info(f"  Summary: {summary_path}")
    
    # Generate protocol summary
    try:
        generate_protocol_summary(results, output_dir)
        logger.info(f"  Protocol summary: {output_dir / 'protocol_summary.txt'}")
    except Exception as e:
        logger.warning(f"Could not generate protocol summary: {e}")
    
    # Create visualizations if importance ranking was computed
    if 'neuron_selection' in results and 'importance_ranking' in results['neuron_selection']:
        try:
            create_neuron_importance_visualization(results['neuron_selection']['importance_ranking'], output_dir)
            logger.info(f"  Visualization: {output_dir / 'neuron_importance_ranking.png'}")
        except Exception as e:
            logger.warning(f"Could not create visualization: {e}")


if __name__ == "__main__":
    main()

