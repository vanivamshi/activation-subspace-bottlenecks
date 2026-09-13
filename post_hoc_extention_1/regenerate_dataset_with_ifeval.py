"""
Script to regenerate the classified dataset file with IFEval included.
RULER is not available on HuggingFace, so we'll use Natural Questions as fallback for long_context.
"""

import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from dataset_classifier import create_classified_dataset_file
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Regenerate the classified dataset file with IFEval."""
    logger.info("=" * 80)
    logger.info("REGENERATING CLASSIFIED DATASET FILE WITH IFEVAL")
    logger.info("=" * 80)
    logger.info("")
    logger.info("Note: RULER is not available on HuggingFace.")
    logger.info("      It's a GitHub repository: https://github.com/NVIDIA/RULER")
    logger.info("      Using Natural Questions as fallback for long_context.")
    logger.info("")
    
    # Create dataset file with IFEval included
    # RULER will fail but that's okay - we'll use Natural Questions for long_context
    datasets = [
        'squad',           # Simple recall, long context
        'hotpotqa',        # Two-hop, three-hop
        'triviaqa',        # Query dataset tasks
        'musique',         # Two-hop, three-hop
        'drop',            # Combined reasoning
        'natural_questions', # Long context (fallback for RULER), stress test
        'ifeval'           # Stress test (primary)
    ]
    
    output_file = "experiment_logs/classified_dataset_questions_new_dataset.json"
    
    logger.info(f"Creating dataset file: {output_file}")
    logger.info(f"Datasets: {', '.join(datasets)}")
    logger.info("")
    
    try:
        classified = create_classified_dataset_file(
            datasets=datasets,
            num_per_dataset=100,
            target_per_category=100,
            output_file=output_file
        )
        
        # Print summary
        logger.info("")
        logger.info("=" * 80)
        logger.info("SUMMARY")
        logger.info("=" * 80)
        for category, questions in classified.items():
            if questions:
                sources = {}
                for q in questions:
                    source = q.get('source', 'unknown')
                    sources[source] = sources.get(source, 0) + 1
                logger.info(f"  {category}: {len(questions)} questions")
                logger.info(f"    Sources: {sources}")
            else:
                logger.warning(f"  {category}: 0 questions (EMPTY)")
        
        logger.info("")
        logger.info(f"✅ Dataset file created: {output_file}")
        logger.info("")
        logger.info("Next steps:")
        logger.info(f"  1. Use this file: --dataset_file {output_file}")
        logger.info("  2. For RULER: Clone https://github.com/NVIDIA/RULER and load manually")
        
    except Exception as e:
        logger.error(f"❌ Error creating dataset file: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())


