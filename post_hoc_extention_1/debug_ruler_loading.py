"""
Quick debug script to test RULER loading - minimal dataset loading.
"""

import sys
import logging
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from dataset_classifier import DatasetQuestionClassifier

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


def test_ruler_loading_quick():
    """Quick test of RULER loading - minimal samples."""
    logger.info("=" * 80)
    logger.info("QUICK RULER LOADING TEST")
    logger.info("=" * 80)
    logger.info("")
    
    classifier = DatasetQuestionClassifier()
    
    # Test with minimal samples (5) for speed
    logger.info("Testing RULER loading with 5 samples...")
    logger.info("")
    
    try:
        result = classifier.load_ruler(5)
        
        # Check results
        total_questions = sum(len(questions) for questions in result.values())
        
        if total_questions > 0:
            logger.info(f"✅ SUCCESS: Loaded {total_questions} questions")
            for category, questions in result.items():
                if questions:
                    logger.info(f"   - {category}: {len(questions)} questions")
                    # Show sample
                    sample = questions[0]
                    logger.info(f"     Sample source: {sample.get('source', 'unknown')}")
                    logger.info(f"     Question preview: {sample.get('question', '')[:80]}...")
            return True
        else:
            logger.warning("⚠️  EMPTY: No questions loaded")
            logger.info("   This is EXPECTED - RULER is not on HuggingFace")
            logger.info("   RULER is a GitHub repository: https://github.com/NVIDIA/RULER")
            logger.info("   Natural Questions will be used as fallback for long_context")
            return True  # This is expected, not an error
            
    except Exception as e:
        logger.error(f"❌ ERROR: {str(e)}")
        import traceback
        logger.error(f"   Traceback:\n{traceback.format_exc()}")
        return False


def main():
    """Run quick RULER test."""
    logger.info("Quick RULER loading test - checking for errors...")
    logger.info("")
    
    success = test_ruler_loading_quick()
    
    logger.info("")
    logger.info("=" * 80)
    if success:
        logger.info("✅ TEST COMPLETE: No errors in RULER loading")
        logger.info("   (Empty result is expected - RULER not on HuggingFace)")
    else:
        logger.error("❌ TEST FAILED: Errors found in RULER loading")
    logger.info("=" * 80)
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
