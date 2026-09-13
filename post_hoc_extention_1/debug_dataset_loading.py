"""
Debug script to check if all datasets are loaded properly.
Tests all dataset loaders and reports their status.
"""

import sys
import logging
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from dataset_classifier import DatasetQuestionClassifier
from query_dataset_loader import QueryDatasetLoader

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


def test_dataset_classifier():
    """Test DatasetQuestionClassifier for all datasets."""
    logger.info("=" * 80)
    logger.info("TESTING DATASET CLASSIFIER")
    logger.info("=" * 80)
    
    classifier = DatasetQuestionClassifier()
    
    datasets_to_test = [
        ('squad', 'load_squad', 10),
        ('hotpotqa', 'load_hotpotqa', 10),
        ('triviaqa', 'load_triviaqa', 10),
        ('musique', 'load_musique', 10),
        ('drop', 'load_drop', 10),
        ('natural_questions', 'load_natural_questions', 10),
        ('ruler', 'load_ruler', 10),
        ('ifeval', 'load_ifeval', 10),
    ]
    
    results = {}
    
    for dataset_name, method_name, num_samples in datasets_to_test:
        logger.info(f"\n{'='*80}")
        logger.info(f"Testing {dataset_name.upper()}")
        logger.info(f"{'='*80}")
        
        try:
            method = getattr(classifier, method_name)
            result = method(num_samples)
            
            # Count questions per category
            total_questions = sum(len(questions) for questions in result.values())
            
            if total_questions > 0:
                logger.info(f"✅ {dataset_name}: SUCCESS - Loaded {total_questions} questions")
                for category, questions in result.items():
                    if questions:
                        logger.info(f"   - {category}: {len(questions)} questions")
                results[dataset_name] = {
                    'status': 'SUCCESS',
                    'total': total_questions,
                    'categories': {k: len(v) for k, v in result.items() if v}
                }
            else:
                logger.warning(f"⚠️  {dataset_name}: LOADED BUT EMPTY - No questions found")
                results[dataset_name] = {
                    'status': 'EMPTY',
                    'total': 0,
                    'categories': {}
                }
                
        except Exception as e:
            logger.error(f"❌ {dataset_name}: FAILED - {str(e)}")
            results[dataset_name] = {
                'status': 'FAILED',
                'error': str(e),
                'total': 0
            }
    
    return results


def test_query_dataset_loader():
    """Test QueryDatasetLoader for query datasets."""
    logger.info("\n" + "=" * 80)
    logger.info("TESTING QUERY DATASET LOADER")
    logger.info("=" * 80)
    
    loader = QueryDatasetLoader()
    
    datasets_to_test = [
        ('squad', 'load_squad_queries', 10),
        ('natural_questions', 'load_natural_questions', 10),
        ('triviaqa', 'load_triviaqa', 10),
    ]
    
    results = {}
    
    for dataset_name, method_name, num_samples in datasets_to_test:
        logger.info(f"\n{'='*80}")
        logger.info(f"Testing {dataset_name.upper()} queries")
        logger.info(f"{'='*80}")
        
        try:
            method = getattr(loader, method_name)
            result = method(num_samples)
            
            if result:
                logger.info(f"✅ {dataset_name}: SUCCESS - Loaded {len(result)} queries")
                # Show sample
                if result:
                    sample = result[0]
                    logger.info(f"   Sample keys: {list(sample.keys())}")
                    logger.info(f"   Sample prompt (first 100 chars): {sample.get('prompt', '')[:100]}...")
                results[dataset_name] = {
                    'status': 'SUCCESS',
                    'count': len(result)
                }
            else:
                logger.warning(f"⚠️  {dataset_name}: LOADED BUT EMPTY - No queries found")
                results[dataset_name] = {
                    'status': 'EMPTY',
                    'count': 0
                }
                
        except Exception as e:
            logger.error(f"❌ {dataset_name}: FAILED - {str(e)}")
            import traceback
            logger.error(f"   Traceback: {traceback.format_exc()}")
            results[dataset_name] = {
                'status': 'FAILED',
                'error': str(e),
                'count': 0
            }
    
    return results


def test_classify_all_datasets():
    """Test the classify_all_datasets method."""
    logger.info("\n" + "=" * 80)
    logger.info("TESTING CLASSIFY_ALL_DATASETS")
    logger.info("=" * 80)
    
    classifier = DatasetQuestionClassifier()
    
    datasets_to_test = [
        ['squad', 'hotpotqa', 'triviaqa', 'musique', 'drop', 'natural_questions'],
        ['squad', 'hotpotqa', 'triviaqa', 'musique', 'drop', 'natural_questions', 'ruler', 'ifeval'],
    ]
    
    for dataset_list in datasets_to_test:
        logger.info(f"\n{'='*80}")
        logger.info(f"Testing with datasets: {', '.join(dataset_list)}")
        logger.info(f"{'='*80}")
        
        try:
            result = classifier.classify_all_datasets(
                datasets=dataset_list,
                num_per_dataset=20,
                target_per_category=20
            )
            
            total_questions = sum(len(questions) for questions in result.values())
            logger.info(f"✅ classify_all_datasets: SUCCESS - Total {total_questions} questions")
            
            for category, questions in result.items():
                if questions:
                    sources = {}
                    for q in questions:
                        source = q.get('source', 'unknown')
                        sources[source] = sources.get(source, 0) + 1
                    logger.info(f"   - {category}: {len(questions)} questions")
                    logger.info(f"     Sources: {sources}")
                else:
                    logger.warning(f"   - {category}: 0 questions (EMPTY)")
                    
        except Exception as e:
            logger.error(f"❌ classify_all_datasets: FAILED - {str(e)}")
            import traceback
            logger.error(f"   Traceback: {traceback.format_exc()}")


def test_get_query_tasks():
    """Test the get_query_tasks method."""
    logger.info("\n" + "=" * 80)
    logger.info("TESTING GET_QUERY_TASKS")
    logger.info("=" * 80)
    
    loader = QueryDatasetLoader()
    
    datasets_to_test = [
        ['squad'],
        ['squad', 'natural_questions'],
        ['squad', 'natural_questions', 'triviaqa'],
    ]
    
    for dataset_list in datasets_to_test:
        logger.info(f"\n{'='*80}")
        logger.info(f"Testing with datasets: {', '.join(dataset_list)}")
        logger.info(f"{'='*80}")
        
        try:
            result = loader.get_query_tasks(
                datasets=dataset_list,
                num_per_dataset=10
            )
            
            total_queries = sum(len(queries) for queries in result.values())
            logger.info(f"✅ get_query_tasks: SUCCESS - Total {total_queries} queries")
            
            for dataset_name, queries in result.items():
                logger.info(f"   - {dataset_name}: {len(queries)} queries")
                
        except Exception as e:
            logger.error(f"❌ get_query_tasks: FAILED - {str(e)}")
            import traceback
            logger.error(f"   Traceback: {traceback.format_exc()}")


def print_summary(classifier_results, query_loader_results):
    """Print a summary of all test results."""
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    
    logger.info("\n📊 Dataset Classifier Results:")
    success_count = sum(1 for r in classifier_results.values() if r['status'] == 'SUCCESS')
    empty_count = sum(1 for r in classifier_results.values() if r['status'] == 'EMPTY')
    failed_count = sum(1 for r in classifier_results.values() if r['status'] == 'FAILED')
    
    logger.info(f"   ✅ Success: {success_count}/{len(classifier_results)}")
    logger.info(f"   ⚠️  Empty: {empty_count}/{len(classifier_results)}")
    logger.info(f"   ❌ Failed: {failed_count}/{len(classifier_results)}")
    
    logger.info("\n📊 Query Dataset Loader Results:")
    success_count = sum(1 for r in query_loader_results.values() if r['status'] == 'SUCCESS')
    empty_count = sum(1 for r in query_loader_results.values() if r['status'] == 'EMPTY')
    failed_count = sum(1 for r in query_loader_results.values() if r['status'] == 'FAILED')
    
    logger.info(f"   ✅ Success: {success_count}/{len(query_loader_results)}")
    logger.info(f"   ⚠️  Empty: {empty_count}/{len(query_loader_results)}")
    logger.info(f"   ❌ Failed: {failed_count}/{len(query_loader_results)}")
    
    logger.info("\n🔍 Detailed Results:")
    logger.info("\nDataset Classifier:")
    for dataset_name, result in classifier_results.items():
        status_icon = "✅" if result['status'] == 'SUCCESS' else "⚠️" if result['status'] == 'EMPTY' else "❌"
        logger.info(f"   {status_icon} {dataset_name}: {result['status']}")
        if result['status'] == 'SUCCESS':
            logger.info(f"      Total: {result['total']} questions")
            if 'categories' in result:
                for cat, count in result['categories'].items():
                    logger.info(f"      - {cat}: {count}")
        elif result['status'] == 'FAILED':
            logger.info(f"      Error: {result.get('error', 'Unknown error')[:100]}")
    
    logger.info("\nQuery Dataset Loader:")
    for dataset_name, result in query_loader_results.items():
        status_icon = "✅" if result['status'] == 'SUCCESS' else "⚠️" if result['status'] == 'EMPTY' else "❌"
        logger.info(f"   {status_icon} {dataset_name}: {result['status']}")
        if result['status'] == 'SUCCESS':
            logger.info(f"      Count: {result['count']} queries")
        elif result['status'] == 'FAILED':
            logger.info(f"      Error: {result.get('error', 'Unknown error')[:100]}")


def main():
    """Run all tests."""
    logger.info("=" * 80)
    logger.info("DATASET LOADING DEBUG SCRIPT")
    logger.info("=" * 80)
    logger.info("This script tests all dataset loaders to verify they work correctly.")
    logger.info("=" * 80)
    
    # Test individual dataset loaders
    classifier_results = test_dataset_classifier()
    query_loader_results = test_query_dataset_loader()
    
    # Test combined methods
    test_classify_all_datasets()
    test_get_query_tasks()
    
    # Print summary
    print_summary(classifier_results, query_loader_results)
    
    logger.info("\n" + "=" * 80)
    logger.info("DEBUG COMPLETE")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()


