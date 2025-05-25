import unittest
import sys
import os

def run_tests():
    # Get the directory containing test files
    test_dir = os.path.join(os.path.dirname(__file__), 'tests')
    
    # Discover and run tests
    loader = unittest.TestLoader()
    suite = loader.discover(test_dir, pattern='test_*.py')
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Return 0 if tests passed, 1 if any failed
    return 0 if result.wasSuccessful() else 1

if __name__ == '__main__':
    sys.exit(run_tests())