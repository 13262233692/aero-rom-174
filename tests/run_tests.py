"""
运行所有单元测试
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

test_dir = os.path.dirname(__file__)

loader = unittest.TestLoader()
suite = loader.discover(test_dir, pattern='test_*.py')

runner = unittest.TextTestRunner(verbosity=2)
result = runner.run(suite)

sys.exit(0 if result.wasSuccessful() else 1)
