import unittest
import numpy as np

import mltrainingtools.metaparameters as mt


INTEGER_DEF_1 = {
    'Integer1':
        {
            'base': 10,
            'range': 90,
            'default': 5,
            'type': 'integer'
        }
}

SMALL_FLOAT_DEF_1 = {
   'SmallFloat1':
       {
           'base': 1,
           'range': 4,
           'default': 0.0,
           'type': 'smallfloat'
       }
}


class MyTest(unittest.TestCase):
    def test_generate_integers(self):
        results = mt.generate_metaparameters(1, INTEGER_DEF_1)
        self.assertEqual(len(results['Integer1']), 1)
        self.assertGreater(results['Integer1'][0], 10)
        self.assertLess(results['Integer1'][0], 100)

    def test_generate_smallfloats(self):
        results = mt.generate_metaparameters(1, SMALL_FLOAT_DEF_1)
        self.assertEqual(len(results['SmallFloat1']), 1)
        self.assertGreater(results['SmallFloat1'][0], np.power(10, -5.))
        self.assertLess(results['SmallFloat1'][0], np.power(10, -1.))

    def test_generate_multiple_numbers(self):
        results = mt.generate_metaparameters(10, SMALL_FLOAT_DEF_1)
        self.assertEqual(len(results['SmallFloat1']), 10)

    def test_static_generation(self):
        results = mt.generate_metaparameters(15, INTEGER_DEF_1, static=True)
        self.assertEqual(len(results['Integer1']), 15)
        self.assertEqual(sum([e == 5 for e in results['Integer1']]), 15)

    def test_to_list(self):
        results = mt.generate_metaparameters(15, INTEGER_DEF_1, static=True)
        result_list = mt.to_list(results)
        self.assertEqual(len(result_list), 15)
        self.assertEqual(list(result_list[0].keys())[0], 'Integer1')
