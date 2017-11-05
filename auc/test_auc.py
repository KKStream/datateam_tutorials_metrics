"""
"""
import unittest

from six.moves import range

import auc


class TestAuc(unittest.TestCase):
    """
    """
    def test_perfect_feature(self):
        """
        If there exists a threshold q which:
            * all items with probabilities less than q are negative.
            * all items with probabilities greater than q are positive.
        Then we have found a perfect feature that can split entire dataset into
        2 groups (true positive and true negative).

        AUC for this case should be 1.0.

        guess: [0.2, 0.4, 0.6, 0.8, 1.0]
        truth: [  F,   F,   T,   T,   T]
        """
        guess = [x / 10.0 for x in range(10)]
        truth = [0 if x < 5 else 1 for x in range(10)]

        value = auc.auc(guess, truth)

        self.assertTrue(value == 1.0)

    def test_worst_feature(self):
        """
        If there exists a threshold q which:
            * all items with probabilities less than q are positive.
            * all items with probabilities greater than q are negative.
        Then we have found a worst feature that can split entire dataset into
        2 groups (false positive and false negative).

        AUC for this case should be 0.0.

        guess: [0.2, 0.4, 0.6, 0.8, 1.0]
        truth: [  T,   T,   T,   F,   F]
        """
        guess = [x / 10.0 for x in range(10)]
        truth = [0 if x > 5 else 1 for x in range(10)]

        value = auc.auc(guess, truth)

        self.assertTrue(value == 0.0)

    def test_tutorial_sample(self):
        """
        """
        guess = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        truth = [0, 0, 1, 0, 0, 1, 0, 1, 1, 1]

        value = auc.auc(guess, truth)

        self.assertTrue(abs(value - 0.84) < 1e-5)
