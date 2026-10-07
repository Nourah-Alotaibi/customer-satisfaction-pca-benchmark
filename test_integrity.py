import unittest
import numpy as np, pandas as pd
from train import split_indices


class SplitIntegrity(unittest.TestCase):
    def test_duplicate_predictors_with_conflicting_labels_stay_together(self):
        x = pd.DataFrame(
            {"a": np.repeat(np.arange(200), 2), "b": np.repeat(np.arange(200) ** 2, 2)}
        )
        y = pd.Series(np.tile([0, 1], 200))
        dev, tr, va, te, groups = split_indices(x, y)
        self.assertEqual(set(np.r_[tr, va, te]), set(range(len(x))))
        for a, b in [(tr, va), (tr, te), (va, te)]:
            self.assertFalse(set(groups[a]) & set(groups[b]))
        self.assertEqual(set(dev), set(np.r_[tr, va]))


if __name__ == "__main__":
    unittest.main()
