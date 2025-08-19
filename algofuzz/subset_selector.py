from numpy.typing import NDArray
import numpy as np

def select_subset(X: NDArray, true_labels: NDArray, percentage: float):
    """
    Select a subset of the dataset based on the given percentage.

    If true labels are provided, the subset will maintain the class distribution.
    """
    if percentage >= 1:
        # Percentage is 100%: return all samples
        return X, true_labels
    elif true_labels is not None:
        labels = np.array(true_labels)
        classes, counts = np.unique(labels, return_counts=True)

        # prepare selection: percentage per class with same distribution
        rng = np.random.default_rng()
        per_class = np.maximum(1, (counts * percentage).astype(int))  # at least 1 per class

        selected = []
        for cls, k in zip(classes, per_class):
            idx = np.where(labels == cls)[0]
            chosen = rng.choice(idx, size=int(k), replace=False)
            selected.append(chosen)

        eval_idx = np.concatenate(selected).astype(int)
        rng.shuffle(eval_idx)

        # samples are columns in X (shape (features, samples))
        small_X = X[:, eval_idx]
        small_true_labels = labels[eval_idx]

        return small_X, small_true_labels
    else:
        # No true labels available: pick percentage% of samples uniformly at random
        rng = np.random.default_rng()
        n_samples = X.shape[1]  # samples are columns
        k = max(1, int(np.floor(percentage * n_samples)))
        eval_idx = rng.choice(n_samples, size=k, replace=False).astype(int)
        rng.shuffle(eval_idx)

        small_X = X[:, eval_idx]
        small_true_labels = None

        return small_X, small_true_labels
