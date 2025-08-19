from numpy.typing import NDArray
from sklearn.metrics import confusion_matrix

from algofuzz.validation.confusion_matrix import find_best_permutation
from algofuzz.validation.validity_index import adjusted_rand_index, normalized_mutual_information, purity

import logging

def evaluate_true_labels(predicted_labels: NDArray, true_labels: NDArray) -> list[float]:
    """
    Evaluate the clustering results. Currently uses the true labels of the dataset to perform the evaluations.

    Parameters:
        predicted_labels (np.ndarray): The predicted labels of the dataset.
        true_labels (np.ndarray): The true labels of the dataset.

    Returns:
        list[float]: A list containing the following evaluation metrics:
            - Purity
            - Adjusted Rand Index
            - Normalized Mutual Information
    """
    if len(predicted_labels) != len(true_labels):
        logging.warning("Length of predicted labels does not match length of true labels.")

    conf_matrix = confusion_matrix(true_labels, predicted_labels[:len(true_labels)])
    best_permuted_confusion = find_best_permutation(conf_matrix)

    pur = purity(best_permuted_confusion)
    ari = adjusted_rand_index(best_permuted_confusion)
    nmi = normalized_mutual_information(best_permuted_confusion)

    return pur, ari, nmi