from numpy.typing import NDArray
from sklearn.metrics import confusion_matrix, davies_bouldin_score, silhouette_score

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

def evaluate_inner_metrics(X: NDArray, labels: NDArray) -> tuple[float, float]:
    """
    Evaluate the clustering results using inner metrics.

    Parameters:
        X (NDArray): The input data.
        labels (NDArray): The predicted labels of the dataset.

    Returns:
        tuple[float, float]: A tuple containing the following evaluation metrics:
            - Davies-Bouldin Index (minimum 0, no upper bound)
            - Silhouette Score (normalized to 0-1)
    """
    db_index = davies_bouldin_score(X.T, labels)
    silhouette = silhouette_score(X.T, labels)

    # Normalize Silhouette Score from [-1, 1] to [0, 1]
    normalized_silhouette = (silhouette + 1) / 2

    # Davies-Bouldin Index is 0 to infinity, lower is better.
    return db_index, normalized_silhouette