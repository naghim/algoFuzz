import numpy as np

def purity(confusion_matrix: np.ndarray) -> float:
    """
    Calculate the purity metric from the confusion matrix.

    Parameters:
        confusion_matrix (np.ndarray): The confusion matrix of shape (c, c).

    Returns:
        float: The purity value.
    """
    return np.sum(np.diag(confusion_matrix)) / np.sum(confusion_matrix)

def adjusted_rand_index(confusion_matrix: np.ndarray) -> float:
    """
    Calculate the adjusted Rand index from the confusion matrix.

    Parameters:
        confusion_matrix (np.ndarray): The confusion matrix of shape (c, c).

    Returns:
        float: The adjusted Rand index value.
    """
    c = confusion_matrix.shape[0]
    row_sum = np.sum(confusion_matrix, axis=1)
    col_sum = np.sum(confusion_matrix, axis=0)
    n = np.sum(confusion_matrix)
    
    positives = 0  
    sum_a = 0  
    sum_b = 0 
    
    for i in range(c):
        sum_a += row_sum[i] * (row_sum[i] - 1) / 2
        sum_b += col_sum[i] * (col_sum[i] - 1) / 2
        for j in range(c):
            positives += confusion_matrix[i, j] * (confusion_matrix[i, j] - 1) / 2
            
    avg_ab = (sum_a + sum_b) / 2
    expected_rand_index = sum_a * sum_b * 2 / (n * (n - 1))
    
    return (positives - expected_rand_index) / (avg_ab - expected_rand_index)

def normalized_mutual_information(confusion_matrix: np.ndarray) -> float:
    """
    Calculate the normalized mutual information from the confusion matrix.

    Parameters:
        confusion_matrix (np.ndarray): The confusion matrix of shape (c, c).

    Returns:
        float: The normalized mutual information value.
    """
    c = confusion_matrix.shape[0]
    row_sum = np.sum(confusion_matrix, axis=1)
    col_sum = np.sum(confusion_matrix, axis=0)
    n = np.sum(confusion_matrix)

    entropy_true_labels = 0  # Entropy of true labels
    entropy_predicted_label = 0  # Entropy of predicted labels
    conditional_entorpy_of_labels = 0  # Conditional entropy of true labels given predicted labels
    
    for i in range(c):
        if row_sum[i] > 0:
            entropy_true_labels -= (row_sum[i] / n) * np.log2(row_sum[i] / n)
        if col_sum[i] > 0:
            entropy_predicted_label -= (col_sum[i] / n) * np.log2(col_sum[i] / n)
        for j in range(c):
            if confusion_matrix[j, i] > 0:
                conditional_entorpy_of_labels = conditional_entorpy_of_labels - (col_sum[i] / n) * (confusion_matrix[j, i] / col_sum[i]) * np.log2(confusion_matrix[j, i] / col_sum[i])
                
    return 2 * (entropy_true_labels - conditional_entorpy_of_labels) / (entropy_true_labels + entropy_predicted_label)
