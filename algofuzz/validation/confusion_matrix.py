from itertools import permutations
import numpy as np

def find_best_permutation(confusion: np.ndarray) -> np.ndarray:
    """
    Find the best permutation of classes for the confusion matrix.

    Given a confusion matrix, this function generates all possible permutations
    of class labels and calculates the accuracy for each permutation based on
    the diagonal elements of the permuted confusion matrix. It returns the best
    permuted confusion matrix that yields the highest accuracy.

    Parameters:
        confusion (np.ndarray): The confusion matrix of shape (c, c), where c
            is the number of classes. The confusion matrix represents the
            predicted class labels against the true class labels.

    Returns:
        np.ndarray: The best permuted confusion matrix of shape (c, c),
            representing the confusion matrix after the best class label
            permutation is applied.

    Example:
        # Assuming 'confusion' is a 3x3 numpy array representing the confusion matrix.
        # This will return the confusion matrix with the best class label permutation.
        best_permuted_confusion = find_best_permutation(confusion_matrix)
    """
    c = confusion.shape[0]
    best_accuracy = 0
    best_permuted_confusion = None
    
    for p in permutations(range(c)):
        permuted_confusion = np.array([confusion[idx] for idx in p])
        accuracy = np.sum(np.diag(permuted_confusion)) / np.sum(permuted_confusion)
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_permuted_confusion = permuted_confusion
            
    return best_permuted_confusion

if __name__ == '__main__':
    true_labels = [0] * 150 + [1] * 150 + [2] * 150
    predicted_labels = [1] * 150 + [2] * 150 + [0] * 150
    predicted_labels2 = [1] * 2 + [2] * 148 + [1] * 150 + [0] * 150

    from sklearn.metrics import confusion_matrix
    conf_matrix = confusion_matrix(true_labels, predicted_labels2[:len(true_labels)])
    print("Confusion Matrix:")
    print(conf_matrix)
    best_permuted_confusion, best_accuracy = find_best_permutation(conf_matrix)
    print("Best permutationed confusion matrix:")
    print(best_permuted_confusion)

