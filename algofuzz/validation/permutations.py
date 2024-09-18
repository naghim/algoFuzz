from itertools import permutations
import numpy as np

def find_best_permutation(confusion):
    c = confusion.shape[0]
    best_accuracy = 0
    best_permuted_confusion = None

    for p in permutations(range(c)):
        permuted_confusion = np.array([confusion[idx] for idx in p])
        accuracy = np.sum(np.diag(permuted_confusion)) / np.sum(permuted_confusion)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_permuted_confusion = permuted_confusion

    return best_permuted_confusion, best_accuracy

if __name__ == '__main__':
    # Test usage:
    confusion_matrix_standard = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
    best_permuted_confusion, best_accuracy = find_best_permutation(confusion_matrix_standard)
    print("Best permutationed confusion matrix:")
    print(best_permuted_confusion)
    print("Best accuracy:", best_accuracy)