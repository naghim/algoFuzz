from algofuzz.fcm.nonoptimized_fp3cm import NonoptimizedFP3CM
from algofuzz.fcm.proposed_fcm import STPFCM
from algofuzz.fcm.nonoptimized_fpcm import NonoptimizedFPCM
from algofuzz.fcm.possibilistic_fcm import PFCM
from algofuzz.fcm.fcm import FCM
from algofuzz.fcm.fcplus1m import FCPlus1M
from algofuzz.validation.confusion_matrix import find_best_permutation
from algofuzz.validation.validity_index import purity, normalized_mutual_information, adjusted_rand_index
from algofuzz.enums import CentroidStrategy, DatasetType, FCMType
from algofuzz.util import load_dataset
from sklearn.metrics import confusion_matrix
import numpy as np

def test_kappa(dataset: DatasetType):
    np.random.seed(0)

    X, c, true_labels = load_dataset(dataset)
    m = 2  # Fuzzifier parameter
    p = 2  # Exponent parameter
    eta = 0.1  # Learning rate parameter
    v0 = np.random.rand(X.shape[0], c)  # Initial cluster centroids
    a = 1.0  # Weighting factor for the membership
    steps = 10  # Number of iterations
    preprocess_iter = 15 # Number of iterations for Pal FCM preprocessing
    b = 1.0 # Weighting factor for the distance

    PUR = []
    ARI = []
    NMI = []
    kappa_values = list(range(1, 35, 1))

    for kappa in kappa_values:
        fcm = STPFCM(
            num_clusters=c,
            m=m,
            p=p,
            eta=eta,
            centroids=v0,
            weight=a,
            max_iter=steps,
            kappa=kappa
        )
        fcm.fit(X)
        conf_matrix = confusion_matrix(true_labels, fcm.labels[:len(true_labels)])
        best_permuted_confusion = find_best_permutation(conf_matrix)

        PURR = purity(best_permuted_confusion)
        ARII = adjusted_rand_index(best_permuted_confusion)
        NMII = normalized_mutual_information(best_permuted_confusion)

        PUR.append(PURR)
        ARI.append(ARII)
        NMI.append(NMII)

    print("ARI:", ARI)
    print("NMI:", NMI)
    print("PUR:", PUR)
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))  # Adjust the size as needed
    plt.plot(kappa_values, PUR, marker='o', label='PUR')
    plt.plot(kappa_values, ARI, marker='o', label='ARI')
    plt.plot(kappa_values, NMI, marker='o', label='NMI')

    # Add labels and title
    plt.xlabel('Kappa')
    plt.ylabel('Score')
    plt.title('Comparison of PUR, ARI, and NMI Scores')
    plt.legend()
    # Show the plot
    plt.grid(True)
    plt.axis('equal')
    plt.show()

def main(dataset: DatasetType, fcm: FCMType, centroid_strategy: CentroidStrategy):
    # np.random.seed(0)

    X, c, true_labels = load_dataset(dataset)
    m = 2 # Fuzzifier parameter
    p = 2  # Exponent parameter
    eta = 2.0  # Penalty term parameter
    a = 1.0  # Weighting factor for the membership
    steps = 150  # Number of iterations
    preprocess_iter = 15 # Number of iterations for Pal FCM preprocessing
    b = 1.0 # Weighting factor for the distance
    beta = 100
    noise = 40

    if fcm == FCMType.STPFCM:
        fcm = STPFCM(
            num_clusters=c,
            m=m,
            p=p,
            eta=eta,
            weight=a,
            max_iter=steps,
            centroid_strategy=centroid_strategy
        )
    elif fcm == FCMType.NonoptimizedFPCM:
        fcm = NonoptimizedFPCM(
            num_clusters=c,
            m=m,
            p=p,
            beta=beta,
            max_iter=steps,
            noise=noise,
            centroid_strategy=centroid_strategy
        )
    elif fcm == FCMType.FCM:
        fcm = FCM(
            num_clusters=c,
            m=m,
            max_iter=steps,
            noise=noise,
            centroid_strategy=centroid_strategy
        )
    elif fcm == FCMType.FCPlus1M:
        fcm = FCPlus1M(
            num_clusters=c,
            m=m,
            max_iter=steps,
            noise=noise,
            centroid_strategy=centroid_strategy
        )
    elif fcm == FCMType.PFCM:
        fcm = PFCM(
            num_clusters=c,
            m=m,
            max_iter=steps,
            preprocess_iter=preprocess_iter,
            p=p,
            a=a,
            b=b,
            centroid_strategy=centroid_strategy
        )
    elif fcm == FCMType.NonoptimizedFP3CM:
        fcm = NonoptimizedFP3CM(
            num_clusters=c,
            m=m,
            max_iter=steps,
            p=p,
            eta=eta,
            centroid_strategy=centroid_strategy
        )
    else:
        raise ValueError('Invalid FCM type')

    fcm.fit(X)

    # Print the results
    # print()
    #print("Membership matrix:")
    #print(fcm.member)

    # if hasattr(fcm, 'alpha'):
    #     print()
    #     print("Alpha values:")
    #     print(fcm.alpha)

    # print()
    # print("Final eta values:")
    # print(fcm.cluster_eta)
    # print()
    #print("Labels:")
    #print(fcm.labels)
    #print()

    #print('fcm shape', fcm.labels.shape)
    conf_matrix = confusion_matrix(true_labels, fcm.labels[:len(true_labels)])
    best_permuted_confusion = find_best_permutation(conf_matrix)
    print(best_permuted_confusion)
    print(np.sum(np.diag(best_permuted_confusion)))

    PUR = purity(best_permuted_confusion)
    ARI = adjusted_rand_index(best_permuted_confusion)
    NMI = normalized_mutual_information(best_permuted_confusion)

    # print("ARI:", ARI)
    # print("NMI:", NMI)
    # print("PUR:", PUR)
    # print("Final eta values:")
    # print(fcm.cluster_eta)
    # print()
    #fcm.plot_clusters(X)

    fcm.evaluate(true_labels)

if __name__ == '__main__':
    strategy=CentroidStrategy.Random
    #main(DatasetType.NormalizedBreastCancer, FCMType.FCM, strategy)
    #main(DatasetType.Iris, FCMType.PFCM, strategy)
    #main(DatasetType.NormalizedBreastCancer, FCMType.NonoptimizedFPCM, strategy)
    #main(DatasetType.NormalizedBreastCancer, FCMType.NonoptimizedFPCM, strategy)
    main(DatasetType.Glass, FCMType.FCPlus1M, strategy)
    # main(DatasetType.Iris, FCMType.PFCM, strategy)
    # main(DatasetType.Iris, FCMType.STPFCM, strategy)
    # main(DatasetType.Iris, FCMType.NonoptimizedFP3CM, strategy)
