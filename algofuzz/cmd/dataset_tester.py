from algofuzz.param_tester import MultivariateParamTester
from algofuzz.enums import DatasetType, FCMType, CentroidStrategy

def arange(start, stop, step):
    while start <= stop:
        yield start
        start += step

if __name__ == '__main__':
    #m = list(arange(1.2, 3, 0.2))
    #m.remove(1.0)
    #p = list(arange(1.1, 3, 0.2))
    m = [2]
    p = [2]

    mvpt = MultivariateParamTester({
        # 'type': [FCMType.FCM, FCMType.PFCM, FCMType.STPFCM],
        # 'dataset': [DatasetType.Iris, DatasetType.NormalizedIris, DatasetType.NoisyNormalizedIris],
        # 'num_clusters': [3],
        # 'max_iter': [150],
        # 'm': m,
        # # FCM
        # 'kappa': list(arange(1.1, 2, 0.2)),
        # # PFCM
        # 'p': p,
        # 'weight': [1,2,5,10,1.5,3], #list(arange(0.1, 2, 0.2)),
        # 'b': [1]#list(arange(0.1, 2, 0.2)),
        # # Proposed FCM
        # #'eta': [1e-9, 1e-7, 1e-5, 1e-3, 1e-1, 1]

        'type': [FCMType.NonoptimizedFPCM],
        'dataset': [DatasetType.NormalizedIris],
        'num_clusters': [3],
        'max_iter': [100],
        'noise': [None], #[10 ** t for t in arange(0, 3.02, 0.02)],
        'm': [2],
        'beta': list(arange(0, 1000, 1)),
        'p': [2],
        'centroid_strategy': [CentroidStrategy.NormalizedIrisDiagonal]
    })

    filename = 'test'
    values_to_export = ['m', 'p', 'beta', 'purity', 'nmi', 'ari']

    mvpt.fit()
    mvpt.write_csv(f'{filename}.csv')
    mvpt.write_mat(f'{filename}.mat', values_to_export)
    print('Total time:', mvpt.total_time)