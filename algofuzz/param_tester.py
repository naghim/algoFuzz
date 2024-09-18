from algofuzz.util import get_fcm_by_type, load_dataset
from scipy.io import savemat
from itertools import product
from p_tqdm import p_map
from tqdm import tqdm
import numpy as np
import time
import csv
import traceback
import random

class MultivariateParamTesterException(Exception):
    pass

class MultivariateParamTester(object):
    """
    MultivariateParamTester is a class that performs
    grid parameter testing on FCMs.

    It leverages multiprocessing to speed up the process.

    param_grid: dict[str, list]
        A dictionary containing the parameters to be searched. The keys are the parameter names, and the values are the list of values to be searched
    """
    def __init__(self, param_grid: dict[str, list]):
        self.param_grid = param_grid
        self.headers = None
        self.values = None
        self.errors = None
        self.dataset_cache = None
        self.already_done = None
        self.in_queue = None

    def queue_options(self) -> None:
        parameter_names = list(self.param_grid.keys())
        literal_fields = ['trained']

        type_index = parameter_names.index('type')
        dataset_index = parameter_names.index('dataset')
        num_cluster_index = parameter_names.index('num_clusters')

        try:
            random_seed_index = parameter_names.index('seed')
        except:
            random_seed_index = -1

        self.dataset_cache = {}
        self.already_done = set()

        print('Generating options')

        self.headers = ['index'] + parameter_names + ['eta', 'time', 'purity', 'nmi', 'ari']
        process_queue = []
        i = 0

        for item in tqdm(product(*self.param_grid.values())):
            item = list(item)

            # Load FCM
            fcm_type = item[type_index]
            fcm_class = get_fcm_by_type(fcm_type)

            if fcm_class is None:
                raise MultivariateParamTesterException('Invalid FCM type {}'.format(fcm_type))

            # Load random seed
            if random_seed_index == -1:
                random_seed = -1
            else:
                random_seed = item[random_seed_index]

            # Preload datasets
            dataset_name = item[dataset_index]
            dataset_key = (dataset_name, random_seed)

            if dataset_key in self.dataset_cache:
                dataset = self.dataset_cache[dataset_key]
            else:
                if random_seed != -1:
                    np.random.seed(random_seed)

                dataset = load_dataset(dataset_name, num_clusters=item[num_cluster_index])
                self.dataset_cache[dataset_key] = dataset

            # Extract fields from FCM class
            fields = fcm_class.__fields__.keys() - literal_fields

            # Update cluster number of item
            X, num_clusters, true_labels = dataset
            item[num_cluster_index] = num_clusters

            # Extract arguments
            kwargs = [(parameter_names[i], field) for i, field in enumerate(item)]

            # Actual arguments are the arguments that really matter for this
            # FCM type. Irrelevant parameters will be removed
            actual_kwargs = [kwarg for kwarg in kwargs if kwarg[0] in fields]
            actual_kwargs = dict(actual_kwargs)

            #if 'm' in actual_kwargs and 'p' in actual_kwargs and actual_kwargs['m'] > actual_kwargs['p']:
            #    continue

            # Create hash of actual parameters to check if the parameter
            # changes are relevant
            already_done_dict = {'type': fcm_type.value, 'dataset': dataset_name.value, 'seed': random_seed}
            already_done_dict.update(actual_kwargs)
            already_done_key = hash(frozenset(already_done_dict.items()))

            # Don't repeat test if parameters are irrelevant
            if already_done_key in self.already_done:
                print('already done?')
                continue

            self.already_done.add(already_done_key)

            # Queue task
            process_queue.append([i, fcm_type, dataset_name, random_seed, actual_kwargs, item])
            i += 1

        random.shuffle(process_queue)
        return process_queue

    def check_options(self) -> None:
        parameter_names = list(self.param_grid.keys())
        required_parameters = ['type', 'dataset', 'num_clusters']

        # Check if all required parameters are given
        for required_parameter in required_parameters:
            if required_parameter not in parameter_names:
                raise MultivariateParamTesterException('Parameter {} is required'.format(required_parameter))

    def fit(self) -> None:
        self.check_options()
        process_queue = self.queue_options()

        self.values = []
        self.errors = []

        start_time = time.time()
        parameter_count = len(process_queue)
        print(f'Testing {parameter_count} parameter combinations')

        for result in p_map(self.worker, process_queue):
            ok, result = result

            if ok:
                self.values.append(result)
            else:
                self.errors.extend(result)

        self.total_time = time.time() - start_time

    def fit_to_csv(self, filename: str) -> None:
        self.check_options()
        process_queue = self.queue_options()

        start_time = time.time()
        parameter_count = len(process_queue)
        print(f'Testing {parameter_count} parameter combinations')

        with open(filename, 'w', newline='') as csv_f, open(filename + '.errors', 'w') as err_f:
            i = 0

            writer = csv.writer(csv_f)
            writer.writerow(self.headers)

            for result in p_map(self.worker, process_queue):
                ok, result = result

                if ok:
                    writer.writerow(result)
                else:
                    err_f.write('\n'.join(result))

                if i % 50 == 0:
                    csv_f.flush()
                    err_f.flush()

                i += 1

        self.total_time = time.time() - start_time

    def write_csv(self, filename: str) -> None:
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(self.headers)
            writer.writerows(self.values)

        with open(filename + '.errors', 'w') as f:
            f.write('\n'.join(self.errors))

    def write_mat(self, filename: str, value_names: tuple[str]) -> None:
        index_index = self.headers.index('index')
        indices = [self.headers.index(value) for value in value_names]
        values = sorted(self.values, key=lambda x: x[index_index])
        values = [[values[i][index] for index in indices] for i in range(len(values))]

        savemat(filename, {'value_names': value_names, 'RESULT': values})

    """
    Runs the FCM algorithm on the given dataset and returns the performance.
    """
    def evaluate(self, item):
        # Unpack item
        i, fcm_type, dataset_name, random_seed, actual_kwargs, item = item

        # Load dataset and FCM
        fcm_class = get_fcm_by_type(fcm_type)
        dataset_key = (dataset_name, random_seed)
        dataset = self.dataset_cache[dataset_key]
        fcm = fcm_class(**actual_kwargs)

        X, num_clusters, true_labels = dataset

        # Apply random number generator seed
        if random_seed != -1:
            np.random.seed(random_seed)

        # Fit model
        start_time = time.time()
        fcm.fit(X)
        duration = time.time() - start_time

        # Evaluate performance
        purity, nmi, ari = fcm.evaluate(true_labels)
        csv_item = [value.name if hasattr(value, 'name') else value for value in item]

        try:
            eta = list(fcm.cluster_eta)
        except:
            eta = None

        return [i] + csv_item + [eta, duration, purity, nmi, ari]

    """
    Worker function that evaluates items from the input queue and puts the
    results in the output queue.
    """
    def worker(self, item):
        try:
            # Evaluate next item
            values = self.evaluate(item)
            return [True, values]
        except:
            return [False, [str(traceback.format_exc()), str(item)]]

if __name__ == '__main__':
    from algofuzz.enums import DatasetType, FCMType

    mvpt = MultivariateParamTester({
        'type': [FCMType.FCM, FCMType.PFCM, FCMType.STPFCM],
        'dataset': [DatasetType.Iris, DatasetType.NormalizedIris, DatasetType.NoisyNormalizedIris, DatasetType.NormalizedWine, DatasetType.BreastCancer, DatasetType.NormalizedBreastCancer, DatasetType.Bubbles1, DatasetType.Bubbles2, DatasetType.Bubbles3, DatasetType.Bubbles4],
        'num_clusters': [3],
        'm': [1.2, 1.5, 2.0, 3.0],
        'p': [1.2, 1.5, 2.0, 3.0],
        'weight': [1, 2],
        'seed': [0]
    })

    mvpt.fit_to_csv('offset.csv')
    print('Total time:', mvpt.total_time)
