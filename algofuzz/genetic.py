import random
import numpy as np
from algofuzz.datasets import load_dataset
from algofuzz.enums import CentroidStrategy
from algofuzz.enums import DatasetType
from algofuzz.subset_selector import select_subset
from deap import base, creator, tools
from stpfcm_module import STPFCM
from sklearn.metrics import confusion_matrix
from algofuzz.validation import find_best_permutation
from algofuzz.algorithm import eaSimple
from algofuzz import centroid_strategy, evaluate
import random

def evaluate_fcm(individual, num_clusters, max_iter, X, true_labels):
    m, p, kappa, w_prob = individual[:4]

    parameters = {
        'max_iter': int(max_iter),
        'm': m,
        'p': p,
        'kappa': kappa,
        'w_prob': w_prob
    }

    if len(individual) == 5:
        # Optimizing for number of clusters
        num_clusters = int(individual[4])
        parameters['num_clusters'] = num_clusters
    else:
        # Not optimizing for number of clusters
        parameters['num_clusters'] = int(num_clusters)

    model = STPFCM(**parameters)
    model.set_centroids(centroid_strategy.create_centroids(X, CentroidStrategy.Random, num_clusters))
    model.fit(X)

    predicted_labels = model.get_predicted_labels()

    try:
        davies, silhouette = evaluate.evaluate_inner_metrics(X, predicted_labels)
    except ValueError as e:
        if 'Number of labels is' in str(e):
            # Labels have been calculated wrong
            davies = 1000
            silhouette = 0
        else:
            # Unknown error
            raise e

    if true_labels is not None:
        purity, nmi, ari = evaluate.evaluate_true_labels(predicted_labels, true_labels)
        return (purity, nmi, ari, silhouette, davies)
    else:
        return (silhouette, davies)

def check_bounds(low, up):
    def decorator(func):
        def wrapper(*args, **kwargs):
            offspring = func(*args, **kwargs)
            for child in offspring:
                for i in range(len(child)):
                    if child[i] < low[i]:
                        child[i] = low[i] + (low[i] - child[i])  # reflect back
                    elif child[i] > up[i]:
                        child[i] = up[i] - (child[i] - up[i])    # reflect back
            return offspring
        return wrapper
    return decorator

def genetic_optimize_fcm(X, num_clusters, max_iter, true_labels, ngen=20, pop_size=30):
    # Parameter bounds
    M = np.arange(1.1, 3.1, 0.1)
    P = np.arange(1.1, 3.1, 0.1)
    KAPPA = (1.1, 2.0)
    W_PROB = (1.1, 5.0)
    NUMBER_OF_CLUSTERS = (5, 5)

    if true_labels is not None:
        # Purity, NMI, ARI, silhouette, Davies
        creator.create("FitnessMulti", base.Fitness, weights=(1.0, 1.0, 1.0, 1.0, -1.0))
    else:
        # Silhouette, Davies
        creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0))

    creator.create("Individual", list, fitness=creator.FitnessMulti)

    toolbox = base.Toolbox()
    toolbox.register("kappa", random.uniform, *KAPPA)
    toolbox.register("w_prob", random.uniform, *W_PROB)

    def constrained_m():
        return random.uniform(M[0], M[-1])
    
    def constrained_p():
        return random.uniform(P[0], P[-1])

    def constrained_num_clusters():
        return random.randint(NUMBER_OF_CLUSTERS[0], NUMBER_OF_CLUSTERS[1])

    toolbox.register("m", constrained_m)
    toolbox.register("p", constrained_p)

    if true_labels is None:
        toolbox.register("num_clusters", constrained_num_clusters)
        params = (
            toolbox.m, toolbox.p, toolbox.kappa, toolbox.w_prob, toolbox.num_clusters
        )
    else:
        params = (
            toolbox.m, toolbox.p, toolbox.kappa, toolbox.w_prob
        )

    toolbox.register(
        "individual",
        tools.initCycle,
        creator.Individual,
        params,
        n=1
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate_fcm, num_clusters=num_clusters, X=X, max_iter=max_iter, true_labels=true_labels)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.2, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)

    BOUNDS_LOW = [M[0], P[0], KAPPA[0], W_PROB[0]]
    BOUNDS_UP = [M[-1], P[-1], KAPPA[1], W_PROB[1]]

    if true_labels is None:
        BOUNDS_LOW.append(NUMBER_OF_CLUSTERS[0])
        BOUNDS_UP.append(NUMBER_OF_CLUSTERS[1])

    toolbox.decorate("mate", check_bounds(BOUNDS_LOW, BOUNDS_UP))
    toolbox.decorate("mutate", check_bounds(BOUNDS_LOW, BOUNDS_UP))

    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(5)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("max", np.max)

    eaSimple(
        pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=ngen,
        stats=stats, halloffame=hof, verbose=True
    )

    return hof[0]

def print_confu(num_clusters, max_iter, X, true_labels, m, p, kappa, w_prob, actual_num_clusters=None):
    if actual_num_clusters is None:
        actual_num_clusters = int(num_clusters)

    actual_num_clusters = int(actual_num_clusters)

    model = STPFCM(
        num_clusters=actual_num_clusters,
        max_iter=int(max_iter),
        m=m,
        p=p,
        kappa=kappa,
        w_prob=w_prob
    )

    model.set_centroids(centroid_strategy.create_centroids(X, CentroidStrategy.Random, actual_num_clusters))
    model.fit(X)

    # Print header and values in a simple table
    print("{:<8} {:<8} {:<8} {:<10} {:<12}".format("m", "p", "kappa", "w_prob", "num_clusters"))
    print("-" * 50)
    print("{:<8.4f} {:<8.4f} {:<8.4f} {:<10.4f} {:<12d}".format(m, p, kappa, w_prob, int(actual_num_clusters)))

    # Provide predicted_labels for the subsequent evaluation code
    predicted_labels = model.get_predicted_labels()

    labels = model.get_predicted_labels()

    if true_labels is not None:
        purity, nmi, ari = evaluate.evaluate_true_labels(labels, true_labels)
        print("Purity:", purity)
        print("NMI:", nmi)
        print("ARI:", ari)
        conf_matrix = confusion_matrix(true_labels, labels[:len(true_labels)])
        best_permuted_confusion = find_best_permutation(conf_matrix)
        print(best_permuted_confusion)
        print(np.sum(np.diag(best_permuted_confusion)))

        fitness_score = 0.4 * purity + 0.3 * nmi + 0.3 * ari
        print("Fitness Score:", fitness_score)
    
    try:
        davies, silhouette = evaluate.evaluate_inner_metrics(X, predicted_labels)
    except ValueError as e:
        if 'Number of labels is' in str(e):
            # Labels have been calculated wrong
            davies = 1000
            silhouette = 0
        else:
            # Unknown error
            raise e

    print(f'Davies: {davies}, silhouette: {silhouette}')

def get_fitness_score(num_clusters, max_iter, X, true_labels, m, p, kappa, w_prob, actual_num_clusters=None):
    if actual_num_clusters is None:
        actual_num_clusters = int(num_clusters)

    actual_num_clusters = int(actual_num_clusters)

    model = STPFCM(
        num_clusters=actual_num_clusters,
        max_iter=int(max_iter),
        m=m,
        p=p,
        kappa=kappa,
        w_prob=w_prob
    )

    model.set_centroids(centroid_strategy.create_centroids(X, CentroidStrategy.Random, actual_num_clusters))
    model.fit(X)

    # Provide predicted_labels for the subsequent evaluation code
    predicted_labels = model.get_predicted_labels()

    labels = model.get_predicted_labels()

    if true_labels is not None:
        purity, nmi, ari = evaluate.evaluate_true_labels(labels, true_labels)
    
    try:
        davies, silhouette = evaluate.evaluate_inner_metrics(X, predicted_labels)
    except ValueError as e:
        if 'Number of labels is' in str(e):
            # Labels have been calculated wrong
            return 0
        else:
            # Unknown error
            raise e

    if true_labels is not None:
        fitness_score = purity
    else:
        fitness_score = silhouette
    
    return fitness_score

def calculate_optimized_hyperparameters():
    """
    Optimized Hyperparameters Table

Purpose: Show the final hyperparameters found by the GA for each dataset.

Columns: Dataset | $m$ | $p$ | $\kappa$ | $a$ | Number of clusters | Min Fitness | Max Fitness | Stddev Fitness

Optional: Include min/max/mean if you ran multiple GA runs to show stability.
    """
    datasets = [
        DatasetType.NormalizedBreastCancer,
        DatasetType.NormalizedSpellman,
        DatasetType.NormalizedWine,
        DatasetType.NormalizedSeeds,
        DatasetType.NormalizedIris
    ]
    #datasets = [
    #    DatasetType.NormalizedIris
    #]
    random_values = [0, 1, 2, 3, 4]
    # build results and save markdown table
    rows = []

    for dataset in datasets:
        X, c, true_labels = load_dataset(dataset)
        run_results = []

        for rand_val in random_values:
            np.random.seed(rand_val)
            random.seed(rand_val)

            max_iter = 100
            percentage = 1
            small_X, small_true_labels = select_subset(X, true_labels, percentage)

            best_params = genetic_optimize_fcm(small_X, c, max_iter, small_true_labels)

            chosen_c = int(best_params[-1]) if len(best_params) == 5 else int(c)
            score = get_fitness_score(chosen_c, max_iter, X, true_labels, *best_params)
            run_results.append((score, best_params))

        best_tuple = max(run_results, key=lambda x: x[0])
        min_tuple = min(run_results, key=lambda x: x[0])
        max_tuple = max(run_results, key=lambda x: x[0])
        stddev = float(np.std([r[0] for r in run_results]))

        best_params = best_tuple[1]
        m = float(best_params[0])
        p = float(best_params[1])
        kappa = float(best_params[2])
        a = float(best_params[3])
        num_clusters = int(best_params[4]) if len(best_params) == 5 else int(c)

        rows.append({
            "dataset": str(dataset),
            "m": m,
            "p": p,
            "kappa": kappa,
            "a": a,
            "num_clusters": num_clusters,
            "min_fitness": float(min_tuple[0]),
            "max_fitness": float(max_tuple[0]),
            "mean_fitness": float(np.mean([r[0] for r in run_results])),
            "stddev": stddev
        })

    # write markdown file
    md_lines = []
    md_lines.append("| Dataset | m | p | κ | a | Number of clusters | Min Fitness | Max Fitness | Mean Fitness | Stddev Fitness |")
    md_lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

    for r in rows:
        md_lines.append(
            "| {dataset} | {m:.4f} | {p:.4f} | {kappa:.4f} | {a:.4f} | {num_clusters} | {min_fitness:.6f} | {max_fitness:.6f} | {mean_fitness:.6f} | {stddev:.6f} |"
            .format(**r)
        )

    out_path = "optimized_hyperparameters.md"
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(md_lines))

    print(f"Wrote optimized hyperparameters to {out_path}")
    return rows


# Example usage:
import sys
if __name__ == "__main__":
    calculate_optimized_hyperparameters()
    sys.exit()
    datasets = [
        DatasetType.NormalizedBreastCancer,
        DatasetType.NormalizedSpellman,
        DatasetType.NormalizedWine,
        DatasetType.NormalizedSeeds,
        DatasetType.NormalizedIris
    ]
    # Generate dummy data
    X, c, true_labels = load_dataset(DatasetType.NormalizedIris)

    import time
    start_time = time.time()

    max_iter = 100
    percentage = 1
    small_X, small_true_labels = select_subset(X, true_labels, percentage)
    best_params = genetic_optimize_fcm(small_X, c, max_iter, small_true_labels)

    print('Best params:', best_params)
    #print("Best parameters found:", best_params)

    #m = 1.02
    #p = 2
    #kappa=1
    #w_prob=1
    #individual = (m,p,kappa,w_prob)
    #print(evaluate_fcm(individual, c, 100, X, true_labels))

    #best_params = [np.float64(2.766645499290813), np.float64(1.457847197366641), 7.6191293435736664, 3.2108470836542113]
    print_confu(c, 100, X, true_labels, *best_params)


    end_time = time.time()
    print(f"Execution Time: {end_time - start_time} seconds")