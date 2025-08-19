import random
import numpy as np
from algofuzz.datasets import load_dataset
from algofuzz.enums import CentroidStrategy
from algofuzz.enums import DatasetType
from deap import base, creator, tools
from stpfcm_module import STPFCM
from sklearn.metrics import confusion_matrix
from algofuzz.validation import find_best_permutation
from algofuzz.algorithm import eaSimple
from algofuzz import centroid_strategy, evaluate

def evaluate_fcm(individual, num_clusters, max_iter, X, true_labels):
    m, p, kappa, w_prob = individual
    #print("Evaluating individual:", individual)
    #print("Parameters - m:", m, "p:", p, "kappa:", kappa, "w_prob:", w_prob)
    #print("Number of clusters:", num_clusters, "Max iterations:", max_iter)
    parameters = {
        'num_clusters': int(num_clusters),
        'max_iter': int(max_iter),
        'm': m,
        'p': p,
        'kappa': kappa,
        'w_prob': w_prob
    }

    model = STPFCM(**parameters)
    model.set_centroids(centroid_strategy.create_centroids(X, CentroidStrategy.Diagonal, num_clusters))
    model.fit(X)

    purity, nmi, ari = evaluate.evaluate_true_labels(model.get_predicted_labels(), true_labels)
    fitness_score = 0.4 * purity + 0.3 * nmi + 0.3 * ari
    return (purity, nmi, ari)

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

    creator.create("FitnessMulti", base.Fitness, weights=(1.0, 1.0, 1.0))
    creator.create("Individual", list, fitness=creator.FitnessMulti)

    toolbox = base.Toolbox()
    toolbox.register("kappa", random.uniform, *KAPPA)
    toolbox.register("w_prob", random.uniform, *W_PROB)

    def constrained_m():
        return random.uniform(M[0], M[-1])
    
    def constrained_p():
        return random.uniform(P[0], P[-1])
 
    toolbox.register("m", constrained_m)
    toolbox.register("p", constrained_p)

    toolbox.register(
        "individual",
        tools.initCycle,
        creator.Individual,
        (toolbox.m, toolbox.p, toolbox.kappa, toolbox.w_prob),
        n=1
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate_fcm, num_clusters=num_clusters, X=X, max_iter=max_iter, true_labels=true_labels)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.2, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)

    BOUNDS_LOW = [M[0], P[0], KAPPA[0], W_PROB[0]]
    BOUNDS_UP = [M[-1], P[-1], KAPPA[1], W_PROB[1]]

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

def print_confu(num_clusters, max_iter, X, true_labels, m, p, kappa, w_prob):
    model = STPFCM(
        num_clusters=int(num_clusters),
        max_iter=int(max_iter),
        m=m,
        p=p,
        kappa=kappa,
        w_prob=w_prob
    )

    model.set_centroids(centroid_strategy.create_centroids(X, CentroidStrategy.Diagonal, num_clusters))
    model.fit(X)

    labels = model.get_predicted_labels()
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

# Example usage:
if __name__ == "__main__":
    # Generate dummy data
    X, c, true_labels = load_dataset(DatasetType.NormalizedSeeds)

    import time
    start_time = time.time()

    labels = np.array(true_labels)
    classes, counts = np.unique(labels, return_counts=True)
    # prepare selection: 10% per class with same distribution
    rng = np.random.default_rng()
    per_class = np.maximum(1, (counts * 1).astype(int))  # at least 1 per class

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

    best_params = genetic_optimize_fcm(small_X, c, 100, small_true_labels)

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