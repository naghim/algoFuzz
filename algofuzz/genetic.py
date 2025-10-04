import random
import numpy as np
from algofuzz.datasets import load_dataset
from algofuzz.enums import CentroidStrategy
from algofuzz.enums import DatasetType
from algofuzz.fcm.python.fcm import FCM
from algofuzz.fcm.possibilistic_fcm import PFCM
from algofuzz.subset_selector import select_subset
from deap import base, creator, tools
from algofuzz._algofuzz import STPFCM
from sklearn.metrics import confusion_matrix
from algofuzz.validation import find_best_permutation
from algofuzz.algorithm import eaSimple
from algofuzz import centroid_strategy, evaluate
import random
import matplotlib.pyplot as plt
import time

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
    toolbox.register("select", tools.selTournament, tournsize=5)

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

    pop, logbook = eaSimple(
        pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=ngen,
        stats=stats, halloffame=hof, verbose=True
    )

    return hof[0], logbook

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
            "| {dataset} | {m:.4f} | {p:.4f} | {kappa:.4f} | {a:.4f} | {num_clusters} | {min_fitness:.2f} | {max_fitness:.2f} | {mean_fitness:.2f} | {stddev:.4f} |"
            .format(**r)
        )

    out_path = "optimized_hyperparameters.md"
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(md_lines))

    print(f"Wrote optimized hyperparameters to {out_path}")
    return rows


def save_clustering_performance_table():
    """
    Clustering Performance Table

    Purpose: Compare ST-PFCM with GA-tuned hyperparameters against:

    Default ST-PFCM parameters
    Classical FCM
    """
    datasets = [
        DatasetType.NormalizedBreastCancer,
        DatasetType.NormalizedSpellman,
        DatasetType.NormalizedWine,
        DatasetType.NormalizedSeeds,
        DatasetType.NormalizedIris
    ]
    
    rows = []

    for dataset_type in datasets:
        np.random.seed(42)
        random.seed(42)

        print(f'========{dataset_type}========')
        X, c, true_labels = load_dataset(dataset_type)
        max_iter = 100
        percentage = 1
        small_X, small_true_labels = select_subset(X, true_labels, percentage)

        # GA Optimized ST-PFCM
        best_params = genetic_optimize_fcm(small_X, c, max_iter, small_true_labels)
        ga_optimized_num_clusters = int(best_params[-1]) if len(best_params) == 5 else int(c)
        
        ga_stpfcm_model = STPFCM(
            num_clusters=ga_optimized_num_clusters,
            max_iter=max_iter,
            m=best_params[0],
            p=best_params[1],
            kappa=best_params[2],
            w_prob=best_params[3]
        )
        ga_stpfcm_model.set_centroids(centroid_strategy.create_centroids(X, CentroidStrategy.Random, ga_optimized_num_clusters))
        ga_stpfcm_model.fit(X)
        ga_stpfcm_predicted_labels = ga_stpfcm_model.get_predicted_labels()

        # Default ST-PFCM
        default_stpfcm_model = STPFCM(num_clusters=c if c is not None else ga_optimized_num_clusters, max_iter=max_iter)
        default_stpfcm_model.set_centroids(centroid_strategy.create_centroids(X, CentroidStrategy.Random, c if c is not None else ga_optimized_num_clusters))
        default_stpfcm_model.fit(X)
        default_stpfcm_predicted_labels = default_stpfcm_model.get_predicted_labels()

        # Classical FCM
        classical_fcm_model = FCM(num_clusters=c if c is not None else ga_optimized_num_clusters, max_iter=max_iter)
        classical_fcm_model.fit(X)
        classical_fcm_predicted_labels = classical_fcm_model.labels

        if true_labels is not None:
            classical_fcm_predicted_labels = classical_fcm_predicted_labels[:len(true_labels)]
        
        # Possibilistic FCM
        possibilistic_fcm_model = PFCM(num_clusters=c if c is not None else ga_optimized_num_clusters, max_iter=max_iter)
        possibilistic_fcm_model.fit(X)
        possibilistic_fcm_predicted_labels = possibilistic_fcm_model.labels

        if true_labels is not None:
            possibilistic_fcm_predicted_labels = possibilistic_fcm_predicted_labels[:len(true_labels)]

        # Evaluate metrics
        def get_metrics(predicted_labels, true_labels, X):
            pur, nmi, ari = ('-', '-', '-')
            if true_labels is not None:
                processed_predicted_labels = predicted_labels[:len(true_labels)]
                pur, ari, nmi = evaluate.evaluate_true_labels(processed_predicted_labels, true_labels)
            else:
                processed_predicted_labels = predicted_labels 
        
            try:
                davies, silhouette = evaluate.evaluate_inner_metrics(X, processed_predicted_labels[:X.shape[1]])
            except ValueError as e:
                if 'Number of labels is' in str(e):
                    davies = 1000
                    silhouette = 0
                else:
                    raise e
            return pur, nmi, ari, davies, silhouette

        ga_stpfcm_metrics = get_metrics(ga_stpfcm_predicted_labels, true_labels, X)
        default_stpfcm_metrics = get_metrics(default_stpfcm_predicted_labels, true_labels, X)
        classical_fcm_metrics = get_metrics(classical_fcm_predicted_labels, true_labels, X)
        possibilistic_fcm_metrics = get_metrics(possibilistic_fcm_predicted_labels, true_labels, X)

        def fmt(x):
            return x if x == '-' else f"{x:.2f}"

        rows.append({
            "dataset": str(dataset_type),
            "ga_pur": fmt(ga_stpfcm_metrics[0]),
            "ga_nmi": fmt(ga_stpfcm_metrics[1]),
            "ga_ari": fmt(ga_stpfcm_metrics[2]),
            "ga_davies": fmt(ga_stpfcm_metrics[3]),
            "ga_silhouette": fmt(ga_stpfcm_metrics[4]),
            "default_pur": fmt(default_stpfcm_metrics[0]),
            "default_nmi": fmt(default_stpfcm_metrics[1]),
            "default_ari": fmt(default_stpfcm_metrics[2]),
            "default_davies": fmt(default_stpfcm_metrics[3]),
            "default_silhouette": fmt(default_stpfcm_metrics[4]),
            "classical_pur": fmt(classical_fcm_metrics[0]),
            "classical_nmi": fmt(classical_fcm_metrics[1]),
            "classical_ari": fmt(classical_fcm_metrics[2]),
            "classical_davies": fmt(classical_fcm_metrics[3]),
            "classical_silhouette": fmt(classical_fcm_metrics[4]),
            "possibilistic_pur": fmt(possibilistic_fcm_metrics[0]),
            "possibilistic_nmi": fmt(possibilistic_fcm_metrics[1]),
            "possibilistic_ari": fmt(possibilistic_fcm_metrics[2]),
            "possibilistic_davies": fmt(possibilistic_fcm_metrics[3]),
            "possibilistic_silhouette": fmt(possibilistic_fcm_metrics[4]),
        })

    # write markdown file
    md_lines = []
    md_lines.append("| Dataset | GA Optimized: PUR | NMI | ARI | Davies | Silhouette | Default parameters: PUR | NMI | ARI | Davies | Silhouette | Classical FCM: PUR | NMI | ARI | Davies | Silhouette | Possibilistic FCM: PUR | NMI | ARI | Davies | Silhouette |")
    md_lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

    for r in rows:
        md_lines.append(
            "| {dataset} | {ga_pur} | {ga_nmi} | {ga_ari} | {ga_davies} | {ga_silhouette} | {default_pur} | {default_nmi} | {default_ari} | {default_davies} | {default_silhouette} | {classical_pur} | {classical_nmi} | {classical_ari} | {classical_davies} | {classical_silhouette} | {possibilistic_pur} | {possibilistic_nmi} | {possibilistic_ari} | {possibilistic_davies} | {possibilistic_silhouette} |"
            .format(**r)
        )

    out_path = "clustering_performance_table.md"
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(md_lines))

    print(f"Wrote clustering performance table to {out_path}")
def plot_fitness_over_generations(filename="fitness_over_generations.png"):
    datasets = [
        DatasetType.NormalizedBreastCancer,
        DatasetType.NormalizedWine,
        DatasetType.NormalizedSeeds,
        DatasetType.NormalizedIris
    ]
    
    all_logbooks = []
    random_values = [0, 1, 2, 3, 4] # Using the same random seeds as calculate_optimized_hyperparameters

    for dataset_type in datasets:
        np.random.seed(21) # Use a fixed seed for reproducibility of the plot
        random.seed(21) # Use a fixed seed for reproducibility of the plot

        X, c, true_labels = load_dataset(dataset_type)
        max_iter = 100
        percentage = 1
        small_X, small_true_labels = select_subset(X, true_labels, percentage)

        # Run genetic optimization and get the logbook
        _, logbook = genetic_optimize_fcm(small_X, c, max_iter, small_true_labels)
        logbook.dataset_name = str(dataset_type).split('.')[-1] # Extract dataset name
        all_logbooks.append(logbook)

    plt.figure(figsize=(12, 8))
    colors = ['blue', 'green', 'red', 'purple', 'orange']
    
    for i, logbook in enumerate(all_logbooks):
        gen = logbook.select("gen")
        hof_fitness = logbook.select("hof_fitness")
        dataset_name = logbook.dataset_name
        print(f'Gen: {gen}')
        print(f'HOF Fitness: {hof_fitness}')
        plt.plot(gen, hof_fitness, label=f'Dataset: {dataset_name}', color=colors[i % len(colors)])

    plt.xlabel("Generation")
    plt.ylabel("Best Fitness Score")
    plt.title("Best Fitness Score Over Generations for Each Dataset")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(filename, dpi=300)
    plt.show()
    return rows

def create_computational_efficiency_table():
    """
    Computational Efficiency Table

    Purpose: Calculate the best params for different percentages of the dataset,
    and provide runtime improvements and performance decrease compared to 100%.

    Columns: Dataset | Percentage | Max Fitness | Mean/Stddev Fitness | Runtime Improvement (%) | Performance Decrease (%)
    """
    datasets = [
        DatasetType.NormalizedBreastCancer,
        DatasetType.NormalizedSpellman,
        DatasetType.NormalizedWine,
        DatasetType.NormalizedSeeds,
        DatasetType.NormalizedIris
    ]
    percentages = [1.0, 0.75, 0.50, 0.25, 0.10]
    random_values = [0, 1, 2, 3, 4]
    max_iter = 100
    
    rows = []

    for dataset_type in datasets:
        X_full, c_full, true_labels_full = load_dataset(dataset_type)
        
        # Run for 100% to get baseline
        full_dataset_runtimes = []
        full_dataset_fitnesses = []
        for rand_val in random_values:
            np.random.seed(rand_val)
            random.seed(rand_val)
            
            start_time = time.time()
            best_params, _ = genetic_optimize_fcm(X_full, c_full, max_iter, true_labels_full)
            end_time = time.time()
            
            runtime = end_time - start_time
            full_dataset_runtimes.append(runtime)
            
            chosen_c = int(best_params[-1]) if len(best_params) == 5 else int(c_full)
            fitness_score = get_fitness_score(chosen_c, max_iter, X_full, true_labels_full, *best_params)
            full_dataset_fitnesses.append(fitness_score)
        
        baseline_mean_runtime = np.mean(full_dataset_runtimes)
        baseline_mean_fitness = np.mean(full_dataset_fitnesses)
        baseline_max_fitness = np.max(full_dataset_fitnesses)
        baseline_stddev_fitness = np.std(full_dataset_fitnesses)

        # Add 100% baseline to rows
        rows.append({
            "dataset": str(dataset_type).split('.')[-1],
            "percentage": 100,
            "mean_max_fitness": baseline_mean_fitness,
            "max_fitness": baseline_max_fitness,
            "stddev_fitness": baseline_stddev_fitness,
            "runtime_improvement": 0.0, # No improvement for 100%
            "performance_decrease": 0.0 # No decrease for 100%
        })
        
        for percentage in percentages[1:]: # Start from 75%
            run_runtimes = []
            run_fitnesses = []
            
            for rand_val in random_values:
                np.random.seed(rand_val)
                random.seed(rand_val)
                
                small_X, small_true_labels = select_subset(X_full, true_labels_full, percentage)
                
                start_time = time.time()
                best_params, _ = genetic_optimize_fcm(small_X, c_full, max_iter, small_true_labels)
                end_time = time.time()
                
                runtime = end_time - start_time
                run_runtimes.append(runtime)
                
                chosen_c = int(best_params[-1]) if len(best_params) == 5 else int(c_full)
                fitness_score = get_fitness_score(chosen_c, max_iter, X_full, true_labels_full, *best_params)
                run_fitnesses.append(fitness_score)
            
            mean_max_fitness = np.mean(run_fitnesses)
            max_fitness = np.max(run_fitnesses)
            stddev_fitness = np.std(run_fitnesses)
            mean_runtime = np.mean(run_runtimes)
            
            runtime_improvement_percent = ((baseline_mean_runtime - mean_runtime) / baseline_mean_runtime) * 100 if baseline_mean_runtime > 0 else 0
            performance_decrease_percent = ((baseline_mean_fitness - mean_max_fitness) / baseline_mean_fitness) * 100 if baseline_mean_fitness > 0 else 0

            rows.append({
                "dataset": str(dataset_type).split('.')[-1],
                "percentage": int(percentage * 100),
                "mean_max_fitness": mean_max_fitness,
                "max_fitness": max_fitness,
                "stddev_fitness": stddev_fitness,
                "runtime_improvement": runtime_improvement_percent,
                "performance_decrease": performance_decrease_percent
            })

    # Write markdown file
    md_lines = []
    md_lines.append("| Dataset | Percentage | Max Fitness | Mean/Stddev Fitness | Runtime Improvement (%) | Performance Decrease (%) |")
    md_lines.append("|---|---:|---:|---:|---:|---:|")

    for r in rows:
        md_lines.append(
            "| {dataset} | {percentage}% | {max_fitness:.3f} | {mean_max_fitness:.3f} (± {stddev_fitness:.3f}) | {runtime_improvement:.2f} | {performance_decrease:.2f} |"
            .format(**r)
        )

    out_path = "computational_efficiency_table.md"
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(md_lines))

    print(f"Wrote computational efficiency table to {out_path}")
    return rows

def create_maxiter_efficiency_table():
    """
    Maxiter Efficiency Table

    Purpose: Calculate the best params for different percentages of the dataset,
    and provide runtime improvements and performance decrease compared to 100%.

    Columns: Dataset | Percentage | Max Fitness | Mean/Stddev Fitness | Runtime Improvement (%) | Performance Decrease (%)
    """
    datasets = [
        DatasetType.NormalizedBreastCancer,
        DatasetType.NormalizedSpellman,
        DatasetType.NormalizedWine,
        DatasetType.NormalizedSeeds,
        DatasetType.NormalizedIris
    ]
    percentages = [1.0, 0.75, 0.50, 0.25, 0.10]
    random_values = [0, 1, 2, 3, 4]
    
    rows = []
    max_iter = 100

    for dataset_type in datasets:
        X_full, c_full, true_labels_full = load_dataset(dataset_type)
        
        # Run for 100% to get baseline
        full_dataset_runtimes = []
        full_dataset_fitnesses = []
        for rand_val in random_values:
            np.random.seed(rand_val)
            random.seed(rand_val)
            
            start_time = time.time()
            best_params, _ = genetic_optimize_fcm(X_full, c_full, max_iter, true_labels_full)
            end_time = time.time()
            
            runtime = end_time - start_time
            full_dataset_runtimes.append(runtime)
            
            chosen_c = int(best_params[-1]) if len(best_params) == 5 else int(c_full)
            fitness_score = get_fitness_score(chosen_c, max_iter, X_full, true_labels_full, *best_params)
            full_dataset_fitnesses.append(fitness_score)
        
        baseline_mean_runtime = np.mean(full_dataset_runtimes)
        baseline_mean_fitness = np.mean(full_dataset_fitnesses)
        baseline_max_fitness = np.max(full_dataset_fitnesses)
        baseline_stddev_fitness = np.std(full_dataset_fitnesses)

        # Add 100% baseline to rows
        rows.append({
            "dataset": str(dataset_type).split('.')[-1],
            "percentage": 100,
            "mean_max_fitness": baseline_mean_fitness,
            "max_fitness": baseline_max_fitness,
            "stddev_fitness": baseline_stddev_fitness,
            "runtime_improvement": 0.0, # No improvement for 100%
            "performance_decrease": 0.0 # No decrease for 100%
        })
        
        for percentage in percentages[1:]: # Start from 75%
            run_runtimes = []
            run_fitnesses = []
            
            for rand_val in random_values:
                np.random.seed(rand_val)
                random.seed(rand_val)
                
                start_time = time.time()
                best_params, _ = genetic_optimize_fcm(X_full, c_full, int(max_iter * percentage), true_labels_full)
                end_time = time.time()
                
                runtime = end_time - start_time
                run_runtimes.append(runtime)
                
                chosen_c = int(best_params[-1]) if len(best_params) == 5 else int(c_full)
                fitness_score = get_fitness_score(chosen_c, max_iter, X_full, true_labels_full, *best_params)
                run_fitnesses.append(fitness_score)
            
            mean_max_fitness = np.mean(run_fitnesses)
            max_fitness = np.max(run_fitnesses)
            stddev_fitness = np.std(run_fitnesses)
            mean_runtime = np.mean(run_runtimes)
            
            runtime_improvement_percent = ((baseline_mean_runtime - mean_runtime) / baseline_mean_runtime) * 100 if baseline_mean_runtime > 0 else 0
            performance_decrease_percent = ((baseline_mean_fitness - mean_max_fitness) / baseline_mean_fitness) * 100 if baseline_mean_fitness > 0 else 0

            rows.append({
                "dataset": str(dataset_type).split('.')[-1],
                "percentage": int(percentage * 100),
                "mean_max_fitness": mean_max_fitness,
                "max_fitness": max_fitness,
                "stddev_fitness": stddev_fitness,
                "runtime_improvement": runtime_improvement_percent,
                "performance_decrease": performance_decrease_percent
            })

    # Write markdown file
    md_lines = []
    md_lines.append("| Dataset | Percentage | Max Fitness | Mean/Stddev Fitness | Runtime Improvement (%) | Performance Decrease (%) |")
    md_lines.append("|---|---:|---:|---:|---:|---:|")

    for r in rows:
        md_lines.append(
            "| {dataset} | {percentage}% | {max_fitness:.3f} | {mean_max_fitness:.3f} (± {stddev_fitness:.3f}) | {runtime_improvement:.2f} | {performance_decrease:.2f} |"
            .format(**r)
        )

    out_path = "maxiter_efficiency_table.md"
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(md_lines))

    print(f"Wrote max iteration efficiency table to {out_path}")
    return rows
if __name__ == "__main__":
    #calculate_optimized_hyperparameters()
    #save_clustering_performance_table()
    #plot_fitness_over_generations()
    create_maxiter_efficiency_table()
    import sys
    sys.exit()