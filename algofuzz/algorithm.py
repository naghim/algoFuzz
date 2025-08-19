from deap import tools
from deap.algorithms import varAnd


def eaSimple(population, toolbox, cxpb, mutpb, ngen, stats=None,
             halloffame=None, verbose=__debug__):
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else []) + ['hof_fitness']

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population)

    record = stats.compile(population) if stats else {}
    hof_fit = halloffame[0].fitness.values[0] if halloffame and halloffame[0].fitness.valid else None
    logbook.record(gen=0, nevals=len(invalid_ind), hof_fitness=hof_fit, **record)
    if verbose:
        print(logbook.stream)
    
    prev_hof = None

    # Begin the generational process
    for gen in range(1, ngen + 1):
        # Select the next generation individuals
        offspring = toolbox.select(population, len(population))

        # Vary the pool of individuals
        offspring = varAnd(offspring, toolbox, cxpb, mutpb)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)

        # Replace the current population by the offspring
        population[:] = offspring

        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        hof_fit = halloffame[0].fitness.values[0] if halloffame and halloffame[0].fitness.valid else None
        logbook.record(gen=gen, nevals=len(invalid_ind), hof_fitness=hof_fit, **record)
        if verbose:
            print(logbook.stream)

        if hof_fit == prev_hof:
            print(f'Hof fit: {hof_fit} is now same as {prev_hof}, stopping early')
            break
        
        print(f'Prev hof: {prev_hof}, new hof fit {hof_fit}')
        prev_hof = hof_fit

    return population, logbook