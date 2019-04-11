import random
import numpy

from deap import base
from deap import creator
from deap import tools
from deap import algorithms


num_days = 6
num_shifts = 3
num_head_chefs = 6
num_sous_chef = 6
num_foh  = 6
num_waiter = 6

creator.create("FitnessMax",base.Fitness,weights=(-1.0,))
creator.create("Individual", numpy.ndarray, fitness=creator.FitnessMax)
toolbox = base.Toolbox()



toolbox.register("random_1",random.randint,0,1)
toolbox.register("individual",tools.initRepeat,creator.Individual,toolbox.random_1,100)
toolbox.register("population",tools.initRepeat, list, toolbox.individual)




def evalSchedule(individual):
    ind = individual.reshape((6,6,6,6,6,3))
    sum = 0
    for i in range(6):
        for j in range(3):
            sum = sum + numpy.sum(ind[:,:,:,:,i,j])
    return sum - 18 ,

def feasible(individual):
    if individual[0] > 0  :
        return True
    else :
        return False

def distance(individual):
    return individual[0] ** 2


def oneMax(individual):
    return sum(individual) ,


def cxTwoPointCopy(ind1, ind2):
    """Execute a two points crossover with copy on the input individuals. The
    copy is required because the slicing in numpy returns a view of the data,
    which leads to a self overwritting in the swap operation. It prevents
    ::

        >>> import numpy
        >>> a = numpy.array((1,2,3,4))
        >>> b = numpy.array((5.6.7.8))
        >>> a[1:3], b[1:3] = b[1:3], a[1:3]
        >>> print(a)
        [1 6 7 4]
        >>> print(b)
        [5 6 7 8]
    """
    size = len(ind1)
    cxpoint1 = random.randint(1, size)
    cxpoint2 = random.randint(1, size - 1)
    if cxpoint2 >= cxpoint1:
        cxpoint2 += 1
    else: # Swap the two cx points
        cxpoint1, cxpoint2 = cxpoint2, cxpoint1

    ind1[cxpoint1:cxpoint2], ind2[cxpoint1:cxpoint2] \
        = ind2[cxpoint1:cxpoint2].copy(), ind1[cxpoint1:cxpoint2].copy()

    return ind1, ind2


toolbox.register("evaluate", oneMax)
toolbox.register("mate", cxTwoPointCopy)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)


def ga_main():
    random.seed(64)
    pop = toolbox.population(n=300)
    hof = tools.HallOfFame(1, similar=numpy.array_equal)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)
    algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=50, stats=stats,
                        halloffame=hof)
    print("population:"+str(pop))
    print("stats:"+str(stats))
    print("hof:"+str(hof))
    print("count:"+str((hof[0] == 1).sum()))



if __name__ == '__main__':

   ga_main()