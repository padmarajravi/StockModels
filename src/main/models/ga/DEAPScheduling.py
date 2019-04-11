import random
import numpy

from deap import base
from deap import creator
from deap import tools
from deap import algorithms


num_days = 3
num_shifts = 6
num_head_chefs = 6
num_sous_chef = 6
num_foh  = 6
num_waiter = 6
total_num = num_head_chefs*num_sous_chef*num_foh*num_waiter*num_shifts*num_days

creator.create("FitnessMax",base.Fitness,weights=(-1.0,))
creator.create("Individual", numpy.ndarray, fitness=creator.FitnessMax)
toolbox = base.Toolbox()


def ranGen():
    x = random.random()
    if  x < 0.5 :
        return 1
    else:
        return 0

toolbox.register("random_1",ranGen)
toolbox.register("individual",tools.initRepeat,creator.Individual,toolbox.random_1,total_num)
toolbox.register("population",tools.initRepeat, list, toolbox.individual)

dim = (num_head_chefs,num_sous_chef,num_foh,num_waiter,num_shifts,num_days)


def evalSchedule(individual):
    ind = individual.reshape(dim)
    sum = 0
    for i in range(num_shifts):
        for j in range(num_days):
            sum = sum + numpy.sum(ind[:,:,:,:,num_shifts,num_days])
    return sum,

def feasible(individual):
    if evalSchedule(individual)[0] >= 0  :
        return True
    else :
        return False

def distance(individual):
    return individual[0] ** 2


def oneMax(individual):
    return sum(individual)


def cxTwoPointCopy(ind1, ind2):
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


toolbox.register("evaluate", evalSchedule)
toolbox.decorate("evaluate", tools.DeltaPenality(feasible, 1.0, distance))
toolbox.register("mate", cxTwoPointCopy)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)


def ga_main():
    random.seed(64)
    pop = toolbox.population(n=1000)
    hof = tools.HallOfFame(1, similar=numpy.array_equal)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)
    algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=100, stats=stats,
                        halloffame=hof)
    print("population:"+str(pop))
    print("stats:"+str(stats))
    print("hof:"+str(hof))
    print("count:"+str((hof[0] == 1).sum()))
    hof_reshaped = hof[0].reshape(dim)
    hof_count = 0
    for day in range(num_days):
        for shift in range(num_shifts):
            for hc in range(num_head_chefs):
                for sc in range(num_sous_chef):
                    for fc in range(num_foh):
                        for wc in range(num_waiter):
                            if(hof_reshaped[hc,sc,fc,wc,shift,day] == 1):
                                print(str((hc,sc,hc,wc))+" on day:"+str(day)+" shift "+str(shift))
                                hof_count = hof_count + 1

    print("Found ones at :"+str(hof_count))


if __name__ == '__main__':

    ga_main()