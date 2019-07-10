import random
import numpy

from deap import base
from deap import creator
from deap import tools
from deap import algorithms
from operator import itemgetter


num_days = 3
num_shifts = 6
num_head_chefs = 6
num_sous_chef = 6
num_foh  = 6
num_waiter = 6
dims = (num_head_chefs,num_sous_chef,num_foh,num_waiter,num_days,num_shifts)

creator.create("FitnessMax",base.Fitness,weights=(-1.0,))
creator.create("Individual", numpy.ndarray, fitness=creator.FitnessMax)
toolbox = base.Toolbox()

def cust_random():
    rand_int = random.randint(0,10000)
    if rand_int < 100 :
        return 1
    else :
        return 0


toolbox.register("random_1",cust_random)
toolbox.register("individual",tools.initRepeat,creator.Individual,toolbox.random_1,num_days*num_shifts*num_head_chefs*num_sous_chef*num_foh*num_waiter)
toolbox.register("population",tools.initRepeat, list, toolbox.individual)



def evalSchedule(individual):
    ind = individual.reshape(dims)
    sum = 0

    # Every shift should have an hc,sc,foh and waiter
    shift_assigned_values = 0
    for i in range(num_days):
        for j in range(num_shifts):
            value = numpy.sum(ind[:,:,:,:,i,j])
            sum = sum + abs(value-1)


    # No person should work in more than one shift a day
    for day in range(num_days):
        for emp in range(6):
            value = numpy.sum(ind[emp,:,:,:,day,:])
            sum = sum + abs(value- 1)
            value = numpy.sum(ind[:,emp,:,:,day,:])
            sum = sum + abs(value - 1)
            value = numpy.sum(ind[:,:,emp,:,day,:])
            sum = sum + abs(value - 1)
            value = numpy.sum(ind[:,:,:,emp,day,:])
            sum = sum + abs(value - 1)



    
    # Not two same persons should work togather more than once.
    for emp1 in range(6):
        for emp2 in range(6):
            sum = sum + abs(numpy.sum(ind[emp1,emp2,:,:,:,:]) - 1)
            sum = sum + abs(numpy.sum(ind[emp1,:,emp2,:,:,:]) - 1)
            sum = sum + abs(numpy.sum(ind[emp1,:,:,emp2,:,:]) - 1)
            sum = sum + abs(numpy.sum(ind[:,emp1,emp2,:,:,:]) - 1)
            sum = sum + abs(numpy.sum(ind[:,emp1,:,emp2,:,:]) - 1)
            sum = sum + abs(numpy.sum(ind[:,:,emp1,emp2,:,:]) - 1)

    


    return sum ,

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


toolbox.register("evaluate", evalSchedule)
toolbox.register("mate", cxTwoPointCopy)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)


def ga_main():
    random.seed(64)
    pop = toolbox.population(n=500)
    hof = tools.HallOfFame(1, similar=numpy.array_equal)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)
    algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.5, ngen=10000, stats=stats,
                        halloffame=hof)
    print("population:"+str(pop))
    print("stats:"+str(stats))
    print("hof:"+str(hof))
    print("count:"+str((hof[0] == 1).sum()))
    hof_reshaped = hof[0].reshape(dims)
    out_list = []
    for hc in range(0,num_head_chefs):
        for sc in range(0,num_sous_chef):
            for foh in range(0,num_foh):
                for w in range(0,num_waiter):
                    for day in range(0,num_days):
                        for shift in range(0,num_shifts):
                           if hof_reshaped[hc,sc,foh,w,day,shift] == 1:
                                out_list.append((day,shift,hc,sc,foh,w))


    out_list.sort(key = itemgetter(0,1))

    print("########## Sorted output ##################")

    for entry in out_list:
        print("#################################")
        print("---------------------------------")
        print("Day:"+str(entry[0])+" Shift:"+str(entry[1]))
        print("---------------------------------")
        print("Head shef : "+str(entry[2]))
        print("Sous shef : "+str(entry[3]))
        print("Foh       : "+str(entry[4]))
        print("waiter    : "+str(entry[5]))
        print("---------------------------------")

    print("Complete")



if __name__ == '__main__':
    ga_main()