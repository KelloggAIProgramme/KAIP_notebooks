import numpy as np, random, operator, pandas as pd
import itertools
import matplotlib.pyplot as plt

'''
Helpers functions for the optimization tutorial
'''

####### <TRAVELLING SALESMAN> #######

### Genetic algorithm code is reused from
### https://towardsdatascience.com/evolution-of-a-salesman-a-complete-genetic-algorithm-tutorial-for-python-6fe5d2b3ca35

class City:
    ''' defines a city with a position,
    and and a way to compute distance to another city.'''

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def distance(self, City):
        xDis = abs(self.x - City.x)
        yDis = abs(self.y - City.y)
        distance = np.sqrt((xDis ** 2) + (yDis ** 2))
        return distance

    def __repr__(self):
        return "(" + str(self.x) + "," + str(self.y) + ")"


def generate_cities(n_cities):
    ''' generate a list of cities with random coordinates'''
    cityList = []

    for i in range(n_cities):
        cityList.append(City(x=int(random.random() * 1000), y=int(random.random() * 1000)))

    return cityList


def brute_force_tsp(cityList):

    counter_routes = 0
    smallest_distance = 0
    best_route = []

    ### create all permutations of the list
    for route in itertools.permutations(cityList):

        ### compute distance for this route.
        distance = 0
        for i in range( len(route) ):
            if (i < len(route) - 1):
                distance += route[i].distance(route[i+1])
            else:
                distance += route[-1].distance(route[0])

        if counter_routes == 0: ### if it's the first route, remember this one.
            smallest_distance = distance
            best_route = route

        else: ### if not first route, only keep it if it's better.
            if distance < smallest_distance:
                smallest_distance = distance
                best_route = route

        counter_routes += 1

    return smallest_distance, best_route, counter_routes


def plot_route(route):

    x_coords = []
    y_coords = []
    for i in range(len(route)):
        x_coords.append(route[i].x)
        y_coords.append(route[i].y)

    ### we want to go full circles so append the last one
    x_coords.append(route[0].x)
    y_coords.append(route[0].y)

    plt.plot(x_coords, y_coords, 'r', lw=3)
    plt.scatter(x_coords, y_coords, s=120)
    plt.xticks(fontsize = 14)
    plt.yticks(fontsize = 14)
    plt.xlabel('x (km)', fontsize = 20)
    plt.ylabel('y (km)', fontsize = 20)
    plt.title('Route', fontsize = 20)
    plt.show()

def plot_cities(cityList):
    x_coords = []
    y_coords = []
    for i in range(len(cityList)):
        x_coords.append(cityList[i].x)
        y_coords.append(cityList[i].y)

    ### we want to go full circles so append the last one
    x_coords.append(cityList[0].x)
    y_coords.append(cityList[0].y)

#    plt.plot(x_coords, y_coords, 'r', lw=3)
    plt.scatter(x_coords, y_coords, s=120)
    plt.xticks(fontsize = 14)
    plt.yticks(fontsize = 14)
    plt.xlabel('x (km)', fontsize = 20)
    plt.ylabel('y (km)', fontsize = 20)
    plt.title('Cities', fontsize = 20)
    plt.show()

def plot_progress(progress):

    plt.plot(progress)
    plt.xticks(fontsize = 14)
    plt.yticks(fontsize = 14)
    plt.ylabel('Distance (km)', fontsize = 20)
    plt.xlabel('Generation', fontsize = 20)
    plt.title('Progress of the genetic algorithm', fontsize = 20)
    plt.show()


####### </TRAVELLING SALESMAN> #######

####### <GENETIC ALGORITHM> #######
class Fitness:
    def __init__(self, route):
        self.route = route
        self.distance = 0
        self.fitness= 0.0

    def routeDistance(self):
        if self.distance ==0:
            pathDistance = 0
            for i in range(0, len(self.route)):
                fromCity = self.route[i]
                toCity = None
                if i + 1 < len(self.route):
                    toCity = self.route[i + 1]
                else:
                    toCity = self.route[0]
                pathDistance += fromCity.distance(toCity)
            self.distance = pathDistance
        return self.distance

    def routeFitness(self):
        if self.fitness == 0:
            self.fitness = 1 / float(self.routeDistance())
        return self.fitness

def createRoute(cityList):
    route = random.sample(cityList, len(cityList))
    return route


def initialPopulation(popSize, cityList):
    population = []

    for i in range(0, popSize):
        population.append(createRoute(cityList))
    return population


def rankRoutes(population):
    fitnessResults = {}
    for i in range(0,len(population)):
        fitnessResults[i] = Fitness(population[i]).routeFitness()
    return sorted(fitnessResults.items(), key = operator.itemgetter(1), reverse = True)


def selection(popRanked, eliteSize):
    selectionResults = []
    df = pd.DataFrame(np.array(popRanked), columns=["Index","Fitness"])
    df['cum_sum'] = df.Fitness.cumsum()
    df['cum_perc'] = 100*df.cum_sum/df.Fitness.sum()

    for i in range(0, eliteSize):
        selectionResults.append(popRanked[i][0])
    #for i in range(0, len(popRanked) - eliteSize):
    while len(selectionResults) < len(popRanked):
        pick = 100*random.random()
        for j in range(eliteSize, len(popRanked)):
            if pick <= df.iat[j,3]:
                selectionResults.append(popRanked[j][0])
                break

#    for i in range(0, eliteSize):
#        selectionResults.append(popRanked[i][0])
#    for i in range(0, len(popRanked) - eliteSize):
#        pick = 100*random.random()
#        for i in range(0, len(popRanked)):
#            if pick <= df.iat[i,3]:
#                selectionResults.append(popRanked[i][0])
#                break
    return selectionResults


def matingPool(population, selectionResults):
    matingpool = []
    for i in range(0, len(selectionResults)):
        index = selectionResults[i]
        matingpool.append(population[index])
    return matingpool


def breed(parent1, parent2):
    child = []
    childP1 = []
    childP2 = []

    geneA = int(random.random() * len(parent1))
    geneB = int(random.random() * len(parent1))

    startGene = min(geneA, geneB)
    endGene = max(geneA, geneB)

    for i in range(startGene, endGene):
        childP1.append(parent1[i])

    childP2 = [item for item in parent2 if item not in childP1]

    child = childP1 + childP2
    return child


def breedPopulation(matingpool, eliteSize):
    children = []
    length = len(matingpool) - eliteSize
    pool = random.sample(matingpool, len(matingpool))

    for i in range(0,eliteSize):
        children.append(matingpool[i])

    for i in range(0, length):
        child = breed(pool[i], pool[len(matingpool)-i-1])
        children.append(child)
    return children

def mutate(individual, mutationRate):
    for swapped in range(len(individual)):
        if(random.random() < mutationRate):
            swapWith = int(random.random() * len(individual))

            city1 = individual[swapped]
            city2 = individual[swapWith]

            individual[swapped] = city2
            individual[swapWith] = city1
    return individual

def mutatePopulation(population, mutationRate):
    mutatedPop = []

    for ind in range(0, len(population)):
        mutatedInd = mutate(population[ind], mutationRate)
        mutatedPop.append(mutatedInd)
    return mutatedPop

def nextGeneration(currentGen, eliteSize, mutationRate):
    popRanked = rankRoutes(currentGen)
    selectionResults = selection(popRanked, eliteSize)
    matingpool = matingPool(currentGen, selectionResults)
    children = breedPopulation(matingpool, eliteSize)
    nextGeneration = mutatePopulation(children, mutationRate)
    return nextGeneration


def geneticAlgorithm(population, popSize, eliteSize, mutationRate, generations):
    pop = initialPopulation(popSize, population)
    print("Initial distance: " + str(1 / rankRoutes(pop)[0][1]) + 'km')

    progress = []
    for i in range(0, generations):
        pop = nextGeneration(pop, eliteSize, mutationRate)
        progress.append(1 / rankRoutes(pop)[0][1])
        routeIndex = rankRoutes(pop)[0][0]
        route = pop[routeIndex]
        fitness = Fitness(route)

#        if (i == 0):
#            bestRouteIndex = routeIndex
#            bestRoute = route
#            best_fitness = fitness
#        else:
#            if (fitness.routeDistance() < best_fitness.routeDistance()):
#                bestRouteIndex = routeIndex
#                bestRoute = route
#                best_fitness = fitness
        bestRouteIndex = routeIndex
        bestRoute = route
        best_fitness = fitness

    print("Final distance: " + str(fitness.routeDistance()) + 'km')
    #print("Best distance: " + str(best_fitness.routeDistance()) + ' km')

    return bestRoute, best_fitness.routeDistance(), progress

####### </GENETIC ALGORITHM> #######
