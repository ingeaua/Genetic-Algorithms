from random import randint, random
from math import log2, ceil
import matplotlib.pyplot as plt
import numpy as np


def readInput():

    fileName = input("Enter file name: ").strip()
    file = open(fileName, 'r')

    populationSize = int(file.readline().strip())
    lowerBound, upperBound = [int(x) for x in file.readline().strip().split()]
    a, b, c = [int(x) for x in file.readline().strip().split()]
    precision = int(file.readline().strip())
    crossoverProbability = float(file.readline().strip())
    mutationProbability = float(file.readline().strip())
    stageCount = int(file.readline().strip())

    return (populationSize, lowerBound, upperBound, a, b, c, precision,
            crossoverProbability, mutationProbability, stageCount)


# function that creates a random initial population of chromosomes
def randomInitialPopulation(bitCnt, popSzie):

    return [[randint(0, 1) for _ in range(bitCnt)] for _ in range(popSzie)]


def quadraticFunction(a, b, c, x):

    return a * x**2 + b * x + c


def decode(chromosome, discreteStep, precision):

    x = ''.join(str(i) for i in chromosome)

    index = int(x, 2)

    return round(lowerBound + index * discreteStep, precision + 1)


def binarySearch(x, v, left, right):

    index = 0

    while left <= right:
        middle = (left + right) // 2
        if v[middle] <= x:
            index = middle
            left = middle + 1
        elif v[middle] > x:
            right = middle - 1

    return index


def crossover(popSize, crossoverProb, newChromosomesList):

    crossoverIdxs = []

    # for each chromosome we generate a random number. if the random number is smaller than
    # the crossover probability, then that chromosome takes part in the crossover
    for i in range(popSize):

        uniformNumber = random()
        if uniformNumber < crossoverProb:
            crossoverIdxs.append(i)

    # we need an even number of chromosomes to crossover
    if len(crossoverIdxs) % 2 == 1:
        crossoverIdxs.pop()

    # while we have chromosomes to crossover, we get 2 random ones and we cross them over in a random point

    while len(crossoverIdxs):

        chromosome1Index = crossoverIdxs[randint(0, len(crossoverIdxs)) - 1]
        chromosome2Index = crossoverIdxs[randint(0, len(crossoverIdxs)) - 1]

        # skip if we get the same chromosome twice
        if chromosome1Index == chromosome2Index:
            continue

        crossoverPoint = randint(0, bitCount - 1)

        newChromosomesList[chromosome1Index] = (newChromosomesList[chromosome1Index][:crossoverPoint]
                                                + newChromosomesList[chromosome2Index][crossoverPoint:])

        newChromosomesList[chromosome2Index] = (newChromosomesList[chromosome2Index][:crossoverPoint]
                                                + newChromosomesList[chromosome1Index][crossoverPoint:])

        crossoverIdxs.remove(chromosome1Index)
        crossoverIdxs.remove(chromosome2Index)

# mutation = a random bit from a chromosome is flipped
def mutation(popSize, mutationProb, newChromosomesList):

    mutationIndexes = []

    for i in range(popSize):

        uniformNumber = random()
        if uniformNumber < mutationProb:
            mutationIndexes.append(i)

    for chromoIndex in mutationIndexes:
        bit = randint(0, bitCount - 1)
        if newChromosomesList[chromoIndex][bit] == 0:
            newChromosomesList[chromoIndex][bit] = 1
        else:
            newChromosomesList[chromoIndex][bit] = 0


def plotGraphs(lowBound, upBound, a, b, c, medianFitnesses):

    fig, axis = plt.subplots(1, 2)

    x = np.linspace(lowBound, upBound, 1000)
    y = a * x**2 + b * x + c

    axis[0].plot(x, y)
    axis[0].set_title(f"Graph of the quadratic in [{lowBound} - {upBound}] interval")

    y_lowBnd = quadraticFunction(a, b, c, lowBound)
    y_upBnd = quadraticFunction(a, b, c, upBound)

    x_maxY, maxY = None, None

    parabola_vertex_x = -(b / (2 * a))

    if lowerBound <= parabola_vertex_x <= upperBound and a < 0:
        x_maxY = parabola_vertex_x
        maxY = -(b ** 2 - 4 * a * c) / (4 * a)
    else:

        if y_upBnd > y_lowBnd:
            x_maxY = upBound
            maxY = y_upBnd
        else:
            x_maxY = lowBound
            maxY = y_lowBnd

    axis[0].plot(x_maxY, maxY, 'ro')

    axis[1].plot(list(range(len(medianFitnesses))), medianFitnesses, '.', color="b")
    axis[1].set_title("Median value of the generation")
    axis[1].axhline(y=maxY, color='r', linestyle="dashed", label="Function maximum")

    plt.show()


# reading the input & setting the output file
(populationSize, lowerBound, upperBound, a, b, c, precision,
 crossoverProbability, mutationProbability, stageCount) = readInput()

outputFile = open("evolution.txt", 'w')

# the number of bits needed to encode the numbers from the [a, b] interval (equal to the length of a chromosome)
bitCount = ceil(log2((upperBound - lowerBound) * 10 ** precision))
discreteStep = (upperBound - lowerBound) / 2 ** bitCount

chromosomes = randomInitialPopulation(bitCount, populationSize)

# list of values for graphing the median fitnesses over the stages
medianFitnesses = []

for stage in range(stageCount):

    # decoding the value of each chromosome & evaluating the function in each decoded value
    decodedChromosomesValues = [decode(chromosomes[i], discreteStep, precision) for i in range(populationSize)]
    chromosomeFunctionValues = [quadraticFunction(a, b, c, decodedChromosomesValues[i]) for i in range(populationSize)]

    medianFitness = sum(chromosomeFunctionValues) / populationSize
    medianFitnesses.append(medianFitness)

    # partial fitnesses -> partial sum vector used for selection
    totalFitness = 0.0
    partialFitnesses = [0.0]

    # we keep track of the best chromosome for elitist selection
    maxFitnessIndex = 0
    maxFitness = 0.0

    # constructing the partial fitnesses & finding the chromosome with the best fitness score
    for index, fitness in zip(range(populationSize), chromosomeFunctionValues):
        totalFitness += fitness
        partialFitnesses.append(totalFitness)

        if fitness > maxFitness:
            maxFitness = fitness
            maxFitnessIndex = index

    # for each chromosome we calculate the prob. of being picked then we build the selection intervals from the selection probabilities
    selectionProbabilities = [chromosomeValue / totalFitness for chromosomeValue in chromosomeFunctionValues]
    selectionIntervals = [0.0]
    for probability in selectionProbabilities:
        selectionIntervals.append(selectionIntervals[-1] + probability)

    # generating random numbers from [0,1) and we select the chromosome from that selection interval
    uniformRandomNumbers = [random() for _ in range(populationSize)]
    newChromosomesIndexes = [binarySearch(u, selectionIntervals, 0, populationSize) for u in uniformRandomNumbers]

    # writing the selection probabilities & selection probability intervals &
    # selected chromosomes with their uniform generated numbers

    newChromosomesList = [chromosomes[i] for i in newChromosomesIndexes]

    # writing the new chromosomes
    if stage == 0:
        newDecodedChromosomesValueList = [decode(newChromosomesList[i], discreteStep, precision) for i in range(populationSize)]
        newChromosomeFunctionValueList = [quadraticFunction(a, b, c, newDecodedChromosomesValueList[i]) for i in range(populationSize)]

    # crossover
    crossover(populationSize, crossoverProbability, newChromosomesList)

    # mutation
    mutation(populationSize, mutationProbability, newChromosomesList)

    # the first chromosome will be the best from last generation (elitist selection)
    newChromosomesList[0] = chromosomes[maxFitnessIndex]
    chromosomes = newChromosomesList

plotGraphs(lowerBound, upperBound, a, b, c, medianFitnesses)
