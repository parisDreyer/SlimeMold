import random
import enum
import numpy
import abc

class Chromosome(enum.Enum):
    a = 0
    b = 1
    c = 2
    d = 3
    e = 4
    f = 5
    g = 6
    h = 7
    i = 8
    j = 9
    k = 10
    l = 11
    m = 12
    n = 13
    o = 14
    p = 15
    q = 16

    @classmethod
    def random(cls):
        return random.choice(list(cls.__members__.values()))

class ChromosomePair(object):
    def __init__(self, firstChromosome: Chromosome, secondChromosome: Chromosome):
        self.firstChromosome = firstChromosome
        self.secondChromosome = secondChromosome

    def randomChromosome(self, mutation: Chromosome) -> Chromosome:
        if random.random() < 0.5:
            return mutation
        else:
            return self.firstChromosome if random.random() > 0.5 else self.secondChromosome


class Utilities(object):
    @classmethod
    def indicesOfMaxValues(cls, array: [float]) -> [int]:
        indices = []
        maxValue = max(array)
        length = len(array)
        for x in range(0, length):
            if array[x] == maxValue:
                indices.append(x)
        return indices

    @classmethod
    def chromosomePairsFrom(cls, chromosomes: [Chromosome]) -> [ChromosomePair]:
        chromosomePairs: [ChromosomePair] = []
        half = int(len(chromosomes) / 2 - 1)
        for i in range(0, half - 1, 1):
            chromosomePairs.append(ChromosomePair(chromosomes[i], chromosomes[i + half]))
        return chromosomePairs

class Influence(object):
    def __init__(self, effectedAreas: [float]):
        self.effectedAreas = effectedAreas

    def effectSize(self) -> int:
        return len(self.effectedAreas)
    
    def chromosome(self, importanceMap: [float]) -> Chromosome:
        chromosomesWithImportanceScore = self.shapeImportanceMapToFitChromosomes(importanceMap) * self.shapeImportanceMapToFitChromosomes(self.effectedAreas)
        # best fit random chromosome
        mostAffectedChromosomeIndices = Utilities.indicesOfMaxValues(chromosomesWithImportanceScore)
        hasAnyChromosomes = len(mostAffectedChromosomeIndices) > 0
        if hasAnyChromosomes:
            # for now return a random choice of one relevant chromosome, later refactor to multiple affected chromosomes
            semiRandomChromosomeIndex = random.sample(mostAffectedChromosomeIndices, 1)[0]
            return Chromosome(semiRandomChromosomeIndex)
        else:
            return Chromosome.random()

    def shapeImportanceMapToFitChromosomes(self, importanceMap: [float]) -> [float]:
        chromosomeShape = numpy.ones(len(Chromosome))
        reshapedImportanceMap = numpy.outer(chromosomeShape, importanceMap)
        chromosomesWithImportanceScore = numpy.dot(reshapedImportanceMap, importanceMap)
        return chromosomesWithImportanceScore

    def differenceFrom(self, influence) -> float:
        chromosomeFromOutsideInfluence = self.chromosome(influence.effectedAreas)
        chromosomeFromSelfInfluence = influence.chromosome(self.effectedAreas)
        chromosomeDifference = float(abs(chromosomeFromOutsideInfluence.value - chromosomeFromSelfInfluence.value))
        reshapedOutsideInfluence = numpy.dot(influence.effectedAreas, numpy.outer(influence.effectedAreas, self.effectedAreas))
        effectiveDifference = chromosomeDifference / numpy.dot(self.effectedAreas, reshapedOutsideInfluence)
        return effectiveDifference

class Cell(object):
    def __init__(self, pairedChromosomes: [ChromosomePair]):
        self.chromosomes = pairedChromosomes
        self.unpairedChromosomes: [Chromosome] = []

    def produce(self, influences: [Influence], scalingAmount: float = 0.01) -> Influence:
        # produce cell influences from input
        # this is the function that makes order out of chaos and ties influence patterns to previous influences
        # the chromosomes kept in the cell will decide what influence is returned - and the chromosomes in the cell
        #   are decided by selecting the best fit randomly generated cells through
        #   an efficacy score evaluated in the outer mold (that is the system for ranking the improvement of a given cell)
        importanceMap = self.__mutationPriority()
        influenceScore = [0 for chromosome in Chromosome]
        for influence in influences:
            chromosome = influence.chromosome(importanceMap)
            influenceScore[chromosome.value] += scalingAmount
        generatedInfluence = Influence(influenceScore)
        return generatedInfluence

    def mutate(self, influence: Influence):
        importanceMap = self.__mutationPriority()
        prioritizedChromosome = influence.chromosome(importanceMap)
        self.unpairedChromosomes.append(prioritizedChromosome)
    
    def geneticMaterialWith(self, cell) -> [Chromosome]:
        offspringChromosomes = self.__randomChromosomes() + cell.__randomChromosomes()
        return offspringChromosomes

    # returns a scaling vector to magnify or minimize the effect of an influence value
    def __mutationPriority(self, scalingAmount: float = 0.01) -> [float]:
        # the more instances of a given chromosome in the affected area, the less important it is to add more of that chromosome
        effectedChromosomePriorities = numpy.ones(len(Chromosome))
        for chromosome in self.unpairedChromosomes:
            effectedChromosomePriorities[chromosome.value] -= scalingAmount
        return effectedChromosomePriorities

    def __randomChromosomes(self, fraction: float = 0.5) -> [Chromosome]:
        subsetCount = int(round(len(self.chromosomes) * fraction))
        sampleSet = random.sample(self.chromosomes, subsetCount)
        if len(self.unpairedChromosomes) > 0:
            return [pair.randomChromosome(random.choice(self.unpairedChromosomes)) for pair in sampleSet]
        else:
            return [pair.randomChromosome(Chromosome.random()) for pair in sampleSet]

class ExternalFactors(abc.ABC):
    @abc.abstractmethod
    def fitnessScore(self, cell: Cell):
        pass

class SlimeMold(object):
    def __init__(self, firstCell: Cell):
        self.cells = [firstCell]

    def getCells(self) -> [Cell]:
        return self.cells

    def propogate(self):
        self.cells = self.__reproduce() + self.__reproduce()

    def mutate(self, influence: Influence, fraction: float = 0.4):
        subsetCount = int(round(len(self.cells) * fraction))
        affectedCells = random.sample(self.cells, subsetCount)
        for cell in affectedCells:
            cell.mutate(influence)

    def produce(self, influences: [Influence]) -> [Influence]:
        # produce mold influences from input
        return [cell.produce(influences) for cell in self.cells]

    def __reproduce(self) -> [Cell]:
        return [
            Cell(
                Utilities.chromosomePairsFrom(
                    cell.geneticMaterialWith(random.choice(self.cells))
                    )
                )
            for cell in self.cells
            ]

    def cullByFitnessScore(self, environment: ExternalFactors, populationDeclineRate: float = 0.2):
        # remove cells that have worse fitness score
        # environment evaluates each cell for fitness
        # sort cells by `environment.fitnessScore(cell: self.cells[index])`` and then remove the bottom `populationDeclineRate` percentile
        cellsByFitnessScore = [(environment.fitnessScore(cell), cell) for cell in self.cells]
        #  cells sorted by least erro to greatest error
        cellsByFitnessScore.sort(key=lambda tupl: tupl[0])
        numberOfCellsThatWillDie = int(len(cellsByFitnessScore) * populationDeclineRate)
        tokenCellDeath = range(0, numberOfCellsThatWillDie)
        [cellsByFitnessScore.pop() for death in tokenCellDeath]
            
        self.cells = [survivingCell[1] for survivingCell in cellsByFitnessScore]

class Environment(ExternalFactors):
    # make sure inputs and targetOutputs are the same shape
    def __init__(self, inputs: [[float]], targetOutputs: [[float]]):
        self.influences = [Influence(input) for input in inputs]
        self.targetOutputs = [Influence(output) for output in targetOutputs]

        chromosomes = [influence.chromosome(numpy.ones(influence.effectSize())) for influence in self.influences]
        if len(chromosomes) % 2 is 1 and len(chromosomes) > 1:
            chromosomes.pop()
        else:
            chromosomes += chromosomes
        chromosomePairs = Utilities.chromosomePairsFrom(chromosomes)
        self.slimeMold = SlimeMold(Cell(chromosomePairs))

    def go(self, numberOfGenerations: int = 400, maxEpochs: int = 60000, targetFitness: float = 0.05):
        fitnessScores: [float] = [1]
        currentEpoch: int = 0
        while currentEpoch < maxEpochs and fitnessScores[-1] > targetFitness:
            self.run(numberOfGenerations)
            self.run(numberOfGenerations)
            self.slimeMold.cullByFitnessScore(self)
            print(f'Current Fitness: {fitnessScores[-1]}')
            print(f'Target Fitness: {targetFitness}')
            print(f'Epoch Number: {currentEpoch}')
            print(f'Final Epoch: {maxEpochs}')
            print('--------------=)--------------')

            fitnessScores.append(self.aggregateFitness())
            currentEpoch += 1

    def run(self,  numberOfGenerations: int):
        self.trainTheSlime(numberOfGenerations)
        self.multiplyTheSlime()


    def trainTheSlime(self, numberOfGenerations: int):
        for _ in range(0, numberOfGenerations):
            for influence in self.influences:
                self.slimeMold.mutate(influence)

    def multiplyTheSlime(self):
        self.slimeMold.propogate()

    def aggregateFitness(self) -> float:
        slimeMoldCells = self.slimeMold.getCells()
        fitness = [self.fitnessScore(cell) for cell in slimeMoldCells]
        return numpy.average(fitness)

    # the closer to zero the better the score
    def fitnessScore(self, cell: Cell) -> float:
        # evaluate cell
        producedInfluence = cell.produce(self.influences)
        delta = self.differenceFromTargetOutput(producedInfluence)
        percentageError = 1 / delta if delta != 0 else 1
        return 1 - percentageError

    def differenceFromTargetOutput(self, influence: Influence) -> float:
        # calc average diff between influence and self.targetOutputs
        differenceScores = [influence.differenceFrom(target) for target in self.targetOutputs]
        # consider using the cell's importanceMap as weights for calculating this average
        return numpy.average(differenceScores)
