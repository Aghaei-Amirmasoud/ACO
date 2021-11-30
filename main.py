import numpy as np
from numpy.random import choice as np_choice


class AntColony(object):

    def __init__(self, costs, nAnts, nBest, nGeneration, decay, alpha=1, beta=1):
        """
        Args: costs (2D numpy.array): Square matrix of distances.
        nAnts (int): Number of ants running per iteration
        nBest (int): Number of best ants who deposit pheromone
        nGeneration (int): Number of iterations
        decay (float): Rate it which pheromone decays. The pheromone value is multiplied by decay, so 0.95 will lead to
        decay, 0.5 to much faster decay.
        alpha (int or float): exponent on pheromone, higher alpha gives pheromone more weight. Default=1
        beta (int or float): exponent on distance, higher beta give distance more weight. Default=1
        Example: antColony = AntColony(assignment matrix, 100, 20, 2000, 0.95, alpha=1, beta=2)
        """
        self.distances = costs
        self.pheromone = np.ones(self.distances.shape) / len(costs)
        self.allInd = range(len(costs))
        self.nAnts = nAnts
        self.nBest = nBest
        self.nGeneration = nGeneration
        self.decay = decay
        self.alpha = alpha
        self.beta = beta

    def run(self):
        global leastCost
        leastCost = None
        allTimeLeastCost = ("placeholder", np.inf)
        for element in range(self.nGeneration):
            allPermutation = self.generatePermutations()
            self.spreadPheromone(allPermutation, self.nBest)
            leastCost = min(allPermutation, key=lambda x: x[1])
            print("iteration ", element, ": Cost =", leastCost[-1])
            if leastCost[1] < allTimeLeastCost[1]:
                allTimeLeastCost = leastCost
            self.pheromone *= self.decay
        return allTimeLeastCost

    def generatePermutations(self):
        allPermutations = []
        for element in range(self.nAnts):
            per = self.makeChoice(0)
            allPermutations.append((per, self.calculateCost(per)))
        return allPermutations

    def makeChoice(self, start):
        permutation = []
        visited = set()
        visited.add(start)
        prev = start
        for element in range(len(self.distances) - 1):
            move = self.nextMove(self.pheromone[prev], self.distances[prev], visited)
            permutation.append((prev, move))
            prev += 1
            visited.add(move)
        permutation.append((prev, start))
        return permutation

    def nextMove(self, pheromone, dist, visited):
        pheromone = np.copy(pheromone)
        pheromone[list(visited)] = 0

        row = pheromone ** self.alpha * ((1.0 / dist) ** self.beta)

        norm_row = row / row.sum()
        move = np_choice(self.allInd, 1, p=norm_row)[0]
        return move

    def spreadPheromone(self, allPermutation, nBest):
        sortedPermutations = sorted(allPermutation, key=lambda x: x[1])
        for per, dist in sortedPermutations[:nBest]:
            for move in per:
                self.pheromone[move] += 1.0 / self.distances[move]

    def calculateCost(self, permutation):
        totalCost = 0
        for element in permutation:
            totalCost += self.distances[element]
        return totalCost


with open("job3.txt") as textFile:
    matrix = [line.split() for line in textFile]

for i in range(len(matrix)):
    for j in range(len(matrix[i])):
        matrix[i][j] = int(matrix[i][j])

distances = np.array(matrix)

antColony = AntColony(distances, nAnts=10, nBest=1,
                      nGeneration=200, decay=0.1,
                      alpha=0.6, beta=100)
leastCost = antColony.run()
print("best assignment: {}".format(leastCost))