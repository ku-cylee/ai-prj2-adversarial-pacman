# myTeam.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

from captureAgents import CaptureAgent
import random, time, util
from game import Directions
import game

EPSILON = 1e-10

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first='OffensiveAgent', second='DefensiveAgent'):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --redOpts and --blueOpts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """

    # The following line is an example only; feel free to change it.
    return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########

class BaseAgent(CaptureAgent):
    """
    A Dummy agent to serve as an example of the necessary agent structure.
    You should look at baselineTeam.py for more details about how to
    create an agent as this is the bare minimum.
    """

    def registerInitialState(self, gameState):
        """
        This method handles the initial setup of the
        agent to populate useful fields (such as what team
        we're on).

        A distanceCalculator instance caches the maze distances
        between each pair of positions, so your agents can use:
        self.distancer.getDistance(p1, p2)

        IMPORTANT: This method may run for at most 15 seconds.
        """

        '''
        Make sure you do not delete the following line. If you would like to
        use Manhattan distances instead of maze distances in order to save
        on initialization time, please take a look at
        CaptureAgent.registerInitialState in captureAgents.py.
        '''
        super().registerInitialState(gameState)
        '''
        Your initialization code goes here, if you need any.
        '''


    def chooseAction(self, gameState):
        self.state = gameState.getAgentState(self.index)
        actions = gameState.getLegalActions(self.index)
        values = [self.evaluateAction(gameState, action) for action in actions]
        maxValue = max(values)
        bestActions = [a for a, v in zip(actions, values) if v == maxValue]
        return random.choice(bestActions)


    def evaluateAction(self, gameState, action):
        features = self.getFeatures(gameState, action)
        weights = self.getWeights(gameState, action)
        return features * weights


    def getFeatures(self, gameState, action):
        raise NotImplementedError()


    def getWeights(self, gameState, action):
        raise NotImplementedError()


class OffensiveAgent(BaseAgent):

    def getFeatures(self, gameState, action):
        successor = gameState.generateSuccessor(self.index, action)
        position = successor.getAgentState(self.index).getPosition()

        nearestFoodDistance = self.getNearestFoodDistance(successor, position)
        foodsLeft = self.getFoodsLeft(successor)
        nearestGhostDistance = self.getNearestGhostDistance(successor, position)
        desireToReturn = self.getDesireToReturn(successor, position)

        return util.Counter({
            'nearestFoodDistance': nearestFoodDistance,
            'foodsLeft': foodsLeft,
            'ghostThreat': 1 / (nearestGhostDistance + EPSILON),
            'desireToReturn': desireToReturn,
        })


    def getWeights(self, gameState, action):
        return util.Counter({
            'nearestFoodDistance': -2,
            'foodsLeft': -20,
            'ghostThreat': -19,
            'desireToReturn': -2,
        })


    def getNearestFoodDistance(self, gameState, position):
        foods = self.getFood(gameState).asList()
        return min(self.getMazeDistance(position, food) for food in foods)


    def getNearestGhostDistance(self, gameState, position):
        opponentStates = [gameState.getAgentState(opIdx) for opIdx in self.getOpponents(gameState)]
        ghostPositions = [opState.getPosition() for opState in opponentStates if not opState.isPacman]
        return min(self.getMazeDistance(position, gPos) for gPos in ghostPositions)


    def getDesireToReturn(self, gameState, position):
        foodCarrying = self.state.numCarrying
        if foodCarrying == 0:
            return 0

        walls = gameState.getWalls()
        borderIdx = walls.width // 2 + (-1 if self.red else 0)
        borderPositions = [(borderIdx, hIdx) for hIdx in range(walls.height) if not walls[borderIdx][hIdx]]
        nearestBorderDistance = min(self.getMazeDistance(position, bPos) for bPos in borderPositions)
        return foodCarrying ** 2 * nearestBorderDistance


    def getFoodsLeft(self, gameState):
        foods = self.getFood(gameState).asList()
        return len(foods)


class DefensiveAgent(BaseAgent):

    def getFeatures(self, gameState, action):
        successor = gameState.generateSuccessor(self.index, action)
        position = successor.getAgentState(self.index).getPosition()

        invaderPositions = self.getInvaderPositions(successor)
        invaderDistance = self.getNearestInvaderDistance(invaderPositions, position)
        defendBorder = self.getDefendBorder(successor, position)

        return util.Counter({
            'invaders': len(invaderPositions),
            'invaderDistance': invaderDistance,
            'defendBorder': defendBorder,
        })


    def getInvaderPositions(self, gameState):
        opponentStates = [gameState.getAgentState(opIdx) for opIdx in self.getOpponents(gameState)]
        return [opState.getPosition() for opState in opponentStates if opState.isPacman]


    def getNearestInvaderDistance(self, invaderPositions, position):
        if invaderPositions:
            return min(self.getMazeDistance(position, ePos) for ePos in invaderPositions)
        return 0


    def getDefendBorder(self, gameState, position):
        walls = gameState.getWalls()
        borderIdx = walls.width // 2 + (-1 if self.red else 0)
        borderPositions = [(borderIdx, hIdx) for hIdx in range(walls.height) if not walls[borderIdx][hIdx]]
        return min(self.getMazeDistance(position, bPos) for bPos in borderPositions)


    def getWeights(self, gameState, action):
        return util.Counter({
            'invaders': -100,
            'invaderDistance': -10,
            'defendBorder': -1,
        })
