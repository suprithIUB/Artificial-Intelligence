import robotparser

#__author__ = 'suprchan'
#Name: Suprith Chandrashekharachar
#Username: suprchan

# This file contains support code for B551 Hw2                                 # File version:  September 19, 2015                                            #
import time
from heapq import *
import functools
import copy
import math
import random
import matplotlib.pyplot as plt
import os.path
import subprocess
import numpy as np
from matplotlib import colors
from collections import deque
from math import sqrt
import sys
from mpl_toolkits.mplot3d import Axes3D

infinity = float("inf")

class RobotWorld:
    "The idea is that a RobotWorld is the datatype that goes in the queue."
    def __init__(self,width,length,height,initial,goal):
        self.hand = {'location' : (0,0,0), 'held' : None, 'closed?' : False}
        self.width, self.length, self.height = width, length, height
        self.blocks, self.goal = initial, goal
        self.cost, self.title = 0, ''
        assert(initial.keys() == goal.keys()) #there can't be blocks without a goal state, or goal states with a block that is not initialized.
        self.colormap, possible_colors = {}, set(colors.cnames.keys()) #for the sake of the visualization, not important to the function of RobotWorlds.
        for blockname in self.blocks:
            self.colormap[blockname] = possible_colors.pop()


    def __lt__(self,other):

        "Not meaningful, but necessary for RobotWorld to interact with a heapq"
        return True
    # The actions return the change in cost
    def moveUp(self):
        (x,y,z) = self.hand['location']
        if z < (self.height - 1):
            self.hand['location'] = (x,y,z+1)
            if self.hand['held']: self.blocks[self.hand['held']] = (x,y,z+1)
            self.cost += 1.0
            return 1.0
        else:
            print("Why is the 'moveUp' action occuring? The hand is as far up as it can go.")
            return 0
    def moveDown(self):
        (x,y,z) = self.hand['location']
        if z > 0:
            self.hand['location'] = (x,y,z-1)
            if self.hand['held']: self.blocks[self.hand['held']] = (x,y,z-1)
            self.cost += 1.0
            return 1.0
        else:
            print("Why is the 'moveDown' action occuring? The hand is on the floor.")
            return 0
    def moveLeft(self):
        (x,y,z) = self.hand['location']
        if y > 0:
            self.hand['location'] = (x,y-1,z)
            if self.hand['held']: self.blocks[self.hand['held']] = (x,y-1,z)
            self.cost += 1.0
            return 1.0
        else:
            print("Why is the 'moveLeft' action occuring? The hand is on the left edge.")
            return 0
    def moveRight(self):
        (x,y,z) = self.hand['location']
        if y < (self.length - 1):
            self.hand['location'] = (x,y+1,z)
            if self.hand['held']: self.blocks[self.hand['held']] = (x,y+1,z)
            self.cost += 1.0
            return 1.0
        else:
            print("Why is the 'moveRight' action occuring? The hand is on the right edge.")
            return 0
    def moveForward(self):
        (x,y,z) = self.hand['location']
        if x < (self.width - 1):
            self.hand['location'] = (x+1,y,z)
            if self.hand['held']: self.blocks[self.hand['held']] = (x+1,y,z)
            self.cost += 1
            return 1.0
        else:
            print("Why is the 'moveForward' action occuring? The hand is on the front edge.")
            return 0
    def moveBackward(self):
        (x,y,z) = self.hand['location']
        if x > 0:
            self.hand['location'] = (x-1,y,z)
            if self.hand['held']: self.blocks[self.hand['held']] = (x-1,y,z)
            self.cost += 1
            return 1.0
        else:
            print("Why is the 'moveBackward' action occuring? The hand is on the back edge.")
            return 0
    # the reason for the handOpen and handClose actions to have non-zero costs is to prevent the search from considering the "I'll close my hand on this block" and "I'll open and close my hand on this block 20 million times" as equivalent.
    def handOpen(self):
        if not self.hand['closed?']:
            print("Why is the 'handOpen' action occuring? The hand is already open.")
            return 0
        self.hand['closed?'] = False
        if self.hand['held']:
            self.hand['held'] = None
            self.cost += 0.1
            return 0.1
    def handClose(self):
        if self.hand['closed?']:
            print("Why is the 'handClose' action occuring? The hand is already closed.")
            return 0
        else:
            for (name,location) in self.blocks.iteritems():
                if location == self.hand['location']:
                    self.hand['held'] = name
                    self.hand['closed?'] = True
                    self.cost += 0.1
                    return 0.1
            print("Why did the 'handClose' action occur? There is no block here at {}".format(str(self.hand['location'])))
            return 0
    def isGoal(self):
        return self.blocks == self.goal
    def allowedActions(self):
        def alreadyThere(coord):
            for (block_name, block_coord) in self.blocks.iteritems():
                if (block_coord == coord) and (block_coord == self.goal[block_name]):
                    return True
            return False
        possibilities = ['close','forward','backward','left','right','up','down']
        if self.hand['closed?'] and alreadyThere(self.hand['location']): #try to open first if its a good idea
            possibilities = ['open'] + possibilities
        # Removing the close action if 'alreadyThere' is True could lead to problems in a 3D world, or even rare scenarios in a 2D world.
        # This is not relevant to the worlds here, but this should be removed if we re-use this for planning.
        if self.hand['closed?'] or (self.hand['location'] not in self.blocks.values()) or (alreadyThere(self.hand['location'])):
            possibilities.remove('close')
        (x,y,z) = self.hand['location']
        if x == 0: possibilities.remove('backward')
        if x == (self.length - 1): possibilities.remove('forward')
        if y == 0: possibilities.remove('left')
        if y == (self.width - 1): possibilities.remove('right')
        if z == 0: possibilities.remove('down')
        if z == (self.height - 1): possibilities.remove('up')
        if self.hand['closed?'] and ('open' not in possibilities): #try to put 'open' at the end, so moving around happens before
            possibilities = possibilities + ['open']               #pointless opening and closing.
        return possibilities
    def do(self,action):
        '''action is a string indicating an action for the RobotWorld to do. These strings come from the 'allowedActions' method, and are part of the process of
           iterating over neighboring nodes in the graph.'''
        if (action == 'up'): return self.moveUp()
        elif (action == 'down'): return self.moveDown()
        elif (action == 'left'): return self.moveLeft()
        elif (action == 'right'): return self.moveRight()
        elif (action == 'forward'): return self.moveForward()
        elif (action == 'backward'): return self.moveBackward()
        elif (action == 'open'): return self.handOpen()
        elif (action == 'close'): return self.handClose()
        else: print("Unexpected action {}".format(action))
    def visualize(self,serial_number):
        '''The blocks are colored according to their identity (exactly one color per block). These are persistent over time, so a block with a certain name in one frame of the animation will have the same color in all other frames.
           The triangles of the same color as a block indicate where the block's goal location is.
           The title 'cost=<number>' at the top indicates the cumulative cost of actions so far to bring the RobotWorld into its current state.
           The cross indicates the location of the robot hand.
        '''
        fig = plt.figure()
        axis = fig.add_subplot(111,projection='3d')
        axis.set_xlim(0, self.width)
        axis.set_ylim(0, self.length)
        axis.set_zlim(0, self.height)
        axis.set_xlabel('X Axis')
        axis.set_ylabel('Y Axis')
        axis.set_zlabel('Z Axis')
        n = len(self.blocks)
        for block in self.blocks.iteritems():
            blockname, coord = block
            axis.scatter(coord[0], coord[1], coord[2], s=500, marker=(4,0), c=self.colormap[blockname])
            target = self.goal[blockname]
            axis.scatter(target[0], target[1], target[2], s=300, marker=(3,0), c=self.colormap[blockname])
        robo_x, robo_y, robo_z = self.hand['location']
        axis.scatter(robo_x,robo_y,robo_z,s=500,marker='+', c='black')
        axis.set_title("cost={}".format(str(self.cost)))
        plt.savefig("snapshots/{}_{}.png".format(self.title,str(serial_number)))
        plt.close()

def manhattandistance(self, held = None):
    pointA = self.hand['location']
    if held == None:
        if len(self.blocks.keys()) == 1:
            if self.blocks['A'] != self.goal['A']:
                (x,y,z) = self.hand['location']
                (l,m,n) = self.blocks['A']
                return abs(x-l) + abs(y-m) + abs(z-n)
        elif len(self.blocks.keys()) == 2:
            if self.blocks['A'] != self.goal['A']:
                (x,y,z) = self.hand['location']
                (l,m,n) = self.blocks['A']
                return abs(x-l) + abs(y-m) + abs(z-n)
            if self.blocks['B'] != self.goal['B']:
                (x,y,z) = self.hand['location']
                (l,m,n) = self.blocks['B']
                return abs(x-l) + abs(y-m) + abs(z-n)
        elif len(self.blocks.keys()) == 3:
            if self.blocks['A'] != self.goal['A']:
                (x,y,z) = self.hand['location']
                (l,m,n) = self.blocks['A']
                return abs(x-l) + abs(y-m) + abs(z-n)
            if self.blocks['B'] != self.goal['B']:
                (x,y,z) = self.hand['location']
                (l,m,n) = self.blocks['B']
                return abs(x-l) + abs(y-m) + abs(z-n)
            if self.blocks['C'] != self.goal['C']:
                (x,y,z) = self.hand['location']
                (l,m,n) = self.blocks['C']
                return abs(x-l) + abs(y-m) + abs(z-n)
    else:
        (x,y,z) = self.hand['location']
        (l,m,n) = self.goal[self.hand['held']]
        return abs(x-l) + abs(y-m) + abs(z-n)

def combination(robotarm):
    return manhattanMaxGoal(robotarm) * euclideanDistanceSum(robotarm)

def manhattanMaxBlockGoal_QueuePriority(robotarm):
    if robotarm.hand['held'] == None:
        listOfBlocks = robotarm.blocks.values()
        listOfGoals = robotarm.goal.values()
        hueristicsSum = 0.0
        for i in range(0, len(listOfGoals)):
            if listOfBlocks[i] != listOfGoals[i]:
                pointA = listOfBlocks[i]
                pointB = listOfGoals[i]
                (x,y,z) = pointA
                (l, m, n) = pointB
                hueristicsSum += (abs(x-l) + abs(y-m) + abs(z-n))
        return hueristicsSum
    else:
        (x,y,z) = robotarm.hand['location']
        (l,m,n) = robotarm.goal[robotarm.hand['held']]
        return abs(x-l) + abs(y-m) + abs(z-n)


def manhattan(robotarm):
    pointA = robotarm.hand['location']
    pointB = (0,0,0)
    if robotarm.hand['held'] == None: #if arm is not holding anything, pick the first ungoaled block
        listOfGoals = robotarm.goal.values()
        listOfBlocks = robotarm.blocks.values()
        for item in listOfBlocks:
            if item not in listOfGoals:
                pointB = item
    else:
        if robotarm.blocks[robotarm.hand['held']] != robotarm.goal[robotarm.hand['held']]: # arm holding a block due to Close action via neihbor states
            pointB = robotarm.blocks[robotarm.hand['held']]
        pointB = robotarm.goal[robotarm.hand['held']] # arm is holding a block and moving towards the goal
    (x,y,z) = pointA
    (l,m,n) = pointB
    return abs(x-l) + abs(y-m) + abs(z-n)


def manhattan_min(robotarm):
    pointA = robotarm.hand['location']
    #pointB = (0,0,0)
    hueristicsList = []
    if robotarm.hand['held'] == None: #if arm is not holding anything, pick the first ungoaled block
        listOfBlocks = robotarm.blocks.values()
        (x,y,z) = pointA
        for item in listOfBlocks:
            (l,m,n) = item
            hueristicsList.append(abs(x-l) + abs(y-m) + abs(z-n))
        return min(hueristicsList)
    else:
        if robotarm.blocks[robotarm.hand['held']] != robotarm.goal[robotarm.hand['held']]: # arm holding a block due to Close action via neihbor states
            pointB = robotarm.blocks[robotarm.hand['held']]
        else:
            pointB = robotarm.goal[robotarm.hand['held']] # arm is holding a block and moving towards the goal
        (x,y,z) = pointA
        (l,m,n) = pointB
        return abs(x-l) + abs(y-m) + abs(z-n)

def manhattanMaxGoal(robotarm):

    if robotarm.hand['held'] == None:
        pointA = robotarm.hand['location']
        pointB = (0,0,0)
        hueristicsSum = 0.0
        listOfGoals = robotarm.goal.values()
        (x,y,z) = pointA
        for item in listOfGoals:
            (l, m, n) = item
            hueristicsSum += (abs(x-l) + abs(y-m) + abs(z-n))
        return hueristicsSum
    else:
        (x,y,z) = robotarm.hand['location']
        (l,m,n) = robotarm.goal[robotarm.hand['held']]
        return abs(x-l) + abs(y-m) + abs(z-n)

def manhattanNearestGoal(robotarm):
    pointA = robotarm.hand['location']
    pointB = (0,0,0)
    hueristicsSum = 0.0
    listOfGoals = robotarm.goal.values()
    (x,y,z) = pointA
    min(listOfGoals)
    for item in listOfGoals:
        (l, m, n) = item
        hueristicsSum += (abs(x-l) + abs(y-m) + abs(z-n))
    return hueristicsSum

def chebyshevDistance(robotarm):
    pointA = robotarm.hand['location']
    pointB = (0,0,0)
    hueristicsSumList = []
    listOfGoals = robotarm.goal.values()
    (x,y,z) = pointA
    for item in listOfGoals:
        (l, m, n) = item
        hueristicsSumList.append(abs(x-l) + abs(y-m) + abs(z-n))
    return max(hueristicsSumList)

def manhattanMaxBlockGoal(robotarm):
    listOfBlocks = robotarm.blocks.values()
    listOfGoals = robotarm.goal.values()
    hueristicsSum = 0.0
    for i in range(0, len(listOfGoals)):
        if listOfBlocks[i] != listOfGoals[i]:
            pointA = listOfBlocks[i]
            pointB = listOfGoals[i]
            (x,y,z) = pointA
            (l, m, n) = pointB
            hueristicsSum += (abs(x-l) + abs(y-m) + abs(z-n))
    return hueristicsSum



def euclideanDistanceSum(robotarm):
    listOfBlocks = robotarm.blocks.values()
    listOfGoals = robotarm.goal.values()
    hueristicsSum = 0.0
    for i in range(0, len(listOfGoals)):
        if listOfBlocks[i] != listOfGoals[i]:
            pointA = listOfBlocks[i]
            pointB = listOfGoals[i]
            (x,y,z) = pointA
            (l, m, n) = pointB
            hueristicsSum += sqrt(sum((pointA-pointB)**2 for pointA, pointB in zip(pointA, pointB)))
    return hueristicsSum

def euclideandistance_iterative(pointA, pointB):
    return sqrt(sum((pointA-pointB)**2 for pointA, pointB in zip(pointA, pointB)))

def euclideandistance_new(self, held = None):
    pointA = self.hand['location']
    pointB = (0,0,0)
    if held == None:
        if len(self.blocks.keys()) == 1:
            if self.blocks['A'] != self.goal['A']:
                pointB = self.blocks['A']
        elif len(self.blocks.keys()) == 2:
            if self.blocks['A'] != self.goal['A']:
               pointB = self.blocks['A']
            if self.blocks['B'] != self.goal['B']:
                pointB = self.blocks['B']
        elif len(self.blocks.keys()) == 3:
            if self.blocks['A'] != self.goal['A']:
                pointB = self.blocks['A']
            if self.blocks['B'] != self.goal['B']:
               pointB = self.blocks['B']
            if self.blocks['C'] != self.goal['C']:
                pointB = self.blocks['C']

    return sqrt(sum((pointA-pointB)**2 for pointA, pointB in zip(pointA, pointB)))

def graphsearch(queue,queue_modification,timeout,visualize=False):
    "The things that I can think to say about graphsearch are mentioned in the directions on the website. If you have questions, email me at aseewald@indiana.edu"
    t0 = time.time()
    visited, history = [], []
    solution_info = {'cost' : infinity, 'num_expanded' : infinity, 'walltime' : infinity}
    while len(queue) > 0:
        if timeout == 0:
            print("Ran out of time for thinking")
            print("The queue is of size: {}".format(len(queue)))
            return solution_info
        queue, expanded = queue_modification(queue,visited)
        visited.append(expanded)
        if expanded.isGoal():
            tf = time.time()
            history.append(expanded)
            solution_info['cost'] = expanded.cost
            solution_info['num_expanded'] = len(history)
            solution_info['walltime'] = tf - t0
            print("The solution's cost is: {}, and involved expanding {} verticies".format(str(expanded.cost), str(len(history))))
            print("Finding the solution involved {} seconds of walltime".format(str(tf - t0)))
            print("----------------------------------------------------------------------------------")
            if visualize:
                for (i, world) in enumerate(history):
                    print i, world.hand
                    world.visualize(i)
            return solution_info
        else:
            timeout -= 1
            history.append(expanded)
    print("No possible actions left. And you may ask yourself, well, how did I get here? You may tell yourself, this is not my beautiful search algorithm.")
    return solution_info

def duplicateWorld(world,worlds):
    "You may want to test if a world is in some visited set. This function is needed to define a more loose idea of equality, so that worlds with different accumulated costs are considered equivalent."
    for w in worlds:
        if (w.hand == world.hand) and (w.blocks == world.blocks):
            return True
    return False

def dummyExample(queue,visited):
    "Shows the mechanics of a RobotWorld's 'do', and 'allowedActions' methods, as well as how to use a queue."
    easy = RobotWorld(5,5,1,{'A' : (1,0,0), 'B' : (3,1,0)}, {'A' : (1,1,0), 'B' : (3,1,0)})
    if easy.isGoal():
        print("You've found the solution")
    queue = []
    heappush(queue, (0, easy)) #this is a minqueue where '0' is the key.
    for action in easy.allowedActions():
        neighbor = copy.deepcopy(easy)
        deltaCost = neighbor.do(action)
        print("You now have a neighbor of the start state. What should you do with it?")

def breadthFirst(queue,visited):
    "For you to do, define this"

    robotworld_tuple = heappop(queue)
    robotworld = robotworld_tuple[1]
    tp = robotworld.hand['location']

    if robotworld.isGoal():
        robotworld = copy.deepcopy(robotworld)
        print "Goal found at", tp
        return queue, robotworld
    for action in robotworld.allowedActions():
        nextLoc = copy.deepcopy(robotworld)
        nextLoc.do(action)
        duplicate = duplicateWorld(nextLoc, visited)
        if not duplicate:
            heappush(queue, (len(queue), nextLoc))
    return queue, robotworld





def depthFirst(queue,visited):
    currentPrio = queue[0][0]
    robotworld = heappop(queue)[1]
    openList = []
    for item in queue:
        openList.append(item[1])
    if robotworld.isGoal():
        return queue, robotworld
    for action in robotworld.allowedActions():
        nextLoc = copy.deepcopy(robotworld)
        nextLoc.do(action)
        if not duplicateWorld(nextLoc, visited) and not duplicateWorld(nextLoc, openList):
            currentPrio -= 1
            heappush(queue, (currentPrio, nextLoc))
    return queue, robotworld



def bestFirst(queue,visited,heuristic):
    "For you to do, define this"
    openList = list()
    for item in queue:
        openList.append(item[1])
    robotarm = heappop(queue)[1]
    if robotarm.isGoal():
        print "Goal Found"
        return queue, robotarm
    visited.append(robotarm)
    neighbors = []


    for move in robotarm.allowedActions():
        nextState = copy.deepcopy(robotarm)
        dist = nextState.do(move)
        if duplicateWorld(nextState, visited):
            continue

        if not duplicateWorld(nextState, openList):

            nextState_fscore = heuristic(nextState)
            if heuristic.func_name == 'manhattanMaxBlockGoal_QueuePriority':
                    if(move == 'close') and (nextState.hand['held'] is not None) and (robotarm.hand['held'] is None):

                        for idx in range(0, len(queue)):
                            updateState = queue[idx]
                            new_prio = nextState_fscore+updateState[0]
                            queue[idx] = (new_prio, updateState[1])
                        heapify(queue)
            if not duplicateWorld(nextState, openList):
                heappush(queue, (nextState_fscore, nextState))

    return queue, robotarm


def aStar(queue,visited,heuristic):
    "For you to do, define this"
    openList = list()
    for item in queue:
        openList.append(item[1])
    robotarm = heappop(queue)[1]
    if robotarm.isGoal():
        print "Goal Found"
        return queue, robotarm
    visited.append(robotarm)
    neighbors = []


    for move in robotarm.allowedActions():
        nextState = copy.deepcopy(robotarm)
        dist = nextState.do(move)
        if duplicateWorld(nextState, visited):
            continue

        else:
            estimateScore = robotarm.cost + dist
            if not duplicateWorld(nextState, openList) or (estimateScore < nextState.cost):

                nextState.cost = estimateScore
                nextState_fscore = nextState.cost + heuristic(nextState)
                if heuristic.func_name == 'manhattanMaxBlockGoal_QueuePriority':
                    if(move == 'close') and (nextState.hand['held'] is not None) and (robotarm.hand['held'] is None):

                        for idx in range(0, len(queue)):
                            updateState = queue[idx]
                            new_prio = nextState_fscore+updateState[0]
                            queue[idx] = (new_prio, updateState[1])
                        heapify(queue)
                if not duplicateWorld(nextState, openList):

                    heappush(queue, (nextState_fscore, nextState))

    return queue, robotarm






def run(world,title,heuristics,timeout=5000):
    solutions = []
    queue = []
    world0 = copy.deepcopy(world)
    world0.title=(title + '_DFS')
    heappush(queue,(0,world0))
    print("Doing Depth First Search  on {}:".format(title))

    solutions.append(graphsearch(queue, depthFirst, timeout))

    queue = []
    world1 = copy.deepcopy(world)
    world1.title=(title + '_BFS')
    heappush(queue,(0,world1))
    print("Doing Breadth First Search on {}:".format(title))
    solutions.append(graphsearch(queue, breadthFirst, timeout))

    for h in heuristics:
        queue, hname = [], str(h).split(' ')[1]
        world2 = copy.deepcopy(world)
        world2.title=(title + hname + '_BestFirst')
        heappush(queue,(0,world2))
        bestFirst_h = functools.partial(bestFirst,heuristic=h)
        print("Doing Best First with heuristic {} on {}:".format(hname,title))
        solutions.append(graphsearch(queue, bestFirst_h, timeout))

        queue = []
        world4 = copy.deepcopy(world)
        world4.title=(title + hname + '_Astar')
        heappush(queue,(0,world4))
        aStar_h = functools.partial(aStar,heuristic=h)
        print("Doing A* with heuristic {} on {}:".format(hname,title))
        solutions.append(graphsearch(queue, aStar_h, timeout))
    return solutions

def experiment(heuristics,described_worlds):
    data = [run(world,title,heuristics,timeout=10000) for (world,title) in described_worlds]
    fig, axes = plt.subplots(nrows=len(described_worlds))
    fig.set_size_inches(20,30)
    if len(described_worlds) == 1: # a hack to get consistent indexing behavoir from matplotlib.
        axes = [axes]
    color_codes = random.sample(colors.cnames.keys(), len(described_worlds) * (2 + 2 * len(heuristics)))
    labels = ['DFS','BFS']
    for heuristic in heuristics:
        hname = str(heuristic).split(' ')[1]
        labels.append('Best_' + hname)
        labels.append('A*_' + hname)
    # set the legend of the figure to correspond to colors.
    for (i, solutions) in enumerate(data): #2 because discard breadthFirst and depthFirst b/c no heuristic.
        axes[i].set_title(described_worlds[i][1])
        axes[i].set_xlabel("verticies expanded")
        axes[i].set_ylabel("cost of solution")
        for (j, solution) in enumerate(solutions):
           print("{} is {}".format(labels[j], color_codes[j]))
           if solution['cost'] == infinity:
               pass
           else:
               axes[i].scatter(solution['num_expanded'],solution['cost'], color=color_codes[j], label=labels[j])
    plt.savefig("experiment.png")
    plt.close()

if not os.path.exists("snapshots"):
    if subprocess.call(["mkdir","snapshots"]) != 0:
        print("Failed to make a directory to store images in.")

hs = {combination, manhattanMaxBlockGoal, manhattanMaxBlockGoal_QueuePriority, euclideanDistanceSum} #this is a set, put your heuristic functions in here. #euclideandistance
presolved = RobotWorld(4,4,1,{'A' : (3,3,0), 'B' : (3,1,0)}, {'A' : (3,3,0), 'B' : (3,1,0)})
# run(presolved,'presolved',hs)
easy = RobotWorld(5,5,1,{'A' : (1,0,0), 'B' : (3,1,0)}, {'A' : (1,1,0), 'B' : (3,1,0)})
# run(easy,'easy',hs)
medium = RobotWorld(6,6,1,{'A' : (1,1,0), 'B' : (3,1,0)}, {'A' : (4,4,0), 'B' : (4,5,0)})
# run(medium,'medium',hs)
hard = RobotWorld(10,10,1,{'A' : (1,0,0), 'B' : (9,9,0), 'C' : (4,4,0)}, {'A' : (4,4,0), 'B' : (1,0,0), 'C' : (9,9,0)})
# run(hard,'hard',hs)
experiment(hs, [(presolved,'presolved'), (easy, 'easy'), (medium, 'medium'), (hard, 'hard')])

