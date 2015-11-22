import gamePlay
import sys
import time
from math import sqrt
from copy import deepcopy
from getAllPossibleMoves import getAllPossibleMoves
import threading

'''
Name: Suprith Chandrashekharachar
Username: suprchan
Minimax with alpha beta pruning
'''


#Map of weights

ratioDict = {
    'pieceRatio': 0.40,
    'targetArea': 0.12,
    'attackers': 0.12,
    'defenders': 0.20,
    'movableKings': 0.06,
    'safePawns': 0.02,
    'safeKings': 0.02,
    'pawnsOnDiagonal': 0.08,
    'rampageMode': 0.00 
    }

#Map for storing each indvidual heurisitc's score
scoreMap = {
    'pieceRatio': 0.0,
    'makeKings': 0.0,
    'attackers': 0.0,
    'defenders': 0.0,
    'movableKings': 0.0,
    'safePawns': 0.0,
    'safeKings': 0.0,
    'pawnsOnDiagonal': 0.0,
    'killLoners': 0.0
    }


'''
Class: HeuristicThread
Description: Class extending threading module to help in thread creation
'''

class HeuristicThread(threading.Thread):
    def __init__(self, target, *args):
        self._target = target
        self._args = args
        threading.Thread.__init__(self)

    def run(self):
        self._target(*self._args)


'''
Funtion Name: evaluate
Description: Evaluates a given board state with a group of heurisitc functions.
Returns: score
'''

def evaluate(board, color):
    
    listOfThreads = []
    # Spawn a thread for each heuristic function to quicken the evaluation procedure
    for func in heuristicFunctions:
        heuThread = HeuristicThread(func, board, color)
        heuThread.start()
        listOfThreads.append(heuThread)
    for item in listOfThreads:
        item.join()    

    pieceRatio = scoreMap['pieceRatio']
    targetArea = 0.0
    attackMode = False
    rampageMode = False
    normalMode = True
    defensiveMode = False
    
    #Vary game playing modes for pawn movements based on piece ratio

    if pieceRatio < 0.75:
	defensiveMode = True	
    if pieceRatio > 1.50:
        attackMode = True
    elif pieceRatio > 2.25:
        rampageMode = True

    #attack mode weights	
    if attackMode:
        ratioDict['pieceRatio'] = 0.20
        ratioDict['attackers'] = 0.20
        ratioDict['targetArea'] = 0.32
        ratioDict['defenders'] = 0.12
    
    #rampage mode weights
    if rampageMode:
        targetArea = scoreMap['killLoners']
        ratioDict['attackers'] = 0.20
        ratioDict['defenders'] = 0.00
        ratioDict['targetArea'] = 0.32
        ratioDict['pieceRatio'] = 0.20

    #normal mode weights
    if normalMode:
        targetArea = scoreMap['makeKings']
        ratioDict['attackers'] = 0.10
        ratioDict['defenders'] = 0.30
        ratioDict['targetArea'] = 0.12
        ratioDict['pieceRatio'] = 0.40

    #defensive mode weights
    if defensiveMode:
	ratioDict['attackers'] = 0.05
        ratioDict['defenders'] = 0.25
        ratioDict['targetArea'] = 0.05
        ratioDict['pieceRatio'] = 0.65
        
    return ratioDict['pieceRatio']*pieceRatio + \
           ratioDict['attackers']*scoreMap['attackers'] + \
           ratioDict['defenders']*scoreMap['defenders'] + \
	   ratioDict['safePawns']*scoreMap['safePawns'] + \
           ratioDict['safeKings']*scoreMap['safeKings'] + \
           ratioDict['pawnsOnDiagonal']*scoreMap['pawnsOnDiagonal']


'''
Funtion Name: getOpponentsLonerPieces
Description: Gets location of loner pieces of the opponent.
Returns: loner piece list
'''

def getOpponentsLonerPieces(board, color):
    listOfLoners = list()
    opponentColor = gamePlay.getOpponentColor(color)
    
    #find loners of opponent
    for i in range(1,33):
        xy = gamePlay.serialToGrid(i)
        x = xy[0]
        y = xy[1]
        if board[x][y] == opponentColor.upper() or board[x][y] == opponentColor:
            listOfLoners.append(i)
    return listOfLoners


'''
Funtion Name: pawnsOnDiagonal
Description: Get score for pawns on two main diagonal which are in the danger of being captured
Returns: Score for pawns on diagonals
'''


def pawnsOnDiagonal(board, color):
    diagonal = [1,6,10,15,19,24,28,5,9,14,18,23,27,32]
    #lowerDiagonal = [5,9,14,18,23,27,32]
    opponentColor = gamePlay.getOpponentColor(color)
    value = 0

    #loop through pieces on diagonals and find out pawns which are at the risk of being captured
    for item in diagonal:
        xy = gamePlay.serialToGrid(item)
        x = xy[0]
        y = xy[1]
        if board[x][y] == color:
            if gamePlay.isCapturePossibleFromPosition(board, x,y):
                value += 1
        elif board[x][y] == opponentColor:
            if gamePlay.isCapturePossibleFromPosition(board, x,y):
                value -= 1
    scoreMap['pawnsOnDiagonal'] = value
    
    
'''
Funtion Name: getNumberOfMovableKings
Description: Get a ratio of number of kings who can move without capturing. Helps in calculation of heuristic value of move.
Returns: Score for Kings with movable only power.
''' 
def getNumberOfMovableKings(board, color):
    kingOfColor = 0
    kingOfOpponent = 0
    opponentColor = gamePlay.getOpponentColor(color)
    
    #loop through all kings and find out kings that move without capturing
    for i in range(1,33):
        xy = gamePlay.serialToGrid(i)
        x = xy[0]
        y = xy[1]
        if board[x][y] == color.upper():
            if gamePlay.isCapturePossibleFromPosition(board, x,y) == False:
                kingOfColor += 1
        if board[x][y] == opponentColor.upper():
            if gamePlay.isCapturePossibleFromPosition(board, x,y) == False:
                kingOfOpponent += 1

    scoreMap['movableKings'] =  kingOfColor - kingOfOpponent


'''
Funtion Name: euclideanDistance
Description: Get distance from point a to point b with euclidean method. Helps in calculation of heuristic value of move.
Returns: euclidean distance
'''  
def euclideanDistance(pointA, listOfGoals):
    hueristicsSum = 0
    for item in listOfGoals:
        pointB = gamePlay.serialToGrid(item)
        hueristicsSum += sqrt((pointA[0]-pointB[0])**2 + (pointA[1]-pointB[1])**2)
    return hueristicsSum

'''
Funtion Name: manhattanDistance
Description: Get distance from point a to point b with Manhattan method. Helps in calculation of heuristic value of move.
Returns: manhattan distance
'''
def manhattanDistance(tup1, listOfGoals):
    x = tup1[0]
    y = tup1[1]
    hueristicsSum = 0
    for item in listOfGoals:
        (l,m) = gamePlay.serialToGrid(item)
        hueristicsSum += abs(x-l) + abs(y-m)
    return hueristicsSum



'''
Funtion Name: movePawnsToLoners
Descriptions: Moves pawns to loner pieces of the opponent if any. Increases kill power for pawns
Returns: distance measure to loners
'''

def movePawnsToLoners(board, color):
    listOfGoals = getOpponentsLonerPieces(board, color)
    distanceMeasure = 0.0

    #get all pawns to attack the loner pieces
    for i in range(1,33):
                    xy = gamePlay.serialToGrid(i)
                    x = xy[0]
                    y = xy[1]
                    if board[x][y] == color or board[x][y] == color.upper():
                        distanceMeasure += euclideanDistance(xy, listOfGoals)
    scoreMap['killLoners'] =  distanceMeasure        


'''
Funtion Name: movePawnsToOppositeSide
Description: Get a distance measure of pawns to make them kings. Helps in calculation of heuristic value of move.
Returns: sum of distance to opposite side
'''
def movePawnsToOppositeSide(board, color):
    opposideEnd = None;
    distanceMeasure = 0.0
    
    #get distance measure for moving the pawns to opposite side to make them kings
    if color == 'w':
        oppositeEnd = [1,2,3,4]
        for i in range(5,33):
                xy = gamePlay.serialToGrid(i)
                x = xy[0]
                y = xy[1]
                if board[x][y] == 'w':

		    #using euclidean distance as game works with diagonal movements
                    distanceMeasure += euclideanDistance(xy, oppositeEnd)
    else:
        oppositeEnd = [29,30,31,32]
        for i in range(1,28):
                xy = gamePlay.serialToGrid(i)
                x = xy[0]
                y = xy[1]
                if board[x][y] == 'r':
                    distanceMeasure += euclideanDistance(xy, oppositeEnd)
                    
    scoreMap['makeKings'] = distanceMeasure


def getColumn(board, colNumber):
    return [row[colNumber] for row in board]



'''
Funtion Name: safePawns
Descriptions: Number of safe men - men on the edges of the board. Helps in calculation of heuristic value of move.
Returns: number of safe pawns
'''
def safePawns(board, color):
    opponentColor = gamePlay.getOpponentColor(color)
    
    #get count of pawns on four edges for each player
    safePawnsOfPlayer = board[0].count(color) + board[7].count(color) + getColumn(board, 0).count(color) + getColumn(board, 7).count(color)
    safePawnsOfOpponent = board[0].count(opponentColor) + board[7].count(opponentColor) + getColumn(board, 0).count(opponentColor) + getColumn(board, 7).count(opponentColor)
    scoreMap['safePawns'] =  safePawnsOfPlayer-safePawnsOfOpponent


'''
Funtion Name: safeKings
Descriptions: Number of safe kings - kings on the edges of the board. Helps in calculation of heuristic value of move.
Returns: safe kings
'''
def safeKings(board, color):
    opponentColor = gamePlay.getOpponentColor(color)
    
    #get count of kings on four edges for each player
    safeKingsOfPlayer = board[0].count(color.upper()) + board[7].count(color.upper()) + getColumn(board, 0).count(color.upper()) + getColumn(board, 7).count(color.upper())
    safeKingsOfOpponent = board[0].count(opponentColor.upper()) + board[7].count(opponentColor.upper()) + getColumn(board, 0).count(opponentColor.upper()) + getColumn(board, 7).count(opponentColor.upper())
    scoreMap['safeKings'] = safeKingsOfPlayer - safeKingsOfOpponent



'''
Funtion Name: defenders
Descriptions: Number of defenders when compared to opponent's pawns - pawns in 2 edge rows.
Returns: score for defenders 
'''

def defenders(board, color):
    opponentColor = gamePlay.getOpponentColor(color)
    defendersOfPlayer = 0
    defendersOfOpponent = 0
    
    #get count of pawns and kings on edge rows for each player
    if color == 'w':
        defendersOfPlayer = board[0].count(color) + board[0].count(color.upper()) + board[1].count(color) + board[1].count(color.upper())
        defendersOfOpponent =   board[6].count(color) + board[6].count(color.upper()) + board[7].count(color) + board[7].count(color.upper())
    else:
        defendersOfPlayer = board[6].count(color) + board[6].count(color.upper()) + board[7].count(color) + board[7].count(color.upper())
        defendersOfOpponent = board[0].count(color) + board[0].count(color.upper()) + board[1].count(color) + board[1].count(color.upper())
    scoreMap['defenders'] = defendersOfPlayer-defendersOfOpponent


'''
Funtion Name: attackers
Descriptions: Number of attackers when compared to opponent's pawns - pawns in 3 top most rows.
Returns: score for attackers 
'''
def attackers(board, color):
    opponentColor = gamePlay.getOpponentColor(color)
    attackersOfPlayer = 0
    attackersOfOpponent = 0
    
    #get count of pawns and kings on top most rows for each player
    if color == 'w':
        attackersOfPlayer = board[2].count(color) + board[2].count(color.upper()) + board[3].count(color) + board[3].count(color.upper()) + board[4].count(color) + board[4].count(color.upper())
        attackersOfOpponent =   board[5].count(color) + board[5].count(color.upper()) + board[4].count(color) + board[4].count(color.upper()) + board[3].count(color) + board[3].count(color.upper())
    else:
        attackersOfOpponent = board[2].count(color) + board[2].count(color.upper()) + board[3].count(color) + board[3].count(color.upper()) + board[4].count(color) + board[4].count(color.upper())
        attackersOfPlayer =   board[5].count(color) + board[5].count(color.upper()) + board[4].count(color) + board[4].count(color.upper()) + board[3].count(color) + board[3].count(color.upper())
    scoreMap['attackers'] = attackersOfPlayer-attackersOfOpponent

'''
Funtion Name: scoreKingsAndMen
Description: Ratio based on the number of kings and men of both the players. Helps in calculation of heuristic value of move.
Returns: ratio of pieces
'''
def scoreKingsAndMen(board, color):
    opponentColor = gamePlay.getOpponentColor(color)

    value = 0
    # Loop through all board positions
    for piece in range(1, 33):
        xy = gamePlay.serialToGrid(piece)
        x = xy[0]
        y = xy[1]

        if board[x][y].upper() == color.upper():
            value = value + 1
        elif board[x][y].upper() == opponentColor.upper():
            value = value - 1

    scoreMap['pieceRatio'] = value

#List of heurisitc functions
heuristicFunctions = [scoreKingsAndMen, movePawnsToOppositeSide,movePawnsToLoners, attackers, defenders, getNumberOfMovableKings, safePawns, safeKings, pawnsOnDiagonal]
    
'''
Funtion Name: iterativeDeepeningAlphaBetaPruning
Description: Best move given a board state based on alpha-beta iterative deepening method. Given a time limit, decides the depth limit of search and evaluation
Returns: Move
*** This function will be utilized in future submission ***
'''
def iterativeDeepeningAlphaBetaPruning(board, color, player, time, maxRemainingMoves):
    bestMove = None
    best = None
    depth = 12
    
    moves = getAllPossibleMoves(board, color)
    # Set depth limit depending the available time
    if time > 100: depth = 12
    if 50 > time and time < 100: depth = 10
    if 10 > time and time < 50: depth = 7
    if 2 > time and time < 10: depth = 3

    # Evaluate all avaiable moves using alpha-beta pruning with a given depth
    for move in moves:
            newBoard = deepcopy(board)
            gamePlay.doMove(newBoard,move)
            moveVal = alphaBetaPruning(newBoard, color, sys.maxint, -sys.maxint-1, depth) 
            if best == None or moveVal > best:
                bestMove = move
                best = moveVal
    return bestMove



'''
Funtion Name: alphaBetaPruning
Description: Score of the next best move based on a evaluation function
Returns: score
'''       

def alphaBetaPruning(board):
    #Get a list of avaiable moves
	
	# Maximize the player's winning chance
        def maximize(board, alpha, beta, depth):
                global opponentColor
		global myColor
		#Return a heurisitc based score once the depth limit is reached
		if depth <= 0 or not gamePlay.isAnyMovePossible(board, opponentColor):
                        return evaluate(board, myColor)
                score = -sys.maxint-1
                for move in getAllPossibleMoves(board, opponentColor):
                        newBoard = deepcopy(board)
                        gamePlay.doMove(newBoard, move)
                        score = max(score, minimize(newBoard, alpha, beta, depth-1))
			
			#beta cut-off
                        if score >= beta:
                                return score
                        alpha = max(alpha, score)
                return score

        # Minimize the player's losing chance, by considering an opponent's move with the same heuristic s trategy
	def minimize(board, alpha, beta, depth):
		global opponentColor
		global myColor
		#Return a heurisitc based score once the depth limit is reached
                if depth <=0 or not gamePlay.isAnyMovePossible(board, opponentColor):
                        return evaluate(board, myColor)
                score = sys.maxint
                for move in getAllPossibleMoves(board, myColor):
                        newBoard = deepcopy(board)
                        gamePlay.doMove(newBoard, move)
                        score = min(score, maximize(board, alpha, beta, depth-1))
			
			#alpha cut-off
                        if score <= alpha:
                                return score
                        beta = min(beta, score)
                return score


   	depth = 11
	#Start by maximizing the player's winning chance
	val = maximize(board, sys.maxint, -sys.maxint-1, depth)
        return val
		

myColor = None
opponentColor = None

'''
Funtion Name: nextMove
Description: Best move, given a state
Returns: move
'''
def nextMove(board, color, time, movesRemaining):
    
    #Trying to find the move where I have best score
    global myColor
    global opponentColor
    myColor = color
    opponentColor = gamePlay.getOpponentColor(color)
    moves = getAllPossibleMoves(board, color) 
    bestMove = None
    best = None
    for move in moves:
        newBoard = deepcopy(board)
        gamePlay.doMove(newBoard,move)
        #Calling mini-max with alpha-beta pruning

      	moveVal = alphaBetaPruning(newBoard)
        if best == None or moveVal > best:
            bestMove = move
            best = moveVal
    return bestMove

