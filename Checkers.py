import gamePlay
import sys
import time
from math import sqrt
from copy import deepcopy
from getAllPossibleMoves import getAllPossibleMoves
from getAllPossibleMoves import gridToSerial
from getAllPossibleMoves import getAllPossibleMovesAtPosition
import threading
import random
'''
Name: Suprith Chandrashekharachar
Username: suprchan
Minimax with alpha beta pruning
'''


#Map of weights

ratioDict = {
    }

#Map for storing each indvidual heurisitc's score
scoreMap = {
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


    pieces = scoreMap['men'] + scoreMap['kings']
    targetArea = 0.0
    attackMode = False
    rampageMode = False
    normalMode = True
    defensiveMode = False
    
    #Vary game playing modes for pawn movements based on piece ratio

    if pieces < 0:
	defensiveMode = True	
    if pieces >= 5:
        attackMode = True
    elif pieces >= 7:
        rampageMode = True

    #attack mode weights	
    if attackMode:
    	ratioDict['block_promotion'] = 0.12
        ratioDict['men'] = 0.20
        ratioDict['kings'] = 0.25
        ratioDict['defense'] = 0.21
        ratioDict['offense'] = 0.16
	ratioDict['mobility'] = 0.04
    #rampage mode weights
    if rampageMode:
	ratioDict['block_promotion'] = 0.10
        ratioDict['men'] = 0.15
        ratioDict['kings'] = 0.20
        ratioDict['defense'] = 0.20
        ratioDict['offense'] = 0.30
	ratioDict['mobility'] = 0.05

    #normal mode weights
    if normalMode:
        ratioDict['block_promotion'] = 0.25
        ratioDict['men'] = 0.25
        ratioDict['kings'] = 0.25
        ratioDict['defense'] = 0.20
	ratioDict['offense'] = 0.03
	ratioDict['mobility'] = 0.02

    #defensive mode weights
    if defensiveMode:
        ratioDict['block_promotion'] = 0.22
        ratioDict['men'] = 0.24
        ratioDict['kings'] = 0.24
        ratioDict['defense'] = 0.27
        ratioDict['offense'] = 0.00
	ratioDict['mobility'] = 0.00

    return ratioDict['block_promotion']*scoreMap['promotion'] + ratioDict['men']*scoreMap['men'] + ratioDict['kings']*scoreMap['kings'] + ratioDict['defense']*scoreMap['defense'] + 0.02*scoreMap['unoccpromo'] + ratioDict['offense']*scoreMap['offense'] + ratioDict['mobility']*scoreMap['mobility']

'''
Funtion Name: unoccupiedFieldsOnPromotion
Description: Scores the board based on the number of pawns on the back/promotion rowof the opponent. Helps in maximizing the making of kings of opponents
Returns: score
'''

def unoccupiedFieldsOnPromotion(board, color):
	opponentColor = gamePlay.getOpponentColor(color)
	value = 0
	if color == 'r':
		oppRow = [29,30,31,32]
	else:
		oppRow = [1,2,3,4]
	for item in oppRow:
		xy = gamePlay.serialToGrid(item)
                x = xy[0]
                y = xy[1]
                if board[x][y] == opponentColor or board[x][y] == opponentColor.upper():
                        value -= 1
	scoreMap['unoccpromo'] = value
	 	


'''
Funtion Name: promotionRowItems
Description: Scores the board based on the number of pawns on the back/promotion row. Helps in delaying the making of kings of opponents
Returns: score
'''


def promotionRowItems(board, color):
	value = 0
	opponentColor = gamePlay.getOpponentColor(color)
	if color == 'r':
		myRow = [1,2,3,4]
		oppRow = [29,30,31,32]
	else:
		myRow = [29,30,31,32]
		oppRow = [1,2,3,4]

	for i in myRow:
		xy = gamePlay.serialToGrid(i)
		x = xy[0]
		y = xy[1]
		if board[x][y] == color or board[x][y] == color.upper():
			value += 1
	
	for i in oppRow:
		xy = gamePlay.serialToGrid(i)
                x = xy[0]
                y = xy[1]
                if board[x][y] == opponentColor or board[x][y] == opponentColor.upper():
                	value -= 1
	scoreMap['promotion'] = value


'''
Funtion Name: defendKings
Description: Scores each king on their defensiveness for both players and subtracts
Returns: score
'''

def defendKings(board, color):
	opponentColor = gamePlay.getOpponentColor(color)
	val1 = activeDefenseForKings(board, color)
	val2 = activeDefenseForKings(board, opponentColor)
	return val1-val2




'''
Funtion Name: activeDefenseForKings
Description: Scores each king, based on the color and it's defenders. Considers 4 surrounding corner while evaluating the score
Returns: score
'''

def activeDefenseForKings(board, color):
        opponentColor = gamePlay.getOpponentColor(color)
        val1 = 0
	val2 = 0
        forwardRight = (1, -1)
        forwardLeft = (1, 1)
        backwardRight = (-1, -1)
        backwardLeft = (-1, 1)
	postNeighbors = list()
        postNeighbors = [backwardRight, backwardLeft]

	for i in range(1,33):
                xy = gamePlay.serialToGrid(i)
                x = xy[0]
                y = xy[1]
                if board[x][y] == color:
			#for post neighbors of kings, see if there are same colored pawns to rate defensiveness
                        for item in postNeighbors:
                                coord = tuple(sum(x) for x in zip(xy, item))
                                prev_xy = gridToSerial(coord[0], coord[1])
                                if prev_xy in range (1, 33):
                                        new_coord = gamePlay.serialToGrid(prev_xy)
                                        newx = new_coord[0]
                                        newy = new_coord[1]
                                        if board[newx][newy] == color:
                                                val1 += 1


                elif board[x][y] == opponentColor.upper():
                        val2 += 1


	return val1*val2




'''
Funtion Name: activeDefenseForPawns
Description: Scores each piece, based on the color and it's defenders. Considers 4 surrounding corner while evaluating the score
Returns: score
'''


def activeDefenseForPawns(board, color):
	opponentColor = gamePlay.getOpponentColor(color)
	value = 0
	forwardRight = (1, -1)
        forwardLeft = (1, 1)
        backwardRight = (-1, -1)
        backwardLeft = (-1, 1)
	prevNeighbors = list()
	if color == 'r':
		prevNeighbors = [forwardRight, forwardLeft]
	else:
		prevNeighbors = [backwardRight, backwardLeft]

	for i in range(1,33):
		xy = gamePlay.serialToGrid(i)
                x = xy[0]
                y = xy[1]
		if board[x][y] == color:
   			#for post neighbors of pawns, see if there are same colored pawns to rate defensiveness
			for item in prevNeighbors:
				coord = tuple(sum(x) for x in zip(xy, item))
				prev_xy = gridToSerial(coord[0], coord[1])
				if prev_xy in range (1, 33):
                                        new_coord = gamePlay.serialToGrid(prev_xy)
                                        newx = new_coord[0]
                                        newy = new_coord[1]
                                        if board[newx][newy] == color:
                                                value += 1
                #subtract for opponent                   
		elif board[x][y] == opponentColor:
			for item in prevNeighbors:
				coord = tuple(sum(x) for x in zip(xy, item))
                                prev_xy = gridToSerial(coord[0], coord[1])

                                if prev_xy in range (1, 33):
                                	new_coord = gamePlay.serialToGrid(prev_xy)
                                        newx = new_coord[0]
                                        newy = new_coord[1]
                                        if board[newx][newy] == opponentColor:
                                        	value -= 1
	return value
 	

'''
Funtion Name: pawnsOnDiagonal
Description: Get score for pawns on two main diagonal which are in the danger of being captured
Returns: Score for pawns on diagonals
'''


def pawnsOnDiagonal(board, color):
    diagonal = [9,14,18,23]
    opponentColor = gamePlay.getOpponentColor(color)
    value = 0

    #loop through pieces on diagonals and find out pawns which are at the risk of being captured
    for item in diagonal:
        xy = gamePlay.serialToGrid(item)
        x = xy[0]
        y = xy[1]
        if board[x][y] == color or board[x][y] == color.upper():
                value += 1
        elif board[x][y] == opponentColor or board[x][y] == opponentColor.upper():
                value -= 1
    scoreMap['pawnsOnDiagonal'] = value

'''
Funtion Name: mobility
Description: Scores the board based on the number of moves(mobility) a pawn can make. Helps in avoiding positions with limited or no moves.
Returns: score
'''
def mobility(board, color):
	opponentColor = gamePlay.getOpponentColor(color)
	mobilityValue = 0
	for i in range(1,33):
		xy = gamePlay.serialToGrid(i)
	        x = xy[0]
        	y = xy[1]
		if board[x][y] == color.upper() or board[x][y] == color:
			moves, isCapturePossible = getAllPossibleMovesAtPosition(board, x, y)
			if len(moves) >= 1:
				mobilityValue += 1
		if board[x][y] == opponentColor.upper() or board[x][y] == opponentColor:
                        moves, isCapturePossible = getAllPossibleMovesAtPosition(board, x, y)
                        if len(moves) >= 1:
                                mobilityValue -= 1
	scoreMap['mobility'] = mobilityValue
    


'''
Funtion Name: offense
Description: Scores the board based on the how advanced the pieces are. Helps in making kings and offense.
Returns: score
'''
def offense(board, color):
    opposideEnd = None;
    opponentColor = gamePlay.getOpponentColor(color)
    distanceMeasure = 0.0
    count1 = 0
    count2 = 0
    if color == 'r':
   	for i in range(1, 33):
		xy = gamePlay.serialToGrid(i)
		x = xy[0]
		y = xy[1]
		if board[x][y] == color:
			count1 += x
		elif board[x][y] == opponentColor:
			count2 += (7-x)
    elif color == 'w':	
	for i in range(1,33):
		xy = gamePlay.serialToGrid(i)
                x = xy[0]
                y = xy[1]
		if board[x][y] == color:
			count1 += (7-x)
		elif board[x][y] == opponentColor:
                        count2 += x


    scoreMap['offense'] = count1 - count2


'''
Funtion Name: getColumn
Description: Returns a column of the board
Returns: column
'''


def getColumn(board, colNumber):
    return [row[colNumber] for row in board]


'''
Funtion Name: getCenterPieceBothPlayers
Descriptiion: Evaluates board on centrality of the pawns for both players. Helps in quickly reaching all board co ordinates.
Returns: score
'''

def getCenterPieceBothPlayers(board, color):
	opponentColor = gamePlay.getOpponentColor(color)
	valPlay = centerPieces(board, color)
	valOppo = centerPieces(board, opponentColor)
	scoreMap['center'] = valPlay-valOppo


'''
Funtion Name: centerPieces
Descriptiion: Evaluates board on centrality of the pawns for a player. Helps in quickly reaching all board co ordinates.
Returns: score
'''

def centerPieces(board, color):
	value = 0
	opponentColor = gamePlay.getOpponentColor(color)
	if color == 'w':
		centralLocations = [18,19]
		for item in centralLocations:
			xy = gamePlay.serialToGrid(item)
	                x = xy[0]
        	        y = xy[1]
                	if board[x][y] == 'w':
				value += 1
	else:
		centralLocations = [14,15]
                for item in centralLocations:
                        xy = gamePlay.serialToGrid(item)
                        x = xy[0]
                        y = xy[1]
                        if board[x][y] == 'r':
                                value += 1
	return value



'''
Funtion Name: defense
Descriptions: defense strategy for kings and pawns. Helps in calculation of heuristic value of move.
Returns: value for defensiveness of pieces
'''

def defense(board, color):
	val1 = defendKings(board, color)
	val2 = activeDefenseForPawns(board, color)
	val3 = safePawns(board, color)
	val4 = safeKings(board, color)
	scoreMap['defense'] = 0.30*val1 + 0.30*val2 + 0.20*val3 + 0.20*val4


'''
Funtion Name: safePawns
Descriptions: Number of safe men - men on the edges of the board. Helps in calculation of heuristic value of move.
Returns: number of safe pawns
'''
def safePawns(board, color):
    opponentColor = gamePlay.getOpponentColor(color)
    
    #get count of pawns on four edges for each player
    safePawnsOfPlayer = getColumn(board, 0).count(color) + getColumn(board, 7).count(color)
    return safePawnsOfPlayer


'''
Funtion Name: safeKings
Descriptions: Number of safe kings - kings on the corners of the board. Helps in calculation of heuristic value of move.
Returns: safe kings
'''
def safeKings(board, color):
    opponentColor = gamePlay.getOpponentColor(color)
    corner = [5, 1, 32, 28, 29, 4]
    value = 0
    for king in corner:
	xy = gamePlay.serialToGrid(king)
	x = xy[0]
	y = xy[1]
	if board[x][y] == color.upper():
		value += 1
	
    
    return value



'''
Funtion Name: getPawnsOnDogHoles
Descriptions: get pawn on slot 5 for red play and 28 for white play
Returns: 1 if present 
'''

def getPawnsOnDogHoles(board, color):
    temp = 0
    #get pawn in dog holes

    if color == 'w':
        if board[6][7].count(color):
		temp -= 1
    else:
	temp = board[1][0].count(color)
    scoreMap['dogHole'] = temp



'''
Funtion Name: cornerSquares
Descriptions: get count of pawns or kings on corner squares
Returns: pieces on corners
'''

def cornerSquares(board, color):
	drawPositions = [5, 1, 32, 28]
	value = 0
	opponentColor = gamePlay.getOpponentColor(color)
	for item in drawPositions:
		xy = gamePlay.serialToGrid(item)
		x = xy[0]
		y = xy[1]
		if board[x][y] == color or board[x][y] == color.upper():
			value += 1
		elif board[x][y] == opponentColor or board[x][y] == opponentColor.upper():
			value -= 1
	scoreMap['cornerSquare'] = value
				
	

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
    scoreMap['defenders'] = 10*defendersOfPlayer-10*defendersOfOpponent


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
Funtion Name: scoreMen
Description: Difference in men of both the players. Helps in calculation of heuristic value of move.
Returns: men diff
'''
def scoreMen(board, color):
    opponentColor = gamePlay.getOpponentColor(color)

    value = 0
    # Loop through all board positions
    for piece in range(1, 33):
        xy = gamePlay.serialToGrid(piece)
        x = xy[0]
        y = xy[1]

        if board[x][y] == color:
            value = value + 1
        elif board[x][y] == opponentColor:
            value = value - 1

    scoreMap['men'] = value

'''
Funtion Name: scoreKings
Description: Difference in kings of both the players. Helps in calculation of heuristic value of move.
Returns: kings diff
'''


def scoreKings(board, color):
    opponentColor = gamePlay.getOpponentColor(color)

    value = 0
    # Loop through all board positions
    for piece in range(1, 33):
        xy = gamePlay.serialToGrid(piece)
        x = xy[0]
        y = xy[1]

        if board[x][y] == color.upper():
            value = value + 2
        elif board[x][y] == opponentColor.upper():
            value = value - 2

    scoreMap['kings'] = value



#List of heurisitc functions
heuristicFunctions = [scoreKings, scoreMen, promotionRowItems, cornerSquares, getCenterPieceBothPlayers, pawnsOnDiagonal, defense, offense, unoccupiedFieldsOnPromotion, mobility]
    
'''
Funtion Name: iterativeDeepeningAlphaBetaPruning
Description: Best move given a board state based on alpha-beta iterative deepening method. Given a time limit, decides the depth limit of search and evaluation
Returns: Move
'''
def iterativeDeepeningAlphaBetaPruning(board, time, maxRemainingMoves):
    # Set depth limit depending the available time
    
   


    global myColor
    global opponentColor

    # Don't call mini-max, return the best move at the game start according to the player's color. 
    if maxRemainingMoves == 150:
	if myColor == 'r':
		return [11, 15]
	else:
		return [22, 18]

    moves = getAllPossibleMoves(board, myColor)
   
    #return the only move, if any
    if len(moves) == 1:
	return moves[0]
    depth = 4
    myPieces = gamePlay.countPieces(board, myColor)
    opponentPieces = gamePlay.countPieces(board, opponentColor)

    # piece ratio for deciding the depth
    pieceRatio = myPieces/opponentPieces
    if pieceRatio < 1:
	depth = 6

    if time < 30 and pieceRatio < 1: depth = 3
    elif time < 20 and pieceRatio > 1: depth = 2
    elif time < 10: depth = 1 
    bestMove = None
    best = -sys.maxint-1
    for move in moves:
        newBoard = deepcopy(board)
        gamePlay.doMove(newBoard,move)
        #Calling mini-max with alpha-beta pruning

        moveVal = alphaBetaPruning(newBoard, depth,time)
        if best == None or moveVal > best:
            bestMove = move
            best = moveVal
    return bestMove



'''
Funtion Name: alphaBetaPruning
Description: Score of the next best move based on a evaluation function
Returns: score
'''       

def alphaBetaPruning(board, depth, time):
    #Get a list of avaiable moves
	
	# Maximize the player's winning chance
        def maximize(board, alpha, beta, depth, time):
                global opponentColor
		global myColor
		#Return a heurisitc based score once the depth limit is reached
	
		if depth <= 0 or not gamePlay.isAnyMovePossible(board, opponentColor) or time < 7:
                        return evaluate(board, myColor)
                score = -sys.maxint-1
                for move in getAllPossibleMoves(board, opponentColor):
                        newBoard = deepcopy(board)
                        gamePlay.doMove(newBoard, move)
                        score = max(score, minimize(newBoard, alpha, beta, depth-1, time))
			
			#beta cut-off
                        if score >= beta:
                                return score
                        alpha = max(alpha, score)
                return score

        # Minimize the player's losing chance, by considering an opponent's move with the same heuristic s trategy
	def minimize(board, alpha, beta, depth, time):
		global opponentColor
		global myColor
		#Return a heurisitc based score once the depth limit is reached
		
                if depth <=0 or not gamePlay.isAnyMovePossible(board, opponentColor) or time < 7:
                        return evaluate(board, opponentColor)
                score = sys.maxint
                for move in getAllPossibleMoves(board, myColor):
                        newBoard = deepcopy(board)
                        gamePlay.doMove(newBoard, move)
                        score = min(score, maximize(newBoard, alpha, beta, depth-1, time))
			
			#alpha cut-off
                        if score <= alpha:
                                return score
                        beta = min(beta, score)
                return score


	#Start by maximizing the player's winning chance
	val = maximize(board, -sys.maxint-1, sys.maxint, depth, time)
        return val
		

myColor = None
opponentColor = None

'''
Funtion Name: nextMove
Description: Best move, given a state
Returns: move
'''
def nextMove(board, color, time, movesRemaining):
    global myColor
    global opponentColor
    myColor = color
    opponentColor = gamePlay.getOpponentColor(color)
    #Trying to find the move where I have best score
    return iterativeDeepeningAlphaBetaPruning(board, time, movesRemaining)
