#Name: Suprith Chandrashekharachar
#Username: suprchan

import random
import sys

#Function: print_board
#Description: Prints a tic-tac-toe board.
#Input : None
#Output: Prints a tic-tac-toe board.
def print_board():
    for i in range(0,3):
        for j in range(0,3):
            print map[i][j],
            if j != 2:
                print "|",
        print ""


#Function: createMoves
#Description: Generates a list of possible moves, given a state of game board.
#Input : None
#Output: List of moves
		
def createMoves():
    listOfNextMoves = []
      
    if check_done(False):
        return listOfNextMoves

    for i in range(0, 3):
        for j in range(0, 3):
            if map[i][j] == " ":
                listOfNextMoves.append((i,j))
    return listOfNextMoves

#Function: calculateScore
#Description: Calculates a score in range of -100 to 100. Used by Alpha Beta pruning algorithm to determine the best move
#Input : a line(winning pattern) - includes 3 rows, 3 columns, and 2 diagonals
#Output: Score for a winning pattern
	
	
def calculateScore(row1, col1, row2, col2, row3, col3):
    score =0
    if map[row1][col1] == machineRole:
        score = 1
    elif map[row1][col1] == userRole:
        score = -1

    if map[row2][col2] == machineRole:
        if score == 1:
            score = 10
        elif score == -1:
            return 0
        else:
            score = 1
    elif map[row2][col2] == userRole:
        if score == -1:
            score = -10
        elif score == 1:
            return 0
        else:
            score = -1
    if map[row3][col3] == machineRole:
        if score > 0:
            score *= 10
        elif score < 0:
            return 0
        else:
            score = 1
    elif map[row3][col3] == userRole:
        if score < 0:
            score *= 10
        elif score > 1:
            return 0
        else:
            score = -1
    return score

	

#Function: evaluate
#Description: given a state of board, returns a score for each possible winning pattern.	
#Input : list of lines forming winning patterns
#Output: score
	
def evaluate(listOfLines):
    score = 0
    for row1, col1, row2, col2, row3, col3 in listOfLines:
        score += calculateScore(row1, col1, row2, col2, row3, col3)
    return score

#Function: alphaBetaPruning
#Description: given a state of board, return the best move co-ordinates and the corresponding score based on alpha beta pruning.
#Input : depth, the player(machine/user), alpha and beta values 
#Output: score and move co-ordinates
	
def alphaBetaPruning(depth, player, alpha,beta):
    listOfNextMoves = createMoves()
    listOfLines = [(0,0,0,1,0,2),(1,0,1,1,1,2),(2,0,2,1,2,2),(0,0,1,0,2,0),(0,1,1,1,2,1),(0,2,1,2,2,2),(0,0,1,1,2,2),(0,2,1,1,2,0)]
    score = 0
    bestRow = -1
    bestCol = -1
    
    if len(listOfNextMoves) == 0 or depth == 0:
        score = evaluate(listOfLines)
        return [score, bestRow, bestCol]
    else:
        for row, col in listOfNextMoves:
            
            map[row][col] = player
            if(player == machineRole):
                score = alphaBetaPruning(depth-1, userRole, alpha, beta)[0]
                if score > alpha:
                    alpha = score
                    bestRow = row
                    bestCol = col
            else:
                score = alphaBetaPruning(depth-1, machineRole, alpha, beta)[0]
                if score < beta:
                    beta = score
                    bestRow = row
                    bestCol = col
                
            map[row][col] = " "
            if alpha >= beta:
                break
        
        return [alpha if player == machineRole else beta, bestRow, bestCol]


#Function: getNextBestMove
#Description: given a state of board, returns next move for machine.	
#Input : player, machine in this case
#Output: move co-ordinates      


def getNextBestMove(player):
   
    nextBestMoveWithScore = alphaBetaPruning(2, player, -sys.maxint-1, sys.maxint)
    
    return [nextBestMoveWithScore[1], nextBestMoveWithScore[2]]

#Function: check_done
#Description: given a state of board, checks whether the game is over.	
#Input : player
#Output: Win/Lose/Draw   

def check_done(flag, player = "AlphaBeta"):
    for i in range(0,3):
        if map[i][0] == map[i][1] == map[i][2] != " " \
        or map[0][i] == map[1][i] == map[2][i] != " " and flag:
            if player == "User" or player == "Machine":
                print player, "won!!!"
            return True
        
    if map[0][0] == map[1][1] == map[2][2] != " " \
    or map[0][2] == map[1][1] == map[2][0] != " " and flag:
        if player == "User" or player == "Machine":
            print player, "won!!!"
        return True

    if " " not in map[0] and " " not in map[1] and " " not in map[2]:
        if player == "User" or player == "Machine":
            print "Draw"
        return True
        

    return False
    



map = [[" "," "," "],
       [" "," "," "],
       [" "," "," "]]
done = False
machineRole = "X"
machinePlaysFirst = True
userRole = raw_input('Please select your role - either X or O: ')
print



if userRole == "X":
    machineRole = "O"
    machinePlaysFirst = False
    
print "User plays",userRole
print "Machine will play", machineRole
print

print "Machine will play first" if  machinePlaysFirst else "User has to play first"
print
while done != True:
    
#user will play first in this block of code

    moved = False
    if not machinePlaysFirst:
        while moved != True:
            print "Please select position by typing in a number between 1 and 9, see below for which number that is which position..."
            print "1|2|3"
            print "4|5|6"
            print "7|8|9"
            print
            try:
                pos = input("Select: ")
                if pos <=9 and pos >=1:
                    X = pos/3
                    Y = pos%3

                    if Y != 0:
                        Y -=1
                    else:
                        Y = 2
                        X -=1
                        
                    if map[X][Y] == " ":
                        map[X][Y] = "X"
                       
                        done = check_done(True, "User")
                        if done:
                            break
                   
                    try:
                        
                        tup = getNextBestMove(machineRole)
                
                        if tup:
                            X = tup[0]
                            Y = tup[1]
                            print "Machine chooses...", X, Y
                            
                            if map[X][Y] == " ":
                                map[X][Y] = "O"
                                moved = True
                                done = check_done(True, "Machine")
                                print_board()
                                if done:
                                    break
                    except:
                        print "Exception in AlphaBeta"
                    print_board()
                        
            except:
                print "You need to add a numeric value"

#machine will play first in this block of code
    else:
        print "Machine will soon make a choice and game board will be presented"
        while moved != True:
            try:
                tup = getNextBestMove(machineRole)               
                if tup:
                    X = tup[0]
                    Y = tup[1]
                    print "Machine chooses...", X, Y
                            
                    if map[X][Y] == " ":
                        map[X][Y] = "X"
                        moved = True
                        done = check_done(True, "Machine")
                        print_board()
                        if done:
                            break
                        
            except:
                print "Exception in AlphaBeta"
            print_board()
            print "Please select position by typing in a number between 1 and 9, see below for which number that is which position..."
            print "1|2|3"
            print "4|5|6"
            print "7|8|9"
            print
            try:
                pos = input("Select: ")
                if pos <=9 and pos >=1:
                    X = pos/3
                    Y = pos%3

                    if Y != 0:
                        Y -=1
                    else:
                        Y = 2
                        X -=1
                   
                    if map[X][Y] == " ":
                        map[X][Y] = "O"
                        done = check_done(True, "User")
                        if done:
                            break
            except:
                print "You need to add a numeric value"
