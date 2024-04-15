import chess
import chess.svg
import modelRunner
from chessboard import display

board = chess.Board()

def minimax(board, depth, isMaximizing, alpha, beta):
    if depth == 0 or board.is_game_over():
        return modelRunner.evalPos(board.fen())
    
    if isMaximizing:
        bestValue = -10000
        for move in board.legal_moves:
            board.push(move)
            value = minimax(board, depth - 1, False, alpha, beta)
            board.pop()
            bestValue = max(bestValue, value)
            alpha = max(alpha, value)
            if beta <= alpha:
                break
        return bestValue
    else:
        bestValue = 10000
        for move in board.legal_moves:
            board.push(move)
            value = minimax(board, depth - 1, True, alpha, beta)
            board.pop()
            bestValue = min(bestValue, value)
            beta = min(beta, value)
            if beta <= alpha:
                break
        return bestValue

def getAiMove(board, depth):
    bestValue = 10000  # Start with a very high value because Black is minimizing
    bestMove = None
    for move in board.legal_moves:
        board.push(move)
        value = minimax(board, depth - 1, True, -10000, 10000)  # Start as True to simulate minimization
        board.pop()
        if value < bestValue:  # Black wants to minimize the value
            bestValue = value
            bestMove = move
    print(f"Best move: {bestMove}, Best Value: {bestValue}")
    return bestMove


if __name__ == "__main__":
    game_board = display.start(board.fen())
    print(f"Legal moves: {board.legal_moves}")
    turn = "u"  # Start with the user and assume the user is White
    while not board.is_game_over():
        if turn == "u":
            userInput = input("Enter move: ")
            userMove = chess.Move.from_uci(userInput)
            while userMove not in board.legal_moves:
                print("Illegal move")
                userInput = input("Enter move: ")
                userMove = chess.Move.from_uci(userInput)
            board.push(userMove)
            turn = "a"  # Switch to AI's turn, AI is Black
        else:
            aiMove = getAiMove(board, 4)  # AI calculates its move as Black
            board.push(aiMove)
            turn = "u"  # Switch back to user's turn
        print(board.fen())
        display.update(board.fen(), game_board )
    input("Game Over")
