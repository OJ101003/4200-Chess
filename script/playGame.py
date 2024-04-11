import chess
import chess.svg
import modelRunner

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
    bestValue = -10000
    bestMove = None
    for move in board.legal_moves:
        board.push(move)
        value = minimax(board, depth - 1, False, -10000, 10000)
        board.pop()
        if value > bestValue:
            bestValue = value
            bestMove = move
    return bestMove

SVG_BASE_URL = "https://us-central1-spearsx.cloudfunctions.net/chesspic-fen-image/" 

def svg_url(fen):
  fen_board = fen.split()[0]
  return SVG_BASE_URL + fen_board

if __name__ == "__main__":
    print(board)
    print(f"Legal moves: {board.legal_moves}")
    turn = "u"
    while not board.is_game_over():
        if turn == "u":
            userInput = input("Enter move: ")
            userMove = chess.Move.from_uci(userInput)
            while userMove not in board.legal_moves:
                print("Illegal move")
                userInput = input("Enter move: ")
                userMove = chess.Move.from_uci(userInput)
            board.push(userMove)
            turn = "a"
        else:
            aiMove = getAiMove(board, 4)
            print(f"AI move: {aiMove}")
            turn = "u"
        print(board)
        print(f"Legal moves: {board.legal_moves}")
