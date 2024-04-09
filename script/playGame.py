import chess
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