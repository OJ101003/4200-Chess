import torch

# Assuming MATERIAL_LOOKUP and fen_to_binary_piece_positions are defined
MATERIAL_LOOKUP = {
    'r': '0001', 'n': '0010', 'b': '0011', 'q': '0100', 'k': '0101', 'p': '0110',
    'R': '0111', 'N': '1000', 'B': '1001', 'Q': '1010', 'K': '1011', 'P': '1100',
    '.': '0000'  # Using '.' to represent empty squares
}

def fen_to_binary_piece_positions(fen):
    piece_positions = fen.split()[0]
    binary_piece_positions = ''
    for char in piece_positions:
        if char.isdigit():
            binary_piece_positions += MATERIAL_LOOKUP['.'] * int(char)
        elif char in MATERIAL_LOOKUP:
            binary_piece_positions += MATERIAL_LOOKUP[char]
        else:  # Ignore slashes
            continue
    return binary_piece_positions

def fen_to_binary(fen):
    # Convert piece positions to binary
    binary_piece_positions = fen_to_binary_piece_positions(fen)
    
    # Extract additional game state information from FEN
    parts = fen.split()
    active_color = '0' if parts[1] == 'w' else '1'
    castling = parts[2]
    castling_binary = ''.join(['1' if char in castling else '0' for char in 'KQkq'])
    
    # Encode en passant target square
    en_passant = parts[3]
    if en_passant == '-':
        en_passant_binary = '00000000'
    else:
        en_passant_binary = '0' * (ord(en_passant[0]) - ord('a')) + '1' + '0' * (7 - (ord(en_passant[0]) - ord('a')))
    
    # Encode halfmove and fullmove numbers
    halfmove_clock = format(int(parts[4]), '08b')
    fullmove_number = format(int(parts[5]), '08b')
    
    # Combine all parts into a single binary string
    binary_representation = binary_piece_positions + active_color + castling_binary + en_passant_binary + halfmove_clock + fullmove_number
    
    # Convert binary string to a tensor
    binary_tensor = torch.tensor([int(bit) for bit in binary_representation], dtype=torch.float32)
    
    return binary_tensor
