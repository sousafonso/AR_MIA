from __future__ import annotations

import numpy as np

from mia_rl.envs.tictactoe import TicTacToeAction, TicTacToeEnv, TicTacToeState

# Each of the 9 cells is encoded as a 3-dim one-hot vector (from the current
# player's perspective): [my piece, opponent's piece, empty].
# Total feature dimension: 9 × 3 = 27.
STATE_FEATURE_DIM: int = 27


def encode_state(board: TicTacToeState, current_player: int) -> np.ndarray:
    """Encode a board as a 27-dim one-hot vector from `current_player`'s perspective.

    For each cell the encoding is:
        [1, 0, 0]  if the cell contains current_player's mark
        [0, 1, 0]  if the cell contains the opponent's mark
        [0, 0, 1]  if the cell is empty

    Using a perspective-relative encoding means the same policy weights work
    regardless of whether the agent is playing as X (+1) or O (-1).

    Args:
        board: length-9 tuple of ints (0 = empty, 1 = X, -1 = O).
        current_player: +1 or -1, identifies whose turn it is.

    Returns:
        np.ndarray of shape (27,), dtype float32.
    """
    # Inicializa o vetor de características com 27 dimensões (9 células x 3 slots por célula).
    phi = np.zeros(STATE_FEATURE_DIM, dtype=np.float32)
    
    # Mapeia cada célula com base na perspetiva do jogador atual.
    # Ao codificar os estados de forma relativa, permitimos que ambos os lados (X e O)
    # partilhem o mesmo conjunto de pesos da política (self-play), duplicando a eficiência de dados.
    for i, cell in enumerate(board):
        if cell == current_player:
            # A minha peça: guardada no primeiro slot da célula i (i * 3 + 0)
            phi[i * 3 + 0] = 1.0   # my piece
        elif cell == -current_player:
            # A peça do oponente: guardada no segundo slot da célula i (i * 3 + 1)
            phi[i * 3 + 1] = 1.0   # opponent's piece
        else:
            # Célula vazia: guardada no terceiro slot da célula i (i * 3 + 2)
            phi[i * 3 + 2] = 1.0   # empty
    return phi
