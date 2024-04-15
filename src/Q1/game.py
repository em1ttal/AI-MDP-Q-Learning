import numpy as np

class Game:
    def __init__(self, size, start_pos, end_pos, board_config, probabilities):
        # Setting up the board
        self.game_matrix = np.ones(size)
        if board_config == 1:
            self.game_matrix *= -1
            self.game_matrix[1, 1] = -np.inf
            self.game_matrix[end_pos] = 100
        elif board_config == 2:
            self.game_matrix = np.array([[-3, -2 , -1, 100], [-4, -np.inf, -2, -1], [-5, -4, -3, -2]])

        # Start and goal for player
        self.start_pos = start_pos
        self.end_pos = end_pos

        # Probabilities for drunken sailor
        # [0] = Follow given direction, [1] = To the right of given direction, [2] = To the left of given direction
        self.probabilities = probabilities

    def move(self, player_pos, blocked_state, action, drunken_sailor):
        if not drunken_sailor:
            if action == 0:
                if player_pos[0] > 0 and player_pos != (blocked_state[0] + 1, blocked_state[1]):
                    new_pos = (player_pos[0] - 1, player_pos[1])
                else:
                    new_pos = (player_pos[0], player_pos[1])
            elif action == 1:
                if player_pos[0] < self.game_matrix.shape[0] - 1 and player_pos != (blocked_state[0] - 1, blocked_state[1]):
                    new_pos = (player_pos[0] + 1, player_pos[1])
                else:
                    new_pos = (player_pos[0], player_pos[1])
            elif action == 2:
                if player_pos[1] > 0 and player_pos != (blocked_state[0], blocked_state[1] + 1):
                    new_pos = (player_pos[0], player_pos[1] - 1)
                else:
                    new_pos = (player_pos[0], player_pos[1])
            else:
                if player_pos[1] < self.game_matrix.shape[1] - 1 and player_pos != (blocked_state[0], blocked_state[1] - 1):
                    new_pos = (player_pos[0], player_pos[1] + 1)
                else:
                    new_pos = (player_pos[0], player_pos[1])
        else:

            if action == 0:
                if np.random.rand() < self.probabilities[0]:
                    if player_pos[0] > 0 and player_pos != (blocked_state[0] + 1, blocked_state[1]):
                        new_pos = (player_pos[0] - 1, player_pos[1])
                    else:
                        new_pos = (player_pos[0], player_pos[1])
                else:
                    if np.random.rand() < (1/3):
                        if player_pos[1] < self.game_matrix.shape[1] - 1 and player_pos != (blocked_state[0], blocked_state[1] - 1):
                            new_pos = (player_pos[0], player_pos[1] + 1)
                        else:
                            new_pos = (player_pos[0], player_pos[1])
                    elif np.random.rand() < (2/3):
                        if player_pos[1] > 0 and player_pos != (blocked_state[0], blocked_state[1] + 1):
                            new_pos = (player_pos[0], player_pos[1] - 1)
                        else:
                            new_pos = (player_pos[0], player_pos[1])
                    else:
                        if player_pos[0] < self.game_matrix.shape[0] - 1 and player_pos != (blocked_state[0] - 1, blocked_state[1]):
                            new_pos = (player_pos[0] + 1, player_pos[1])
                        else:
                            new_pos = (player_pos[0], player_pos[1])

            elif action == 1:
                if np.random.rand() < self.probabilities[0]:
                    if player_pos[0] < self.game_matrix.shape[0] - 1 and player_pos != (blocked_state[0] - 1, blocked_state[1]):
                        new_pos = (player_pos[0] + 1, player_pos[1])
                    else:
                        new_pos = (player_pos[0], player_pos[1])
                else:
                    if np.random.rand() < (1/3):
                        if player_pos[1] > 0 and player_pos != (blocked_state[0], blocked_state[1] + 1):
                            new_pos = (player_pos[0], player_pos[1] - 1)
                        else:
                            new_pos = (player_pos[0], player_pos[1])
                    elif np.random.rand() < (2/3):
                        if player_pos[1] < self.game_matrix.shape[1] - 1 and player_pos != (blocked_state[0], blocked_state[1] - 1):
                            new_pos = (player_pos[0], player_pos[1] + 1)
                        else:
                            new_pos = (player_pos[0], player_pos[1])
                    else:
                        if player_pos[0] > 0 and player_pos != (blocked_state[0] + 1, blocked_state[1]):
                            new_pos = (player_pos[0] - 1, player_pos[1])
                        else:
                            new_pos = (player_pos[0], player_pos[1])

            elif action == 2:
                if np.random.rand() < self.probabilities[0]:
                    if player_pos[1] > 0 and player_pos != (blocked_state[0], blocked_state[1] + 1):
                        new_pos = (player_pos[0], player_pos[1] - 1)
                    else:
                        new_pos = (player_pos[0], player_pos[1])
                else:
                    if np.random.rand() < (1/3):
                        if player_pos[0] > 0 and player_pos != (blocked_state[0] + 1, blocked_state[1]):
                            new_pos = (player_pos[0] - 1, player_pos[1])
                        else:
                            new_pos = (player_pos[0], player_pos[1])
                    elif np.random.rand() < (2/3):
                        if player_pos[0] < self.game_matrix.shape[0] - 1 and player_pos != (blocked_state[0] - 1, blocked_state[1]):
                            new_pos = (player_pos[0] + 1, player_pos[1])
                        else:
                            new_pos = (player_pos[0], player_pos[1])
                    else:
                        if player_pos[1] < self.game_matrix.shape[1] - 1 and player_pos != (blocked_state[0], blocked_state[1] - 1):
                            new_pos = (player_pos[0], player_pos[1] + 1)
                        else:
                            new_pos = (player_pos[0], player_pos[1])

            else:
                if np.random.rand() < self.probabilities[0]:
                    if player_pos[1] < self.game_matrix.shape[1] - 1 and player_pos != (blocked_state[0], blocked_state[1] - 1):
                        new_pos = (player_pos[0], player_pos[1] + 1)
                    else:
                        new_pos = (player_pos[0], player_pos[1])
                else:
                    if np.random.rand() < (1/3):
                        if player_pos[0] < self.game_matrix.shape[0] - 1 and player_pos != (blocked_state[0] - 1, blocked_state[1]):
                            new_pos = (player_pos[0] + 1, player_pos[1])
                        else:
                            new_pos = (player_pos[0], player_pos[1])
                    elif np.random.rand() < (2/3):
                        if player_pos[0] > 0 and player_pos != (blocked_state[0] + 1, blocked_state[1]):
                            new_pos = (player_pos[0] - 1, player_pos[1])
                        else:
                            new_pos = (player_pos[0], player_pos[1])
                    else:
                        if player_pos[1] > 0 and player_pos != (blocked_state[0], blocked_state[1] + 1):
                            new_pos = (player_pos[0], player_pos[1] - 1)
                        else:
                            new_pos = (player_pos[0], player_pos[1])

        return new_pos

