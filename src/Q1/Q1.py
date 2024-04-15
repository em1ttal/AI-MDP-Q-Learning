import game
import numpy as np


def print_q_learning_matrix(q_learning_matrix):
    print("Q-Learning Matrix:")
    print("             Up          Down        Left       Right")
    for i in range(q_learning_matrix.shape[0] // q_learning_matrix.shape[1]):
        for j in range(q_learning_matrix.shape[1]):
            print(f"({i}, {j}): {q_learning_matrix[i * 4 + j]}")


def q_learning_algorithm(game, important_positions, actions, probabilities=[1.0, 0.0, 0.0], drunken_sailor=False):
    # States x Actions
    # Actions: Up, Down, Left, Right
    q_learning_matrix = np.zeros((12, 4))
    end_delta = 0.0001
    board = game.game_matrix
    start_pos = important_positions[0]
    end_pos = important_positions[1]
    block_pos = important_positions[2]

    alpha = 0.3
    gamma = 0.6
    epsilon = 0.3
    num_convergences = 0
    episode = 0
    while True:
        player_pos = start_pos
        total_reward = 0
        old_matrix = q_learning_matrix.copy()
        while player_pos != end_pos:
            # Get index of current state
            index = player_pos[0] * board.shape[1] + player_pos[1]
            # Choose action
            if np.random.rand() < epsilon:
                action = np.random.choice(actions)
            else:
                action = np.argmax(q_learning_matrix[index])
            # Perform action
            new_pos = game.move(player_pos, block_pos, action, drunken_sailor)
            # Update q-learning matrix
            q_learning_matrix[index, action] = (1 - alpha) * q_learning_matrix[index, action] + alpha * (
                        board[new_pos] + gamma * np.max(q_learning_matrix[new_pos[0] * board.shape[1] + new_pos[1]]))
            total_reward += board[new_pos]
            player_pos = new_pos
        # Check if q-learning matrix is converging
        if np.linalg.norm(q_learning_matrix - old_matrix) < end_delta:
            num_convergences += 1
            if num_convergences == 10:
                break

        print("Episode:", episode)
        print("Total reward:", total_reward)
        if episode % 25 == 0:
            print_q_learning_matrix(q_learning_matrix)

        episode += 1

    print("Converged after", episode, "episodes")
    print("Total reward:", total_reward)
    print_q_learning_matrix(q_learning_matrix)


size = (3, 4)
start_pos = (2, 0)
end_pos = (0, 3)
# Up, Down, Left, Right
actions = [0, 1, 2, 3]

# game = game.Game(size, start_pos, end_pos, 1)
game = game.Game(size, start_pos, end_pos, 2, [0.99, 0.05, 0.05], )
# q_learning_algorithm(game, [start_pos, end_pos, (1, 1)], actions)

# Drunken sailor
q_learning_algorithm(game, [start_pos, end_pos, (1, 1)], actions, drunken_sailor=True)
