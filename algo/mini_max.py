import numpy as np

# Define all possible actions for the agent
POSSIBLE_ACTIONS = [
    np.array([0, 0, 0]),  # No action
    np.array([1, 0, 0]),  # Move left
    np.array([0, 1, 0]),  # Jump
    np.array([0, 0, 1]),  # Move right
    np.array([1, 1, 0]),  # Move left + jump
    np.array([0, 1, 1]),  # Move right + jump
    np.array([1, 0, 1]),  # Move left + move right (invalid but harmless)
]

def evaluate_state(state):
    """
    Evaluate the current state by calculating the distance between the ball and the player.
    Closer proximity to the ball results in a better score.
    """
    ball_x_position = state[4]
    player_x_position = state[0]
    return -abs(ball_x_position - player_x_position)  # Smaller distance is better


def minimax_search(environment, observation, depth, is_maximizing_player):
    """
    Perform a minimax search to determine the best action for the current player.
    """
    if depth == 0:
        return evaluate_state(observation), None

    optimal_value = float("-inf") if is_maximizing_player else float("inf")
    optimal_action = None

    for action in POSSIBLE_ACTIONS:
        saved_state = environment.clone_state()
        opponent_action = np.array([0, 0, 0])  # Assume opponent does nothing

        combined_action = np.hstack((action, opponent_action))
        next_observation, reward, done, info = environment.step(combined_action)

        value, _ = minimax_search(environment, next_observation, depth - 1, not is_maximizing_player)
        environment.restore_state(saved_state)

        if is_maximizing_player and value > optimal_value:
            optimal_value = value
            optimal_action = action
        elif not is_maximizing_player and value < optimal_value:
            optimal_value = value
            optimal_action = action

    return optimal_value, optimal_action