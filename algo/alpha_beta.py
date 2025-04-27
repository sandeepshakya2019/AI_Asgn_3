import numpy as np

# Define possible actions for the agent
POSSIBLE_ACTIONS = [
    np.array([0, 0, 0]), np.array([1, 0, 0]),
    np.array([0, 1, 0]), np.array([0, 0, 1]),
    np.array([1, 1, 0]), np.array([0, 1, 1]),
    np.array([1, 0, 1])
]

def evaluate_state(state):
    """
    Evaluate the current state based on the ball's position relative to the player.
    """
    ball_position = state[4]
    player_position = state[0]
    return -abs(ball_position - player_position)

def alpha_beta_search(environment, observation, search_depth, alpha, beta, is_maximizing_player):
    """
    Perform an alpha-beta pruning search to determine the best action for the current player.
    """
    if search_depth == 0:
        return evaluate_state(observation), None

    optimal_value = float("-inf") if is_maximizing_player else float("inf")
    optimal_action = None

    for action in POSSIBLE_ACTIONS:
        saved_state = environment.clone_state()
        opponent_action = np.array([0, 0, 0])  # Assume opponent does nothing

        combined_action = np.hstack((action, opponent_action))
        next_observation, reward, done, info = environment.step(combined_action)

        value, _ = alpha_beta_search(environment, next_observation, search_depth - 1, alpha, beta, not is_maximizing_player)
        environment.restore_state(saved_state)

        if is_maximizing_player:
            if value > optimal_value:
                optimal_value = value
                optimal_action = action
            alpha = max(alpha, optimal_value)
        else:
            if value < optimal_value:
                optimal_value = value
                optimal_action = action
            beta = min(beta, optimal_value)

        if beta <= alpha:
            break

    return optimal_value, optimal_action


