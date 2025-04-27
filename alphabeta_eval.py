import gym
import time
import slimevolleygym
from slimevolleygym.patch_env import patch_env
from slimevolleygym.slimevolley import SlimeVolleyEnv
from algo.random_agent_v1 import RandomAgent

import numpy as np
import imageio
import cv2

# Patch the environment
patch_env()

# Define possible actions for the agent
ACTION_SPACE = [
    np.array([0, 0, 0]),  # do nothing
    np.array([1, 0, 0]),  # move left
    np.array([0, 1, 0]),  # jump
    np.array([0, 0, 1]),  # move right
    np.array([1, 1, 0]),  # move left + jump
    np.array([0, 1, 1]),  # move right + jump
    np.array([1, 0, 1])   # move left + move right (invalid but harmless)
]

def evaluation_function(obs):
    """
    Evaluate the current state based on the ball's position relative to the player.
    Closer proximity to the ball results in a better score.
    """
    ball_x = obs[4]
    player_x = obs[0]
    return -abs(ball_x - player_x)

def alphabeta(env, obs, depth, alpha, beta, maximizing_player):
    """
    Perform an alpha-beta pruning search to determine the best action.
    """
    if depth == 0:
        return evaluation_function(obs), None

    optimal_value = float("-inf") if maximizing_player else float("inf")
    optimal_action = None

    for action in ACTION_SPACE:
        saved_state = env.clone_state()
        opponent_action = np.array([0, 0, 0])  # Assume opponent does nothing

        combined_action = np.hstack((action, opponent_action))
        next_obs, reward, done, info = env.step(combined_action)

        value, _ = alphabeta(env, next_obs, depth - 1, alpha, beta, not maximizing_player)
        env.restore_state(saved_state)

        if maximizing_player:
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

def render_frame_with_score(env, score_left, score_right, final=False):
    """
    Render the current frame with scores overlayed.
    """
    frame = env.render(mode="rgb_array")
    overlay = frame.copy()
    text_color = (255, 255, 255)

    cv2.putText(overlay, f"Yellow: {score_left} | Blue: {score_right}", (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2, cv2.LINE_AA)

    if final:
        if score_left > score_right:
            winner = "Yellow Wins!"
        elif score_right > score_left:
            winner = "Blue Wins!"
        else:
            winner = "It's a Draw!"
        cv2.putText(overlay, winner, (30, 80), cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, (0, 255, 0), 3, cv2.LINE_AA)

    return overlay

def run_alphabeta_game():
    """
    Run a full game using Alpha-Beta pruning against a random opponent.
    Save the rendered game as a video file.
    """
    env = gym.make("SlimeVolley-v0")
    obs = env.reset()
    frames = []
    score_left = 0
    score_right = 0

    fps = 30
    duration_sec = 15
    total_frames = fps * duration_sec

    opponent = RandomAgent()  # Instantiate the RandomAgent

    start_time = time.time()

    for step in range(total_frames):
        frame = render_frame_with_score(env, score_left, score_right)
        frames.append(frame)

        _, action_left = alphabeta(env, obs, depth=3, alpha=float('-inf'), beta=float('inf'), maximizing_player=True)

        if action_left is None:
            action_left = np.array([0, 0, 0])

        # Get the opponent's action
        action_right = opponent.act(obs)

        # Combine the two actions into a single array
        joint_action = np.hstack((action_left, action_right))

        # Pass the combined action to the environment
        obs, reward, done, _ = env.step(joint_action)

        if reward == 1:
            score_left += 1
        elif reward == -1:
            score_right += 1

        if done:
            obs = env.reset()

    final_frame = render_frame_with_score(env, score_left, score_right, final=True)
    for _ in range(fps * 2):
        frames.append(final_frame)

    env.close()

    height, width, _ = frames[0].shape
    height = ((height + 15) // 16) * 16
    width = ((width + 15) // 16) * 16
    resized_frames = [cv2.resize(f, (width, height)) for f in frames]

    out_path = "alphabeta_game.mp4"
    imageio.mimsave(out_path, resized_frames, fps=fps)
    
    elapsed_time = time.time() - start_time
    print(f"✅ Saved 15-second AlphaBeta game to {out_path}")
    print(f"Total frames played: {len(frames)}")
    print(f"Elapsed time: {elapsed_time:.2f}s")
    print(f"Final Score: Yellow={score_left}, Blue={score_right}")



    
if __name__ == "__main__":
    run_alphabeta_game()