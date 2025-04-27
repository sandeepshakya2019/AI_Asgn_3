import gym
import slimevolleygym
from slimevolleygym.patch_env import patch_env
from slimevolleygym.slimevolley import SlimeVolleyEnv
from algo.mini_max import minimax_search  # Ensure the correct function name
from algo.alpha_beta import alpha_beta_search  # Ensure the correct function name
from algo.random_agent_v1 import RandomAgent

import numpy as np
import imageio
import cv2
import time

# Patch the environment
patch_env()

def render_frame_with_scores(env, yellow_score, blue_score, is_final=False):
    """
    Render the current frame with scores overlayed.
    If it's the final frame, display the match result (win/draw).
    """
    frame = env.render(mode="rgb_array")
    overlay = frame.copy()
    text_color = (255, 255, 255)

    # Display scores
    cv2.putText(overlay, f"Yellow: {yellow_score} | Blue: {blue_score}", (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2, cv2.LINE_AA)

    # Display match result if it's the final frame
    if is_final:
        if yellow_score > blue_score:
            result = "Yellow Wins!"
        elif blue_score > yellow_score:
            result = "Blue Wins!"
        else:
            result = "It's a Draw!"
        cv2.putText(overlay, result, (30, 80), cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, (0, 255, 0), 3, cv2.LINE_AA)

    return overlay

def simulate_game(algorithm="minimax"):
    """
    Simulate a full game using the specified algorithm against a random opponent.
    Save the rendered game as a video file and print match statistics.
    """
    # Initialize the environment and reset the state
    env = gym.make("SlimeVolley-v0")
    observation = env.reset()
    frames = []
    yellow_score = 0
    blue_score = 0

    # Game settings
    fps = 30  # Frames per second
    duration_seconds = 40 # Duration of the game in seconds
    total_frames = fps * duration_seconds

    # Opponent setup
    opponent = RandomAgent()
    frame_count = 0
    start_time = time.time()

    # Run the game for the specified duration
    for step in range(total_frames):
        # Render the current frame with scores
        frame = render_frame_with_scores(env, yellow_score, blue_score)
        frames.append(frame)

        # Get the agent's action based on the selected algorithm
        if algorithm == "minimax":
            _, agent_action = minimax_search(env, observation, depth=3, is_maximizing_player=True)
        elif algorithm == "alphabeta":
            _, agent_action = alpha_beta_search(env, observation, search_depth=3, alpha=float('-inf'), beta=float('inf'), is_maximizing_player=True)

        # Default to no-op if the action is None
        if agent_action is None:
            agent_action = np.array([0, 0, 0])

        # Get the random opponent's action
        opponent_action = opponent.act(observation)

        # Combine actions into a single joint action
        joint_action = np.hstack((agent_action, opponent_action))

        # Step through the environment
        observation, reward, done, _ = env.step(joint_action)
        frame_count += 1

        # Update scores based on the reward
        if reward == 1:
            yellow_score += 1
        elif reward == -1:
            blue_score += 1

        # Reset the environment if the episode ends
        if done:
            observation = env.reset()

    # Stop the timer and calculate execution time
    end_time = time.time()
    execution_time = end_time - start_time

    # Render the final result frame for 2 seconds
    final_frame = render_frame_with_scores(env, yellow_score, blue_score, is_final=True)
    for _ in range(fps * 2):
        frames.append(final_frame)

    # Close the environment
    env.close()

    # Resize frames to ensure compatibility with video encoding
    height, width, _ = frames[0].shape
    height = ((height + 15) // 16) * 16
    width = ((width + 15) // 16) * 16

    # Save the rendered frames as a video file
    output_path = f"{algorithm}_game.mp4"
    with imageio.get_writer(output_path, fps=fps) as writer:
        for frame in frames:
            resized_frame = cv2.resize(frame, (width, height))
            writer.append_data(resized_frame)

    # Print results
    print(f"✅ Saved full 30-second game to {output_path}")
    print(f"\n📊 Match Statistics ({algorithm.capitalize()}):")
    print(f"Total Score: Yellow {yellow_score} - Blue {blue_score}")
    print(f"Total Frames Played: {frame_count}")
    print(f"Execution Time: {execution_time:.2f} seconds")

def simulate_both_algorithms():
    """
    Simulate games using both Minimax and Alpha-Beta algorithms sequentially.
    """
    print("🎮 Running Minimax algorithm...")
    simulate_game(algorithm="minimax")

    print("\n🎮 Running Alpha-Beta algorithm...")
    simulate_game(algorithm="alphabeta")

if __name__ == "__main__":
    simulate_both_algorithms()