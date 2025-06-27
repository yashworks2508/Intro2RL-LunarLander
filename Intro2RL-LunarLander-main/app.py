import gradio as gr
import gymnasium as gym
from stable_baselines3 import PPO
import imageio
import numpy as np

# Load trained model
model = PPO.load("ppo_lunar_lander.zip")

# Function to simulate an episode and return as video
def simulate_lunar_lander():
    env = gym.make("LunarLander-v2", render_mode="rgb_array")
    obs, _ = env.reset()
    done = False
    frames = []

    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, _, _ = env.step(action)
        frame = env.render()
        frames.append(frame)

    env.close()

    # Save video
    video_path = "lander.mp4"
    imageio.mimsave(video_path, frames, fps=30)
    return video_path

demo = gr.Interface(fn=simulate_lunar_lander, inputs=[], outputs=gr.Video(label="Lunar Lander Simulation"))

if __name__ == "__main__":
    demo.launch()
