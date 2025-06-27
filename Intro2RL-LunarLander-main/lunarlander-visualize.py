import gymnasium as gym
from stable_baselines3 import PPO

env = gym.make("LunarLander-v2", render_mode="human")

model = PPO.load("ppo_lunar_lander")

obs, info = env.reset()
done = False

while not done:
    action, _states = model.predict(obs)
    obs, reward, done, info, truncated = env.step(action)
    env.render()

env.close()