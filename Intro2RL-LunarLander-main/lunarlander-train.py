import gymnasium as gym
from stable_baselines3 import PPO

env = gym.make('LunarLander-v2')

model = PPO('MlpPolicy', env, verbose=1)

model.learn(total_timesteps=1000000)

model.save("ppo_lunar_lander")

episodes=10
for episode in range(1, episodes+1):
    obs, info = env.reset()
    done = False
    total_reward = 0

    while not done:
        action, _states = model.predict(obs)
        obs, reward, done, info, truncated = env.step(action)
        total_reward += reward
    
    print(f"Episode: {episode+1}, Total Reward: {total_reward}")

env.close()