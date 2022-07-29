import gym

from stable_baselines3 import PPO

env = gym.make("CartPole-v1")

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=99)

obs = env.reset()

action, _states = model.predict(obs, deterministic=True)

#The statements above are straight from the official demo

print('obs',obs)
print('action',action)
#stable baseline 3 internally use PyTorch rather than NumPy
#everything is transformed to tensors
#there is also an extra dimension which seems to allow for running multiple experiments in parallel (not tried)
print('obs tensor',model.policy.obs_to_tensor(obs))
dist = model.policy.get_distribution(model.policy.obs_to_tensor(obs)[0])
print('action distribution',dist)
print('action distribution',dist.distribution)
print('action distribution',dist.distribution.probs)
print('action distribution',dist.distribution.probs.detach().cpu().numpy())
