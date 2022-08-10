nix-shell --run 'python3.9 -c "from stable_baselines3 import A2C; A2C(\"MlpPolicy\", \"CartPole-v1\").learn(total_timesteps=4)"'



import gym
env = gym.make("CartPole-v1")
#dic = env.env.__dict__
e = env
while 1:
   f = e.step
   function f1(*args, **kwargs):
      return f(*args,**kwargs)
   e.step = f1
   del f1
