import gym
import traceback

import stable_baselines3 as sb3

from numpy import array,float32

class FakeEnv:
    def reset(*a,**ka):
        print('RESET')
        return array([0.02850769,0.04330681,0.04995482,0.03192485])
    def step(*a,**ka):
        print('step')
        return (
            array([ 0.09642447,  0.83463216, -0.22616503, -1.4688144 ],
            ), 1.0, False, {})

env = FakeEnv();
env.observation_space = gym.spaces.Box(array([-4.8000002e+00,-3.4028235e+38,-4.1887903e-01,-3.4028235e+38]),array([4.8000002e+00,3.4028235e+38,4.1887903e-01,3.4028235e+38]), (4,), float32)

env.action_space = gym.spaces.Discrete(2)
env.metadata={}

model = sb3.PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=1)
print(111111111111111111111111111111111111111111111111)
model.learn(total_timesteps=1)
print(222222222222222222222222222222222222222222222222)
model.learn(total_timesteps=1)
print(333333333333333333333333333333333333333333333333)
model.learn(total_timesteps=1)
