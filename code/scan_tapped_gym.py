# python3.9 scan_fake_gym.py | uniq -c


import gym
import traceback

import stable_baselines3 as sb3

from numpy import array,float32

class FakeEnv:
    def reset(self,*a,**ka):
        print('\nR',end='')
        self.flip = 0
        return array([0.02850769,0.04330681,0.04995482,0.03192485])
    def step(self,*a,**ka):
        self.flip = 1-self.flip
        b = not self.flip
        print(int(b),end='')
        return (
            array([ 0.09642447,  0.83463216, -0.22616503, -1.4688144 ],
            ), 1.0, b, {})

env = FakeEnv();
env.observation_space = gym.spaces.Box(array([-4.8000002e+00,-3.4028235e+38,-4.1887903e-01,-3.4028235e+38]),array([4.8000002e+00,3.4028235e+38,4.1887903e-01,3.4028235e+38]), (4,), float32)

env.action_space = gym.spaces.Discrete(2)
env.metadata={}

import os

if (os.system('test -s stable_baselines3')):
    print('needs sb3 repo here to scan the source code')
    exit(0)

import subprocess
all_policies = subprocess.check_output("grep -ho '[a-zA-Z]*Policy' -r stable_baselines3 --include \*.py | sort | uniq",shell=True).decode().split('\n')

for k in sb3.__dict__:
    for pol in all_policies:
        try:
            model = sb3.__dict__[k](pol, env)
            print(k,pol)
            #print(model,type(model.policy))
            model.learn(2)
            print("+++++++++++++++++++++++++++++++++++++++++++")
        except:
            #print('failed by',k,pol)
            pass
