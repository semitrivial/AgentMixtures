import gym
import stable_baselines3 as sb3

import os

if (os.system('test -s stable_baselines3')):
    print('needs sb3 repo here to scan the source code')
    exit(0)

all_envs = [env_spec.id for env_spec in gym.envs.registry.all()]
import subprocess
all_policies = subprocess.check_output("grep -ho '[a-zA-Z]*Policy' -r stable_baselines3 --include \*.py | sort | uniq",shell=True).decode().split('\n')
for k in sb3.__dict__:
    for pol in all_policies:
        for env in all_envs:
            try:
                #print('trying',k,pol,env)
                model = sb3.__dict__[k](pol, env).learn(99)
            except:
                print('failed with',k)
                pass

