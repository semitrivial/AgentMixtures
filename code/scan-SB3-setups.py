import gym
import stable_baselines3 as sb3

import os

if (os.system('test -s stable_baselines3')):
    print('needs sb3 repo here to scan the source code')
    exit(0)

#model = sb3.PPO("MlpPolicy", "CartPole-v1").learn(100)
#model.policy
#model = sb3.DQN("MlpPolicy", "CartPole-v1").learn(100)
#model = sb3.A2C("MlpPolicy", "CartPole-v1").learn(100)

#find all policies with easy access to probability by experiment
all_envs = [env_spec.id for env_spec in gym.envs.registry.all()]
#import os
import subprocess
all_policies = subprocess.check_output("grep -ho '[a-zA-Z]*Policy' -r stable_baselines3 --include \*.py | sort | uniq",shell=True).decode().split('\n')
viable = []
viable1 = []
dist_array_duck = {}
for k in sb3.__dict__:
    for pol in all_policies:
        for env in all_envs:
            try:
                #print('trying',k,pol,env)
                model = sb3.__dict__[k](pol, env).learn(99)
                print('model:',model)
                print('policy:',type(model.policy))
                print(model.policy.get_distribution,model.policy.obs_to_tensor)
                obs = gym.make(env).reset()
                dist = model.policy.get_distribution(model.policy.obs_to_tensor(obs)[0])
                print('dist:',dist);
                viable.append((k,pol,env))
                array = dist.distribution.probs.detach().cpu().numpy()
                print(array)
                viable1.append((k,pol,env))
                dist_array_duck.setdefault(k,{})
                dist_array_duck[k].setdefault(pol,{})
                dist_array_duck[k][pol][env] = array
            except:
                print('failed with',k)
                pass

import json
for k in ['viable','viable1','dist_array_duck']:
    with open(k+'.json','w') as f:
        json.dump(globals()[k],f)
