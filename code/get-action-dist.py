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














import stable_baselines3 as sb3

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
                print(dist.distribution.probs.detach().cpu().numpy())
                viable1.append((k,pol,env))
            except:
                #print('failed with',k)
                pass

import json
json.dump(viable,'viable-triples.json')
json.dump(viable1,'viable1-triples.json')

for env in all_envs:
    try:
        print(env, sb3.TD3('MlpPolicy', env))
    except:
        pass



model = sb3.SAC('MlpPolicy', 'Pendulum-v1').learn(20)
model.policy.actor.action_dist
model.policy.actor.action_dist.distribution
