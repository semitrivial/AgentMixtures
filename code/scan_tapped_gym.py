# python3.9 scan_fake_gym.py | uniq -c


import gym
import traceback

import stable_baselines3 as sb3

from numpy import array,float32

import os

if (os.system('test -s stable_baselines3')):
    print('needs sb3 repo here to scan the source code')
    exit(0)

all_envs = [env_spec.id for env_spec in gym.envs.registry.all()]
import subprocess
all_policies = subprocess.check_output("grep -ho '[a-zA-Z]*Policy' -r stable_baselines3 --include \*.py | sort | uniq",shell=True).decode().split('\n')

step_end = '\n' 

import inspect
for k in sb3.__dict__:
    if k[:2]!='__' and not inspect.ismodule(sb3.__dict__[k]):
        for pol in all_policies:
            for env0 in all_envs:
                try:
                    print('\nSTART',k,pol,env0)
                    env = gym.make(env0)
                    e = env
                    for i in range(99999):
                       f = e.step
                       def f1(*args, **kwargs):
                          #print('   ===================')
                          #print('   ',i,f,args,kwargs)
                          #for line in traceback.format_stack():
                          #   print('   ',line.strip())
                          #print('   ===================')
                          r = f(*args,**kwargs)
                          print(int(r[-2]),end=step_end)
                          return r
                       e.step = f1
                       del f1
                       #del f
                       fR = e.reset
                       def f2(*args, **kwargs):
                          #print('+++++++++++++++++++++')
                          #print(i,fR,args,kwargs)
                          #for line in traceback.format_stack():
                          #   print(line.strip())
                          #print('+++++++++++++++++++++')
                          r = fR(*args,**kwargs)
                          #print('RRRR', r)
                          print('R',end=step_end)
                          return r
                       e.reset = f2
                       del f2
                       #del f
                       if 'env' in e.__dict__:
                          e = e.env
                       else:
                          break

                    model = sb3.__dict__[k](pol, env)
                    print(k,pol)
                    #print(model,type(model.policy))
                    model.learn(2)
                    print("#############################################")
                except Exception as e:
                    print('FAIL',k,pol,env0,e)
                    pass
