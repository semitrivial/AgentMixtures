import gym
import traceback

env = gym.make("CartPole-v1")
#dic = env.env.__dict__
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
      print(e.action_space);
      print(e.observation_space);
      print(e.metadata);
      print('RRRR', r)
      return r
   e.step = f1
   del f1
   #del f
   fR = e.reset
   def f2(*args, **kwargs):
      print('+++++++++++++++++++++++++++++++++++++++++++++++++++++')
      print(i,fR,args,kwargs)
      for line in traceback.format_stack():
         print(line.strip())
      print('+++++++++++++++++++++++++++++++++++++++++++++++++++++')
      r = fR(*args,**kwargs)
      print('RRRR', r)
      return r
   e.reset = f2
   del f2
   #del f
   if 'env' in e.__dict__:
      e = e.env
   else:
      break

import stable_baselines3 as sb3

model = sb3.PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=1)
print(111111111111111111111111111111111111111111111111)
model.learn(total_timesteps=1)
print(222222222222222222222222222222222222222222222222)
model.learn(total_timesteps=1)
print(333333333333333333333333333333333333333333333333)
model.learn(total_timesteps=1)
