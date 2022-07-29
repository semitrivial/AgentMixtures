import random 
random.seed(55)

from random import random, shuffle
import gym

action_space = [0,1]

def CoinFlip(A1,A2):
    class Mixture:
        def __init__(self):
            #print('CoinFlip init')
            self.worker1 = A1()
            self.worker2 = A2()
            self.a1_prob = 1
            self.a2_prob = 1

        def act(self, obs):
            if self.a1_prob + self.a2_prob == 0:
                assert(false)
                return 1.0/len(action_space)

            denominator = self.a1_prob + self.a2_prob

            a1_distro = self.worker1.act(obs)
            a2_distro = self.worker2.act(obs)

            distro = {}
            for action in action_space:
                summand1 = self.a1_prob*a1_distro[action]
                summand2 = self.a2_prob*a2_distro[action]
                numerator = summand1+summand2
                distro[action] = float(numerator)/float(denominator)

            return distro

        def train(self, o_prev, action, reward, o_next, done):
            a1_distro = self.worker1.act(o_prev)
            a2_distro = self.worker2.act(o_prev)
            self.a1_prob *= a1_distro[action]
            self.a2_prob *= a2_distro[action]
            self.worker1.train(o_prev, action, reward, o_next, done)
            self.worker2.train(o_prev, action, reward, o_next, done)

    return Mixture


class Q_learner:
  def __init__(self, epsilon=0.1, learning_rate=0.1, gamma=0.9):
    self.epsilon = epsilon
    self.learning_rate = learning_rate
    self.gamma = gamma
    self.qtable = {}

  def maybe_add_obs_to_qtable(self, obs):
    if not((obs, 0) in self.qtable):
      self.qtable.update({(obs, a): 0 for a in action_space})

  def act(self, obs):
    qtable, epsilon = self.qtable, self.epsilon
    self.maybe_add_obs_to_qtable(obs)

    best_action = max(action_space, key=lambda a: qtable[obs,a])
    distro = {a: epsilon/(len(action_space)-1) for a in action_space}
    distro[best_action] = 1-epsilon
    return distro

  def train(self, o_prev, a, r, o_next, done):
    qtable, gamma = self.qtable, self.gamma
    self.maybe_add_obs_to_qtable(o_prev)
    self.maybe_add_obs_to_qtable(o_next)

    if done:
        qtarget = r
    else:
        qtarget = r + gamma * max([qtable[o_next,b] for b in action_space])

    qpredict = qtable[o_prev, a]
    qtable[o_prev, a] += self.learning_rate * (qtarget - qpredict)


class Mixee1:
    def __init__(self):
      #  print('Mixee1 init')
        self.worker = Q_learner(epsilon=0.2, learning_rate=0.1, gamma=0.99)
    def act(self, obs):
        return self.worker.act(obs)
    def train(self, *args):
        self.worker.train(*args)

class Mixee2:
    def __init__(self):
      #  print('Mixee2 init')
        self.worker = Q_learner(epsilon=0.01, learning_rate=0.2, gamma=0.75)
    def act(self, obs):
        return self.worker.act(obs)
    def train(self, *args):
        self.worker.train(*args)

class CartPole_LowRes:
    def __init__(self):
        self.worker = gym.make("CartPole-v0")
        self.worker.seed(55)
    def reduce_resolution(self, obs):
        return tuple(int(x*100) for x in obs)
    def reset(self):
        return self.reduce_resolution(self.worker.reset())
    def step(self, action):
        obs, reward, done, misc_info = self.worker.step(action)
        obs = self.reduce_resolution(obs)
        return obs, reward, done, misc_info

def sample(distro):
    r = random()
    action_space_copy = action_space.copy()
    shuffle(action_space_copy)
    for a in action_space_copy:
        if distro[a] >= r:
            return a
        r -= distro[a]
    return action_space_copy[-1]

def agent_env_interaction(agent_class, env_class, nsteps):
    step = 0
    agent = agent_class()
    env = env_class()
    total_reward = 0
    obs = env.reset()

    while step < nsteps:
        step += 1
        distro = agent.act(obs)
        action = sample(distro)
        next_obs, reward, done, misc_info = env.step(action)
        if done:
            next_obs = env.reset()
            reward = -10
        total_reward += reward
        agent.train(obs, action, reward, next_obs, done)
        obs = next_obs

    return total_reward

total_rewards_mixee1 = []
total_rewards_mixee2 = []
coinflip = CoinFlip(Mixee1, Mixee2)
total_rewards_coinflip = []

for I in range(10000):
    #print("Testing Mixee1 in CartPole_LowRes",I)
    r = agent_env_interaction(Mixee1, CartPole_LowRes, 1000)
    total_rewards_mixee1.append(r)
    result1 = sum(total_rewards_mixee1)/len(total_rewards_mixee1)
    #print(f"#{I}, Avg total reward: {result1}")

    #print("Testing Mixee2 in CartPole_LowRes",I)
    r = agent_env_interaction(Mixee2, CartPole_LowRes, 1000)
    total_rewards_mixee2.append(r)
    result2 = sum(total_rewards_mixee2)/len(total_rewards_mixee2)
    #print(f"#{I}, Avg total reward: {result2}")

    #print("Testing CoinFlip in CartPole_LowRes",I)
    r = agent_env_interaction(coinflip, CartPole_LowRes, 1000)
    total_rewards_coinflip.append(r)
    result3 = sum(total_rewards_coinflip)/len(total_rewards_coinflip)
    #print(f"#{I}, Avg total reward: {result3}")

    avged_mixees = (result1+result2)/2
    #print(f"#{I}, Avg of mixees: {avged_mixees} vs. Mixture's avg: {result3}")
    print(f"#{I:6}: {result1:9.2f} {result2:9.2f} {avged_mixees:9.2f} vs {result3:9.2f}") 
