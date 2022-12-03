from random import random, shuffle
import gym

action_space = [0,1]

def Mix(mixees, weights):
    assert len(mixees) == len(weights)
    assert all([w>0 for w in weights])
    assert sum(weights)==1
    I = range(len(weights))

    class Mixture:
        def __init__(self):
            self.workers = [mixee() for mixee in mixees]
            self.probs = [1 for mixee in mixees]

        def act(self, obs):
            if sum(self.probs) == 0:
                return {a: 1.0/len(action_space) for a in action_space}

            denominator = sum([weights[i]*self.probs[i] for i in I])
            distros = [self.workers[i].act(obs) for i in I]

            distro = {}
            for action in action_space:
                proposals = [distros[i][action] for i in I]
                summands = [weights[i]*proposals[i]*self.probs[i] for i in I]
                numerator = float(sum(summands))
                distro[action] = numerator/denominator

            return distro

        def train(self, o_prev, action, *args):
            for i in I:
                distro = self.workers[i].act(o_prev)
                self.probs[i] *= distro[action]
                self.workers[i].train(o_prev, action, *args)

    return Mixture


class Q_learner:
  def __init__(self, epsilon, learning_rate, gamma):
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
        self.worker = Q_learner(epsilon=0.2, learning_rate=0.1, gamma=0.99)
    def act(self, obs):
        return self.worker.act(obs)
    def train(self, *args):
        self.worker.train(*args)


class Mixee2:
    def __init__(self):
        self.worker = Q_learner(epsilon=0.5, learning_rate=0.2, gamma=0.75)
    def act(self, obs):
        return self.worker.act(obs)
    def train(self, *args):
        self.worker.train(*args)


class CartPole_LowRes:
    def __init__(self):
        self.worker = gym.make("CartPole-v0")
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


print("Testing Mixee1 in CartPole_LowRes")
total_rewards_mixee1 = []
for _ in range(500):
    r = agent_env_interaction(Mixee1, CartPole_LowRes, 1000)
    total_rewards_mixee1.append(r)
result1 = sum(total_rewards_mixee1)/len(total_rewards_mixee1)
print(f"Avg total reward: {result1}")


print("Testing Mixee2 in CartPole_LowRes")
total_rewards_mixee2 = []
for _ in range(500):
    r = agent_env_interaction(Mixee2, CartPole_LowRes, 1000)
    total_rewards_mixee2.append(r)
result2 = sum(total_rewards_mixee2)/len(total_rewards_mixee2)
print(f"Avg total reward: {result2}")


mixture = Mix([Mixee1, Mixee2], [0.5, 0.5])


print("Testing mixture in CartPole_LowRes")
total_rewards_mixture = []
for _ in range(500):
    r = agent_env_interaction(mixture, CartPole_LowRes, 1000)
    total_rewards_mixture.append(r)
result3 = sum(total_rewards_mixture)/len(total_rewards_mixture)
print(f"Avg total reward: {result3}")


avged_mixees = (result1+result2)/2
print(f"Avg of mixees: {avged_mixees} vs. Mixture's avg: {result3}")