from gym.spaces import Box
import numpy as np
import random

class QTables(): 
    def __init__(self, observation_space, action_space, r, lr, eps=0.1):
        self.observation_space = observation_space
        self.action_space = action_space
        self.obs_num = observation_space.shape[0]
        self.action_num = action_space.n
        self.q_tables = [np.zeros((self.obs_num*self.obs_num, self.action_num)) for i in range(self.obs_num)]
        self.r = r
        self.lr = lr
        #self.drone_num = drone_num
        self.eps = eps

        if isinstance(observation_space, Box):
            high = observation_space.high[0]
            low = observation_space.low[0]
        else:
            high = observation_space[0].high[0]
            low = observation_space[0].low[0]

        self.size = int(high - low) + 1
#        self.q_tables = np.zeros([self.size**2, self.action_num])
        self.q_tables = [np.zeros((self.obs_num*self.obs_num, self.action_num)) for i in range(self.obs_num)]

        self.q_tables_count = np.zeros([self.size**2, self.action_num])
    
    # support function: convert the fov to the unique row number in the q table
#    def obs_to_row(self, obs_array):
#        return obs_array[0] * self.size + obs_array[1]
    def obs_to_row(self, obs, obs_array):
        obs_array = np.asarray(obs)
        print("obs_array type:", type(obs_array))  # add this line
        obs_row = obs_array[0] * self.size + obs_array[1]
        return obs_row  

    def get_action(self, obs, i, obs_array):
       #if len(obs) <= i or len(obs_array) <= i:
        #    raise ValueError("Invalid input: obs and obs_array must have length greater than i")

        if len(obs) != self.obs_num or len(obs_array) != self.obs_num:
            raise ValueError(f"Invalid input: obs and obs_array must have length {self.obs_num}")
        obs_row = self.obs_to_row(obs[i], obs_array[i])

        if np.random.rand() < self.eps:
            action = self.action_space.sample()
            greedy = False
        else:
            action = np.argmax(self.q_tables[i][obs_row*self.action_num : (obs_row+1)*self.action_num])
            greedy = True

        return action, greedy
    
    def update_eps(self):
        # update the epsilon
        if self.eps > self.eps_end: # lower bound
            self.eps *= self.r
            
    def train(self, obs, obs_next, action, reward, done, i):
        obs_row = self.obs_to_row(obs[i].astype(int))
        obs_next_row = self.obs_to_row(obs_next[i].astype(int))

        q_current = self.q_tables[i][obs_row*self.action_num + action] # current q value
        q_next_max = np.max(self.q_tables[i][obs_next_row*self.action_num : (obs_next_row+1)*self.action_num]) # the maximum q value in the next state

        # update the q value
        if done:
            self.q_tables[i][obs_row*self.action_num + action] = q_current + self.lr * reward
        else:
            self.q_tables[i][obs_row*self.action_num + action] = q_current + self.lr * (reward + self.gamma * q_next_max - q_current)

        # update the count
        self.q_tables_count[i][obs_row*self.action_num + action] += 1


