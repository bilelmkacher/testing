#--------- Training ---------#
import sys
sys.path.append('/home/bilelmkacher/python_trainings/autonomous-drone-swarm/QL/')
from Grid import Grid
import numpy as np
import random
import time
from qtables_pos import QTables
#from qtables_fov import QTables
from gym.spaces import Box
# metrics for each episode
time_steps = [] # number of time steps in total
epsilons = [] # epsilon at the end of each episode
greedy = [] # the ratio of greedy choices
coverage = [] # the ratio of visited cells at the end
speed = [] # number of time steps to cover decent amount of cells
sum_q_values = [] # sum of q-values
results_mapping = [] # mapping status
results_count = [] # count status
total_reward = []
coverage_threshold = 0.95 #
obs_array =  []
# parameters for training
train_episodes = 50000
max_steps = 5 * 5 * 5

# initialize the environment and the q table
env = Grid(x_size=5, y_size=5, n_agents=1, fov_x=3, fov_y=3, simple=True)
#q = QTables(observation_space=env.observation_space, action_space=env.action_space, r=0.9999995, lr=0.7)
q = QTables(observation_space=env.observation_space, action_space=env.action_space, r=0.9999995, lr=0.7) #, drone_num=1

# training
for episode in range(train_episodes):
    state = env.reset()
    state = [arr.astype('int') for arr in state] # convert from float to integer

    greedy_count = 0 # shared with all agents
    coverage_track = True
    epi_reward = 0
    
    for step in range(max_steps):
        i = 0
        state = env.reset()
        #obs_array = np.zeros((10,2))
        #obs_array[0] = np.array([state[0], state[1]])
        #obs_array[1] = np.array([state[2], state[3]])
        #obs_array[2] = np.array([state[4], state[5]])
        #obs_array[3] = np.array([state[6], state[7]])
        #obs_array[4] = np.array([state[8], state[9]])
        #obs_array[5] = np.array([state[10], state[11]])
        #obs_array[6] = np.array([state[12], state[13]])
        #obs_array[7] = np.array([state[14], state[15]])
        #obs_array[8] = np.array([state[16], state[17]])
        new_obs_array = np.zeros((10,2))
        new_obs_array[:4] = obs_array[:4]
        new_obs_array[4] = state[8]
        new_obs_array[5] = state[9]
        new_obs_array[6:] = obs_array[4:]
        #new_obs_array[4] = np.array([state[8], state[9]])

        for i in range(9):
            #result = q.get_action(state, i, new_obs_array)
            #actions, greedy_tf = q.get_action(state, i)
            #actions, greedy_tf = q.get_action(state, i, obs_array)
            result = q.get_action(state, i, new_obs_array)
            if result is not None:
                actions, greedy_tf = result
            else:
                actions, greedy_tf = None, None

            next_state, rewards, done, info = env.step(actions)
            next_state = [arr.astype('int') for arr in next_state] # convert from float to integer
            #q.train(state, next_state, actions, rewards)
            q.train(state, next_state, actions, rewards, done, i)
            epi_reward += rewards[0]
            greedy_count += sum(greedy_tf)

            # check if decent amoung of cells are visited
            current_coverage = env.get_coverage()
            if current_coverage >= coverage_threshold and coverage_track:
                speed.append(step)
                coverage_track = False

            # check if the task is completed
            if all(done):
                time_steps.append(step)
                break
            elif step == max_steps - 1:
                time_steps.append(step)
                if coverage_track:
                    speed.append(np.nan)
            
            # update the observation
            state = next_state
        
    epsilons.append(q.eps)
    coverage.append(env.get_coverage())
    greedy.append(greedy_count / ((step + 1) * env.n_agents)) # multiply by step
    sum_q_values.append(q.q_tables[0].sum())
    results_mapping.append(env.grid_status)
    results_count.append(env.grid_counts)
    total_reward.append(epi_reward)


    print('//Episode {0}//    Epsilon: {1:.3f},    Steps: {2},    Greedy Choices???(%): {3:.3f},    Coverage (%): {4:.3f},    Steps to Visit {5}% Cells: {6},    Sum of Q-Values: {7:.1f},    Total Reward: {8}'\
          .format(episode+1, q.eps, step+1, greedy[episode], coverage[episode], coverage_threshold * 100, speed[episode], sum_q_values[episode], total_reward[episode]))



#--------- Test ---------#

# metrics for each episode
time_steps_test = [] # number of time steps in total
coverage_test = [] # the ratio of visited cells at the end
speed_test = [] # number of time steps to cover decent amount of cells
results_mapping_test = [] # mapping status
results_count_test = [] # count status
total_reward = []
coverage_threshold = 0.95 #

# parameters for training
test_episodes = 100
max_steps_test = 5 * 5 * 2

# initialize the environment
env_test = Grid(x_size=5, y_size=5, n_agents=1, fov_x=3, fov_y=3, simple=True)

# training
for episode in range(test_episodes):
    state = env_test.reset()
    state = [arr.astype('int') for arr in state] # convert from float to integer

    coverage_track = True
    epi_reward = 0

    for step in range(max_steps_test):
        actions, greedy_tf = q.get_action(state)
        next_state, rewards, done, info = env_test.step(actions)
        next_state = [arr.astype('int') for arr in next_state] # convert from float to integer
        epi_reward += rewards[0]

        # check if decent amoung of cells are visited
        current_coverage = env_test.get_coverage()
        if current_coverage >= coverage_threshold and coverage_track:
            speed_test.append(step)
            coverage_track = False

        # check if the task is completed
        if all(done):
            time_steps_test.append(step)
            break
        elif step == max_steps_test - 1:
            time_steps_test.append(step)
            if coverage_track:
                speed_test.append(np.nan)
        
        # update the observation
        state = next_state
    
    coverage_test.append(env_test.get_coverage())
    results_mapping_test.append(env_test.grid_status)
    results_count_test.append(env_test.grid_counts)
    total_reward.append(epi_reward)

    print('//Episode {0}//    Steps: {1},    Coverage (%): {2:.3f},    Steps to Visit {3}% Cells: {4},    Total Reward {5}'\
          .format(episode+1,  step+1,  coverage_test[episode], coverage_threshold * 100, speed_test[episode], total_reward[episode]))
