import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import os # for creating directories

class healtius_env(object):
    def __init__(self, symptomsQuestions, diseases):
        """
        symptomsQuestions: 
            dictionary of questions mapped to symptoms.
            e.g. { "asthma" : "Have you been diagnosed with asthma?", "sweating_more" : "Are you sweating more than usual?"}
        diseases: 
            list of terminal states / diagnosis
            e.g. [diabetes, fever, ...]
        """
        self.symptomsQuestions = symptomsQuestions
        self.diseases = diseases

        self.possibleActions = list(self.symptomsQuestions.keys()) + diseases
        self.symptomState = np.zeros(len(self.symptomsQuestions.keys()))
        self.takenActions = set([])

    def reset(self):
        """
        Resets the state of the environment and returns an initial observation.
        Returns:
            observation (object): the initial observation.
        """
        self.symptomState = np.zeros(len(self.symptomsQuestions.keys()))
        self.takenActions = set([])
        print("RESET -----")
        return self.symptomState
    
    
    def step(self, action):
        """
        Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.
        Accepts an action and returns a tuple (observation, reward, done, info).
        Args:
            action (object): an action provided by the agent (An integer)
        Returns:
            observation (object): agent's observation of the current environment (state)
            reward (float) : amount of reward returned after previous action
            done (bool): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
        
        # Update State
        if action < len(self.symptomsQuestions):
            answer = self.getAnswer(action)
    #         index = list(self.symptomsQuestions.keys()).index(action)
            self.symptomState[action] = answer

        # Calculate Reward
        done = False
        if action in self.takenActions:
            reward = -1
        elif action > len(self.symptomsQuestions):
            reward = 1
            print("diagnosed: ",action)
            done = True
        else: reward = 0

        # # Is Terminal State
        # done = action in self.diseases

        # Add Action
        self.takenActions.add(action)

        
        return self.symptomState, reward, done, None

    def getAnswer(self, action):
        """
        Arg:
            action is an integer
        Assumption:
            user response is an integer
        """
#         print(action)
#         print(list(self.symptomsQuestions.keys()))
        symptom = list(self.symptomsQuestions.keys())[action]
        answer = input(self.symptomsQuestions[symptom]+"(Answer 1 or -1): ")
        return answer

    def render(self):
        print('------------------------------------------')
        print(self.symptomState)
        print(set(self.takenActions))
        print('------------------------------------------')


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000) # double-ended queue; acts like list, but elements can be added/removed from either end
        self.gamma = 0.95 # decay or discount rate: enables agent to take into account future actions in addition to the immediate ones, but discounted at this rate
        self.epsilon = 1.0 # exploration rate: how much to act randomly; more initially than later due to epsilon decay
        self.epsilon_decay = 0.995 # decrease number of random explorations as the agent's performance (hopefully) improves over time
        self.epsilon_min = 0.01 # minimum amount of random exploration permitted
        self.learning_rate = 0.001 # rate at which NN adjusts models parameters via SGD to reduce cost 
        self.model = self._build_model() # private method 
    
    def _build_model(self):
        # neural net to approximate Q-value function:
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu')) # 1st hidden layer; states as input
        model.add(Dense(24, activation='relu')) # 2nd hidden layer
        model.add(Dense(self.action_size, activation='linear')) # 2 actions, so 2 output neurons: 0 and 1 (L/R)
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # list of previous experiences, enabling re-training later

    def act(self, state):
        if np.random.rand() <= self.epsilon: # if acting randomly, take random action
            return random.randrange(self.action_size)
        act_values = self.model.predict(state) # if not acting randomly, predict reward value based on current state
        return np.argmax(act_values[0]) # pick the action that will give the highest reward (i.e., go left or right?)

    def replay(self, batch_size): # method that trains NN with experiences sampled from memory
        minibatch = random.sample(self.memory, batch_size) # sample a minibatch from memory
        for state, action, reward, next_state, done in minibatch: # extract data for each minibatch sample
            target = reward # if done (boolean whether game ended or not, i.e., whether final state or not), then target = reward
            if not done: # if not done, then predict future discounted reward
                target = (reward + self.gamma * # (target) = reward + (discount rate gamma) * 
                          np.amax(self.model.predict(next_state)[0])) # (maximum target Q based on future action a')
            target_f = self.model.predict(state) # approximately map current state to future discounted reward
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0) # single epoch of training with x=state, y=target_f; fit decreases loss btwn target_f and y_hat
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


if __name__ == '__main__':

    symptomsQuestions = {"symptom0" : "question0",
                        "symptom1" : "question1",
                        "symptom2" : "question2",
                        "symptom3" : "question3",
                        "symptom4" : "question4",
                        "symptom5" : "question5",
                        "symptom6" : "question6",
                        "symptom7" : "question7",
                        "symptom8" : "question8",
                        "symptom9" : "question9",
                        "symptom10" : "question10",
                        "symptom11" : "question11",
                        "symptom12" : "question12",
                        "symptom13" : "question13",
                        "symptom14" : "question14",
                        "symptom15" : "question15",
                        "symptom16" : "question16",
                        "symptom17" : "question17",
                        "symptom18" : "question18",
                        "symptom19" : "question19",
                        "symptom20" : "question20",
                        "symptom21" : "question21",
                        "symptom22" : "question22",
                        "symptom23" : "question23",
                        "symptom24" : "question24",
                        "symptom25" : "question25",
                        "symptom26" : "question26",
                        "symptom27" : "question27",
                        "symptom28" : "question28",
                        "symptom29" : "question29",
                        "symptom30" : "question30",
                        "symptom31" : "question31",
                        "symptom32" : "question32",
                        "symptom33" : "question33",
                        "symptom34" : "question34",
                        "symptom35" : "question35",
                        "symptom36" : "question36",
                        "symptom37" : "question37",
                        "symptom38" : "question38",
                        "symptom39" : "question39",
                        "symptom40" : "question40"}
    diseases = ["disease0",
                "disease1",
                "disease2",
                "disease3",
                "disease4",
                "disease5",
                "disease6",
                "disease7",
                "disease8",
                "disease9",
                "disease10",
                "disease11",
                "disease12",
                "disease13",
                "disease14",
                "disease15",
                "disease16",
                "disease17",
                "disease18",
                "disease19",
                "disease20",
                "disease21",
                "disease22",
                "disease23",
                "disease24",
                "disease25",]
    state_size = len(symptomsQuestions.keys())
    action_size = len(symptomsQuestions.keys()) + len(diseases)
    batch_size = 32
    n_episodes = 1001 # n games we want agent to play (default 1001)
    output_dir = 'model_output/v0/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    agent = DQNAgent(state_size, action_size) # initialise agent
    env = healtius_env(symptomsQuestions, diseases)

    done = False

    
    for e in range(n_episodes): # iterate over new episodes of the game
        state = env.reset() # reset state at start of each new episode of the game
        state = np.reshape(state, [1, state_size])

        for time in range(100):  # time represents a frame of the game; goal is to keep pole upright as long as possible up to range, e.g., 500 or 5000 timesteps
            env.render()
            action = agent.act(state) # action is either 0 or 1 (move cart left or right); decide on one or other here
            next_state, reward, done, _ = env.step(action) # agent interacts with env, gets feedback; 4 state data points, e.g., pole angle, cart position        
            reward = reward if not done else 10 # reward +1 for each additional frame with pole upright        
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done) # remember the previous timestep's state, actions, reward, etc.        
            state = next_state # set "current state" for upcoming iteration to the current next state        
            if done: # episode ends if agent drops pole or we reach timestep 5000
                print("episode: {}/{}, score: {}, e: {:.2}" # print the episode's score and agent's epsilon
                    .format(e, n_episodes, time, agent.epsilon))
                break # exit loop
        if len(agent.memory) > batch_size:
            agent.replay(batch_size) # train the agent by replaying the experiences of the episode
        if e % 50 == 0:
            agent.save(output_dir + "weights_" + '{:04d}'.format(e) + ".hdf5")
