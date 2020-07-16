
import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import os

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
        self.trueDiagnosis = 0

        self.stepcount = 0 # To keep track of number of questions asked

    def reset(self):
        """
        Resets the state of the environment and returns an initial observation.
        Returns:
            observation (object): the initial observation.
        """
        self.symptomState = np.zeros(len(self.symptomsQuestions.keys()))
        self.takenActions = set([])
        self.stepcount = 0
        print("\n########################## STATE RESET ##########################\n\n")
        self.trueDiagnosis = input("(Enter 1 for Anemia, 2 for Hyperthyroidism or 3 for Viral upper respiratory tract infection): ")
        return self.symptomState, self.trueDiagnosis

    
    def step(self, action):
        """
        Run one timestep of the environment's dynamics. When end of
        episode is reached, call `reset()`
        to reset this environment's state.
        Accepts an action and returns a tuple (observation/state, reward, done, info).
        Args:
            action (object): an action provided by the agent (An integer)
        Returns:
            observation/state (object): agent's observation of the current environment (state)
            reward (float) : amount of reward returned after previous action
            done (bool): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
        
        # Update State
        if action < len(self.symptomsQuestions): # action withing "question space" / not ternimal actions
            answer = self.getAnswer(action)
            self.symptomState[action] = answer # update state

        # Calculate Reward
        done = False
        if action in self.takenActions: # repeated action
            reward = -1
            print("REPEATED QUESTION !! ")
            done = True
        elif len(self.takenActions) > 11:
            reward = -1
            print("Too many queations !")
            done = True
        elif action >= len(self.symptomsQuestions): # terminal action / terminal state
            if (action - len(self.symptomsQuestions) + 1) == self.trueDiagnosis:
              reward = 1
              print("DIAGNOSED CORRECTLY !! ",self.trueDiagnosis)
            else:
              reward = -1
              print("FALSE DIAGNOSIS !! ",self.trueDiagnosis,"instead got: ", (action - len(self.symptomsQuestions) + 1))
            done = True
        else: reward = 0

        # Add Action
        self.takenActions.add(action)

        self.stepcount +=1
        
        return self.symptomState, reward, done, None

    def getAnswer(self, action):
        """
        Arg:
            action is an integer
        Assumption:
            user response is an integer (1,-1) Future: magnitude to indicate severity?
        """
#         print(action)
        symptom = list(self.symptomsQuestions.keys())[action]
        while True:
          answer = input(self.symptomsQuestions[symptom]+"(Answer 1 or -1): ")
          if answer not in ("1","-1"):
            print("Invalid answer, please try again")
          else: break
        return answer

    def render(self):
        print('------------------------------------')
        print("Step: ", self.stepcount)
        print("Current state: ", self.symptomState)
        print("Actions taken: ", set(self.takenActions))
        print('------------------------------------')


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
        model.add(Dense(self.action_size, activation='linear')) 
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # list of previous experiences, enabling re-training later

    def act(self, state):
        if np.random.rand() <= self.epsilon: # if acting randomly, take random action
            return random.randrange(self.action_size)
        act_values = self.model.predict(state) # if not acting randomly, predict reward value based on current state
        return np.argmax(act_values[0]) # pick the action that will give the highest reward

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

    symptomsQuestions = {"cough" : "Do you have a cough?",
                        "cough_color" : "Are you coughing up yellow, green or brown phlegm ?",
                        "cough_blood" : "Are you coughing up blood?",
                        "duration_hours" : "Have you had your complaints for hours ?",
                        "duration_days" : "Have you had your complaints for days ?",
                        "duration_weeks" : "Have you had your complaints for weeks ?",
                        "duration_months" : "Have you had your complaints for months ?",
                        "history_smoking" : "Do you smoke ?",
                        "history_allergy" : "Do you have a history of allergies, asthma,eczema ?",
                        "history_chestpain" : "Do you have any chest pain ?",
                        "history_immunity" : "Do you have a significantly weakned immunes system due to an existing condition or from taking medication ?",
                        "history_tired" : "Are you feeling tired, unwell, lethargic or run down ?",
                        "nose_blocked" : "Do you have blocked or stuffy nose ?",
                        "nose_runny" : "Do you have running or dripping nose ?",
                        "nose_sneeze" : "Do you have sneezing more than usual ?",
                        "nose_smell" : "Can  you smell things as well as normal ?",
                        "swallow_difficulty" : "Are you finding it difficult to swallow or having painful to swallow ?",                         
                        "swallow_solidfood" : "Are you having trouble swallowing solid food ?",                         
                        "swallow_fluids" : "Are you having trouble swallowing fluids ?",                         
                        "headache" : "Do you have a headache ?",                         
                        "headache_worst" : "Is the worst headache you can imagine ? ",
                        "headache_gradual" : "Did your headache start gradually ?",
                        "headache_sudden" : "Did your headache start very suddenly?",
                        "headache_discontinous" : "Does your headache come and go ?",
                        "headache_continous" : "Does your headache continue without interruption?",
                        "headache_neck" : "Do you have any problems moving your neck ?",
                        "headache_speech" : "Have you had any speech problems or noticed changes to your voice ?",
                        "headache_muscle" : "Have you noticed any abnormal muscle weakness lately ?",
                        "headache_vision" : "Have you noticed any change to your vision ?",
                        "headache_confused" : "Are you feeling confused or unable to remember things ?",                                     
                        "fever" : "Do you have a fever ?",                                     
                        "fever37" : "Is body temperature lower than  37 degree?",                                     
                        "fever38" : "Is body temperature between 37-38 degrees?",                                     
                        "fever40" : "Is body temperature between 38-40 degrees?",                                     
                        "fever41" : "Is body temperature greater than 40 degree?",                                     
                        "facepain" : "Do you have any pain in your face ?",                                     
                        "facepain_sinus" : "Is pain in your sinus ?",                                     
                        "musclepain" : "Do your muscles ache ?",
                        "jointpain" : "Do you have joint or bone pain in several areas of your body ?",                           
                        "throat" : "Do you have a sore throat ?",  
                        "throat_red" : "Do you have redness at the back of your throat?",  
                        "throat_tonsils" : "Do you have swollen tonsils?",  
                        "throat_white" : "Do you have white spots on your tonsils?",  
                        "throat_lump" : "Do you have lump in your throat?",  
                        "lung" : "Are you currently diagnosed with any of lung disorders?",  
                        "breathing" : "Have you experienced any difficulty breathing",  
                        "breathing_better" : "Is your difficulty breathing getting better ?",  
                        "breathing_same" : "Is your difficulty breathing staying the same ?",  
                        "breathing_worse" : "Is your difficulty breathing getting worse ?",  
                        "breathing_blue" : "Have your hands and feet gone blue",  
                        "breathing_sound" : "Have you noticed your breathing sounds wheezy or noisy"
                     }
   
    diseases = ["Anemia",
                "Hyperthyroidism",
                "Viral upper respiratory tract infection"]
    state_size = len(symptomsQuestions.keys())
    action_size = len(symptomsQuestions.keys()) + len(diseases)
    batch_size = 32
    n_episodes = 5 # n times we want agent to train (default 1001)
    output_dir = 'model_output/v0/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    agent = DQNAgent(state_size, action_size) # initialise agent
    env = healtius_env(symptomsQuestions, diseases)

    # Load weights
    weights_file_name = input("Enter most recent model file name: ")
    agent.load(output_dir + weights_file_name)


    done = False

    
    for e in range(n_episodes): # iterate over new episodes of the game
        [state,_] = env.reset() # reset state at start of each new episode of the game
        state = np.reshape(state, [1, state_size])

        for time in range(100):  # do we need this? 
            env.render()
            action = agent.act(state) # agent picks an action
            next_state, reward, done, _ = env.step(action) # agent interacts with env, gets feedback       
            #reward = reward if not done else 10        
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done) # remember the previous timestep's state, actions, reward, etc.        
            state = next_state # set "current state" for upcoming iteration to the current next state        
            if done: # episode ends if agent drops pole or we reach timestep 5000
                print("episode: {}/{}, steps: {}, e: {:.2}" # print the episode's score and agent's epsilon
                    .format(e, n_episodes, time, agent.epsilon))
                break # exit loop
        if len(agent.memory) > batch_size:
            agent.replay(batch_size) # train the agent by replaying the experiences of the episode
        if e % 50 == 0:
            agent.save(output_dir + "weights_" + '{:04d}'.format(e) + ".hdf5")
