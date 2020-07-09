import numpy as np
class symp_world(object):
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

        self.possibleActions = list(symptomsQuestions.keys()) + diseases
        self.symptomState = np.zeros(len(symptomsQuestions.keys()))

    def reset(self):
        pass

    
    
    def step(self, action):
        """Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.
        Accepts an action and returns a tuple (observation, reward, done, info).
        Args:
            action (object): an action provided by the agent
        Returns:
            observation (object): agent's observation of the current environment (state)
            reward (float) : amount of reward returned after previous action
            done (bool): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """

        # resultingState = self.symptomState

        answer = self.getAnswer(action)

        index = list(self.symptomsQuestions.keys()).index(action)
        self.symptomState[index] = answer

        return self.symptomState, reward, done, None

    def getAnswer(self, action):
        """
        Asks action to user, get response from user
        Assumption:
            - user response is an integer
        """
        answer = input(self.symptomsQuestions[action])
        return answer

    def changeState(self, action, answer):
        index = list(self.symptomsQuestions.keys()).index(action)
        self.symptomState[index] = answer
