#! python

# import importlib as implib
import random
from time import sleep
from collections import deque
import gym
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras import backend as K
import tensorflow as tf

LOGDEBUG = False
MAXMEMORY = 5000

class AtariPlayer:
    # ctor
    def __init__(self, gameName = 'Riverraid-ram-v0'):
        self.__env = gym.make(gameName)
        self.__currentState = self.__env.reset()
        self.__memory = deque(maxlen=MAXMEMORY)

    # add current step to memory
    def _memorize(self, action, reward, next_state, done):
        self.__memory.append( (self.__currentState, action, reward, done, next_state) )

    def stateSize(self):
        return self.__env.observation_space.shape[0]

    def actionSize(self):
        return self.__env.action_space.n

    def getMemory(self):
        return self.__memory

    # environment description
    def printEnvironment(self):
        print(self.__currentState)
        print(self.__env.action_space)
        print(self.__env.observation_space)

    # perform a play session
    def runSession(self, maxSteps, shouldRender = False, actionGen = None):
        #clean up
        totalReward = 0
        self.__currentState = self.__env.reset()
        self.__memory = deque(maxlen=MAXMEMORY)

        # run the game
        for i in range(maxSteps):
            # get a action
            if actionGen != None:
                action = actionGen(self.__currentState)
                if LOGDEBUG:
                    print("DQNN Action: ", action)
            else:
                action = self.__env.action_space.sample()

            # perform a step and memorize outcome
            next_state, reward, done, _ = self.__env.step(action)
            self._memorize(action, reward, next_state, done)
            totalReward += reward

            # check if game is finished
            if done == True:
                break
            else:
                # update __currentState
                self.__currentState = next_state

                # render if needed
                if shouldRender:
                    sleep(0.020) # aprox 50 fps
                    self.__env.render('human')
                    if LOGDEBUG:
                        print("Current State: ")
                        print(self.__memory[-1])

        return (i, totalReward)

    # current environment data
    def getCurrentObservations(self):
        memLen = len(self.__memory)
        if memLen > 0:
            return self.__memory[-1]
        else:
            return None

    # close enrironment
    def shutdown(self):
        self.__env.close()


class DQNNetwork:
    def __init__(self, state_size, action_size, gamma = 0.95, epsilon = 1.0, epsilon_min = 0.01, epsilon_decay = 0.99,learning_rate = 0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma          # discount rate
        self.epsilon = epsilon      # exploration rate
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate

    def generateAction(self, state):
        s = np.reshape(state, [1, self.state_size])
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)

        print("DQNNetwork: Predcited Action")
        act_values = self.__model.predict(s)
        return np.argmax(act_values[0])  # returns action

    def trainModel(self, memory, batch_size):
        minibatch = random.sample(memory, batch_size)
        for state, action, reward, done, next_state in minibatch:
            cs = np.reshape(state, [1, self.state_size])
            ns = np.reshape(next_state, [1, self.state_size])
            # print(state)
            target = self.__model.predict(cs)

            if done:
                target[0][action] = reward
            else:
                t = self.__target_model.predict(ns)[0]
                target[0][action] = reward + self.gamma * np.amax(t)

            self.__model.fit(cs, target, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        DQNNetwork.copyModelWeights(self.__model, self.__target_model)

    def _huber_loss(self, y_true, y_pred, clip_delta=1.0):
        """Huber loss for Q Learning
            References: https://en.wikipedia.org/wiki/Huber_loss
                        https://www.tensorflow.org/api_docs/python/tf/losses/huber_loss
        """
        error = y_true - y_pred
        cond  = K.abs(error) <= clip_delta

        squared_loss = 0.5 * K.square(error)
        quadratic_loss = 0.5 * K.square(clip_delta) + clip_delta * (K.abs(error) - clip_delta)

        return K.mean(tf.where(cond, squared_loss, quadratic_loss))

    def _build_model(self, hiddenLayerSize):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(hiddenLayerSize, input_dim=self.state_size, activation='relu'))
        model.add(Dense(hiddenLayerSize, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss=self._huber_loss, optimizer=Adam(lr=self.learning_rate))
        return model

    def initModel(self, hiddenLayerSize = 128):
        self.__model = self._build_model(hiddenLayerSize)
        self.__target_model = self._build_model(hiddenLayerSize)
        DQNNetwork.copyModelWeights(self.__model, self.__target_model)
        self.__model.summary()

    def load(self, name):
        self.__model.load_weights(name)

    def save(self, name):
        self.__model.save_weights(name)

    @staticmethod
    def copyModelWeights(fromModel, toModel):
        # copy weights between models
        toModel.set_weights(fromModel.get_weights())

dqnn_ = None
player_ = None

def setup(layerDim = 24, gameName = None):
    ## Atari player
    global player_
    player_ = AtariPlayer()
    states = player_.stateSize()
    actions = player_.actionSize()
    # player_.printEnvironment()

    # deep reinforcement q-learn NN
    global dqnn_
    if gameName == None:
        dqnn_ = DQNNetwork(states, actions)
    else:
        dqnn_ = DQNNetwork(states, actions, gameName)
    dqnn_.initModel(layerDim)

def run(episondes = 5, shouldRender = False):
    global dqnn_
    global player_

    # train #episondes sessions
    actGen = dqnn_.generateAction
    for _ in range(episondes):
        # play a session
        actions, total = player_.runSession(MAXMEMORY, shouldRender, actGen)
        print('Performed ',actions,' iterations - Total Score: ', total)
        memory = player_.getMemory()

        # train the model
        if(len(memory)>=500):
            dqnn_.trainModel(memory, 500)

    # player_.printEnvironment()
    # print(player_.getCurrentObservations())

    player_.shutdown()

def loadModel(filename):
    global dqnn_
    dqnn_.load(filename)

def saveModel(filename):
    global dqnn_
    dqnn_.save(filename)

### main
if __name__ == '__main__':
    setup()
    loadModel('simple_dqnn.h5')
    run(100)
    saveModel('simple_dqnn.h5')