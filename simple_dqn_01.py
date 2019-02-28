#! python

# import random
from time import sleep
from collections import deque
import gym
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras import backend as K
import tensorflow as tf

class AtariPlayer:
    # ctor
    def __init__(self, gameName = 'Riverraid-ram-v0'):
        self.__env = gym.make(gameName)
        self.__currentState = self.__env.reset()
        self.__memory = deque()

    # add current 
    def __memorize(self, action, reward, next_state, done, info):
        self.__memory.append( (self.__currentState, action, reward, done, info, next_state) )

    def stateSize(self):
        return len()

    # environment description
    def printEnvironment(self):
        print(self.__currentState)
        print(self.__env.action_space)
        print(self.__env.observation_space)

    # perform a play session
    def runSession(self, maxSteps, shouldRender = False):
        for _ in range(maxSteps):
            self.__env.step(self.__env.action_space.sample())
            # state, reward, done, info
            action = self.__env.action_space.sample()
            next_state, reward, done, info =self.__env.step(action)
            self.__memorize(action, reward, next_state, done, info)

            if done == True:
                break
            else:
                # update __currentState
                self.__currentState = next_state

                # render if needed
                if shouldRender:
                    self.__env.render('human')
                    print("Current State: ")
                    print(self.__memory[-1])
                    sleep(0.034)

        return self.__memory

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

class DQNet:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

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

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss=self._huber_loss,
                      optimizer=Adam(lr=self.learning_rate))
        return model




if __name__ == '__main__':
    rr = AtariPlayer()
    rr.runSession(1000, True)
    # rr.printEnvironment()
    # print(rr.getCurrentObservations())
    rr.shutdown()