#! python

from time import sleep
from collections import deque
import gym

class AtariPlayer:
    # ctor
    def __init__(self, gameName = 'Riverraid-ram-v0'):
        self.__env = gym.make(gameName)
        self.__currentEnviron = self.__env.reset()
        self.__memory = deque(self.__currentEnviron)

    # add current 
    def __memorize(self):
        self.__memory.append(self.__currentEnviron)

    # environment description
    def printEnvironment(self):
        print(self.__currentEnviron)
        print(self.__env.action_space)
        print(self.__env.observation_space)

    # perfomr a play sessions
    def runSession(self, maxSteps, shouldRender = False):
        for _ in range(maxSteps):
            self.__env.step(self.__env.action_space.sample())
            # next_state, reward, done, info
            self.__currentEnviron = self.__env.step(self.__env.action_space.sample())

            if self.__currentEnviron[2] == True:
                break
            else:
                self.__memorize()

            if shouldRender:
                self.__env.render('human')
                print(self.__currentEnviron)
                sleep(0.034)

        return self.__memory

    # current environment data
    def getCurrentEnvironment(self):
        return self.__currentEnviron

    # close enrironment
    def shutdown(self):
        self.__env.close()

if __name__ == '__main__':
    rr = AtariPlayer()
    rr.runSession(5000, True)
    pass