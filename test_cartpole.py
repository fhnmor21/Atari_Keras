import gym

env = gym.make('CartPole-v0')
print( env.reset() )

# environment
print(env.action_space)
print(env.observation_space)

for _ in range(1000):
    env.render()
     # take a random action
    next_state, reward, done, info = env.step(env.action_space.sample())
    # print(next_state, reward, done, info)

env.close()