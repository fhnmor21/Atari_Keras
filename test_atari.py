import gym

env = gym.make('SpaceInvaders-v0')
print( env.reset() )

# environment
print(env.action_space)
print(env.observation_space)

for _ in range(5000):
    env.render('human')
    env.step(env.action_space.sample())
    next_state, reward, done, info = env.step(env.action_space.sample())
    # print(next_state, reward, done, info)

env.close()
