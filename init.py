import gym
gym.logger.set_level(40)

#env = gym.make("Taxi-v2").env

env = gym.make("CartPole-v0").env
"""
print (env.action_space)
env.render()
"""

done =False;
while not done:
    observation, reward, done, _ = env.step(env.action_space.sample())




env.close()