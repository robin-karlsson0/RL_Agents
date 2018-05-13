import gym

env = gym.make("Pong-v0")

for ep in range(100):
	obs = env.reset()
	for step in range(10000):
		env.render()
		action = env.action_space.sample()
		obs, rew, done, _ = env.step(action)
		if done:
			break
