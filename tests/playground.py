# %%
import gym
import matplotlib.pyplot as plt

env_goal = gym.make(
    'gym_obstacles:obstacles-v0',
    plan_or_goal='goal',
    plan_length=None
)

env_plan = gym.make(
    'gym_obstacles:obstacles-v0',
    plan_or_goal='plan',
    plan_length=50
)

# %%
env_goal.reset()
env_goal.render()
plt.show()

env_plan.reset()
env_plan.render()
plt.show()

# %%
