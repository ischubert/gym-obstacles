# %%
import gym
import matplotlib.pyplot as plt

env_goal = gym.make(
    'gym_obstacles:obstacles-v0',
    plan_or_goal='goal',
    plan_length=None,
    n_boxes=3,
    planner_tolerance=0.05
)

env_plan = gym.make(
    'gym_obstacles:obstacles-v0',
    plan_or_goal='plan',
    plan_length=20,
    n_boxes=3,
    planner_tolerance=0.05
)

env_goal.reset()
env_goal.render()
plt.show()

env_plan.reset()
env_plan.render()
plt.show()

# %%
env_plan.reset()
env_plan.render()
plt.show()

# %%
observation, reward, done, info = env_plan.step([1, 0])
print(observation, reward, done, info)
env_plan.render()
plt.show()

# %%
