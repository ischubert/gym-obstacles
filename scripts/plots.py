# %%
"""
Visualize environment for README
"""
import gym
import matplotlib.pyplot as plt

env = gym.make(
    'gym_obstacles:obstacles-v0',
    plan_or_goal='plan',
    plan_length=20,
    n_boxes=3,
    planner_tolerance=0.05
)

for ind in range(3):
    env.reset()
    env.render()
    plt.savefig(str(ind) + "_example.png")
    plt.show()

# %%
