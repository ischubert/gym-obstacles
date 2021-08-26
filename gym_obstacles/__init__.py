"""
OpenAI-Gym Wrapper for PhysX-based Robotic Pushing Environment
"""
from gym.envs.registration import register

register(
    id='obstacles-v0',
    entry_point='gym_obstacles.envs:ObstaclesEnv',
)