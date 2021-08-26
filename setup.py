"""
Set up gym_obstacles
"""
from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='gym_obstacles',
    version='0.0.1',
    author="Ingmar Schubert",
    author_email="mail@ingmarschubert.com",
    description="Obstacle avoidance environment for training plan-conditioned policies",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ischubert/gym-obstacles",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=['numpy', 'gym', 'matplotlib'],
    extras_require={
        "testing": ['pytest', 'stable-baselines3==1.0']
    }
)
