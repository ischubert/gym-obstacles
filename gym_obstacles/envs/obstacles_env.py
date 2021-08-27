"""
2D maze obstacle environments
"""
import numpy as np
import matplotlib.pyplot as plt
import gym

class ObstaclesEnv(gym.Env):
    """
    2D maze obstacle environment
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, plan_or_goal, plan_length, n_boxes):
        self.plan_or_goal = plan_or_goal
        self.plan_length = plan_length

        # create obstacle boxes
        boxes_origin = np.random.rand(
            n_boxes, 2
        )
        boxes_end = boxes_origin.copy()
        mask = np.ones(boxes_end.shape)
        while np.any(mask):
            boxes_end = np.where(
                mask,
                boxes_origin - 0.5 + 1.0* np.random.rand(
                    n_boxes, 2
                ),
                boxes_end
            )
            # First criterion: Box is included in state space
            mask = np.logical_or(
                boxes_end>1, boxes_end<0
            )
            # Second criterion: Box is at least 0.1 in width
            mask = np.logical_or(
                mask,
                np.abs(boxes_end - boxes_origin) < 0.1
            )
        
        self.boxes = np.concatenate(
            (boxes_origin, boxes_end),
            axis=-1
        )

        assert self.plan_or_goal in ["plan", "goal"]
        if self.plan_or_goal == "plan":
            assert self.plan_length > 1

        state_space = gym.spaces.Box(
            low=np.array([0, 0]),
            high=np.array([1, 1])
        )
        achieved_goal_space = gym.spaces.Box(
            low=np.array([0, 0]),
            high=np.array([1, 1])
        )

        if self.plan_or_goal == "goal":
            desired_goal_space = gym.spaces.Box(
                low=np.array([0, 0]),
                high=np.array([1, 1])
            )
        if self.plan_or_goal == "plan":
            desired_goal_space = gym.spaces.Box(
                low=np.array(self.plan_length*[0, 0]),
                high=np.array(self.plan_length*[1, 1])
            )

        self.observation_space = gym.spaces.Dict(
            spaces={
                "observation": state_space,
                "achieved_goal": achieved_goal_space,
                "desired_goal": desired_goal_space
            },
        )

        self.action_space = gym.spaces.Box(
            low=np.array([-0.1, -0.1]),
            high=np.array([0.1, 0.1])
        )

        self.state = None
        self.achieved_goal = None
        self.desired_goal = None

        self.current_target = None


    def step(self, action):
        """
        Simulate the system's transition under an action
        """
        print("CAUTION: ADD NOISE")
        # clip action to [-0.1, 0.1]
        action = np.clip(action, -0.1, 0.1)
        assert action in self.action_space

        # update self.state
        candidate = self.state + action
        if candidate in self.observation_space["observation_space"]:
            if self._not_in_collision(candidate):
                self.state = candidate
        
        # update self.achieved_goal
        self.achieved_goal = self.state.copy()

        # collect output
        observation = {
            'observation': self.state.copy(),
            'achieved_goal': self.achieved_goal.copy(),
            'desired_goal': self.desired_goal.copy()
        }
        done = False
        info = {
            "is_success": float(self._get_binary_reward(
                self.achieved_goal.reshape(1, 2),
                self.current_target.reshape(1, 2)
            ))
        }
        reward = float(self.compute_reward(
            self.achieved_goal.reshape(1, 2),
            self.desired_goal.reshape(1, -1),
            [info]
        ))

        return observation, reward, done, info

    def reset(self):
        """
        Reset environment to random state and random desired goal
        """
        # update self.state
        self.state = self._sample_non_colliding_state()
        
        # update self.achieved_goal
        self.achieved_goal = self.state.copy()

        # update self.current_target
        self.current_target = self._sample_non_colliding_state()

        # update self.desired_goal
        if self.plan_or_goal == "goal":
            self.desired_goal = self.current_target.copy()
        if self.plan_or_goal == "plan":
            self.desired_goal = self._sample_feasible_plan(self.state, self.current_target)
        
        # collect output
        observation = {
            'observation': self.state.copy(),
            'achieved_goal': self.achieved_goal.copy(),
            'desired_goal': self.desired_goal.copy()
        }

        return observation

    def render(self, mode='human'):
        """
        Create interactive view of the environment
        """
        print("TODO: OBSTACLE VIZ STILL MISSING HERE")
        print("TODO: REWARD VIZ STILL MISSING HERE")
        fig, ax = plt.subplots(figsize=(5, 5))

        # add reward
        achieved_goal_plot = np.array(np.meshgrid(
            np.linspace(0, 1, 50),
            np.linspace(0, 1, 50)
        )).reshape(2, -1).T[:, ::-1]
        rewards = self.compute_reward(
            achieved_goal_plot,
            np.repeat(
                self.desired_goal[None, :],
                2500,
                axis=0
            ),
            [
                {
                    "is_success": float(self._get_binary_reward(
                        achieved_goal_now.reshape(1, 2),
                        self.current_target.reshape(1, 2)
                    ))
                }
                for achieved_goal_now in achieved_goal_plot
            ]
        )

        # draw -1 everywhere where collision happens
        not_in_collision = np.array(
            [self._not_in_collision(achieved_goal_now) for achieved_goal_now in achieved_goal_plot]
        )
        image = np.where(
            not_in_collision,
            rewards,
            -1
        )

        ax.imshow(image.reshape(50, 50).T[::-1], extent=[0, 1, 0, 1])

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        ax.scatter(self.state[0], self.state[1], marker='x')
        ax.scatter(self.current_target[0], self.current_target[1], marker='*')
        
        if self.plan_or_goal == "plan":
            plan = self.desired_goal.reshape(2, self.plan_length)
            ax.plot(plan[0, :], plan[1, :])

        return ax

    
    def close(self):
        raise NotImplementedError

    def compute_reward(
        self,
        achieved_goal,
        desired_goal,
        info
    ):
        reward = np.array([
            info_element["is_success"] for info_element in info
        ])
        if self.plan_or_goal == "goal":
            return reward

        if self.plan_or_goal == "plan":
            mask = (reward==0.)
            reward[mask] = self._get_shaping_reward(
                achieved_goal[mask],
                desired_goal[mask]
            )

        return reward

    def _get_binary_reward(self, achieved_goal, current_target):
        """
        Get binary (sparse) goal reward
        """
        return np.linalg.norm(
            achieved_goal[:, :] - current_target[:, :],
            axis=-1
        ) < 0.05

    def _get_shaping_reward(self, achieved_goal, desired_goal):
        plan = desired_goal.reshape(-1, 2, self.plan_length)
        dists = np.linalg.norm(
            achieved_goal[:, :, None] - plan[:, :, :],
            axis=1
        )
        exponential_dists = np.exp(
            -dists**2/2/0.05**2
        )
        # calculate time of smallest (exp.) distance for each sample
        ind_smallest_dist = np.argmax(exponential_dists, axis=-1)
        return 0.5 * exponential_dists[
            np.arange(len(exponential_dists)), ind_smallest_dist
        ] * (ind_smallest_dist/self.plan_length + 0.9)

    def _sample_non_colliding_state(self):
        """
        Get a non-colliding state from state space
        """
        while True:
            candidate = self.observation_space["observation"].sample()
            if self._not_in_collision(candidate):
                return candidate
    
    def _not_in_collision(self, state):
        """
        Return boolean that is true if state is not in collision, and false if it is
        """
        no_collision = True

        for box in self.boxes:
            # check if state is in between (not necessarily box[0]>box[2] etc.)
            if (
                (state[0] - box[0]) * (state[0] - box[2]) <= 0 # >0 if outside
                and (state[1] - box[1]) * (state[1] - box[3]) <= 0
            ):
                no_collision = False
                break # save testing for the other boxes
        
        return no_collision

    def _sample_feasible_plan(self, state, goal):
        """
        Sample a feasible plan leading from state to goal
        """
        print("CAUTION: THIS IS ONLY A TEST IMPLEMENTATION FOR NOW")
        # raise NotImplementedError
        return (
            state[:, None] + (goal-state)[:, None]*np.linspace(0, 1, self.plan_length)[None, :]
        ).reshape(-1)
    