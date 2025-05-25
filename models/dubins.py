from functools import partial

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
from gymnasium import spaces
from jax import jit

from core.commons import RectangularSet, MultiRectangularSet
from models.base_class import BaseEnvironment


class Dubins(BaseEnvironment, gym.Env):
    def __init__(self, args=False):

        # TODO: Make our algorithm correct when wrapping around for steering angle
        self.THETA_WRAPPING = True

        self.variable_names = ['x', 'y', 'ang.vel.']
        self.plot_dim = [0, 1]

        self.min_torque = np.array([-1, -1], dtype=np.float32)
        self.max_torque = np.array([1, 1], dtype=np.float32)

        self.delta = 0.1

        # This will throw a warning in tests/envs/test_envs in utils/env_checker.py as the space is not symmetric
        #   or normalised as max_torque == 2 by default. Ignoring the issue here as the default settings are too old
        #   to update to follow the openai gym api
        self.action_space = spaces.Box(
            low=self.min_torque, high=self.max_torque, shape=(len(self.min_torque),), dtype=np.float32
        )

        self.angle_min = -1
        self.angle_max = 1
        self.full_circle = 2

        maxabsu1 = max(abs(self.min_torque[1]), abs(self.max_torque[1]))
        self.lipschitz_f_l1_A = 1 + np.sqrt(2) * (2 * np.pi) / self.full_circle * maxabsu1 * self.delta * 10  # 3.22
        self.lipschitz_f_l1_B = max(np.sqrt(2) * self.delta * 10, np.sqrt(2) * (
                2 * np.pi) / self.full_circle * maxabsu1 * self.delta ** 2 * 50)  # 1.41
        self.lipschitz_f_l1 = max(self.lipschitz_f_l1_A, self.lipschitz_f_l1_B)

        if args.layout == 1:
            print('- Use easy layout for Dubins')
            self.x_min = -0.5
            self.x_max = 0.5
            self.y_min = -0.5
            self.y_max = 0.5

            # Set target set
            self.target_space = RectangularSet(low=np.array([-0.3, 0.1, self.angle_min]),
                                               high=np.array([-0.1, 0.3, self.angle_max]),
                                               fix_dimensions=[2], dtype=np.float32)

            self.init_space = RectangularSet(low=np.array([0.1, -0.3, 0.35 * self.full_circle]),
                                             high=np.array([0.3, -0.1, 0.4 * self.full_circle]),
                                             fix_dimensions=[2], dtype=np.float32)

            self.init_unsafe_dist = 0.6

        elif args.layout == 2:
            print('- Use easy layout for Dubins')
            self.x_min = -0.5
            self.x_max = 0.5
            self.y_min = -0.5
            self.y_max = 0.5

            # Set target set
            self.target_space = RectangularSet(low=np.array([-0.3, -0.3, self.angle_min]),
                                               high=np.array([-0.05, -0.05, self.angle_max]),
                                               fix_dimensions=[2], dtype=np.float32)

            self.init_space = RectangularSet(low=np.array([0.05, 0.05, 0.4 * self.full_circle]),
                                             high=np.array([0.3, 0.3, 0.5 * self.full_circle]),
                                             fix_dimensions=[2], dtype=np.float32)

            self.init_unsafe_dist = 0.6

        else:
            print(' Use standard layout for Dubins')
            self.x_min = -1
            self.x_max = 2
            self.y_min = -2
            self.y_max = 1

            # Set target set
            self.target_space = RectangularSet(low=np.array([-0.3, -0.3, self.angle_min]),
                                               high=np.array([0.3, 0.3, self.angle_max]),
                                               fix_dimensions=[2], dtype=np.float32)

            self.init_space = RectangularSet(low=np.array([0.8, -1.2, 0.7 * self.full_circle]),
                                             high=np.array([1.2, -0.8, 0.8 * self.full_circle]),
                                             fix_dimensions=[2], dtype=np.float32)

            self.init_unsafe_dist = 0.7

        # Set observation / state space
        low = np.array([self.x_min, self.y_min, self.angle_min], dtype=np.float32)
        high = np.array([self.x_max, self.y_max, self.angle_max], dtype=np.float32)
        self.state_space = RectangularSet(low=low, high=high, fix_dimensions=[2], dtype=np.float32)

        # Set support of noise distribution (which is triangular, zero-centered)
        # TODO enable noise for Dubins again (currently negligible)
        high = np.array([0.0001], dtype=np.float32)
        self.noise_space = spaces.Box(low=-high, high=high, dtype=np.float32)

        self.unsafe_space = MultiRectangularSet([
            # RectangularSet(low=np.array([self.x_min, self.y_min, self.angle_min]),
            #                high=np.array([self.x_min + 0.1, self.y_min + 0.1, self.angle_max]), fix_dimensions=[2],
            #                dtype=np.float32),
            RectangularSet(low=np.array([self.x_min, self.y_min, self.angle_min]),
                           high=np.array([self.x_min + 0.1, self.y_max, self.angle_max]), fix_dimensions=[2],
                           dtype=np.float32),
            RectangularSet(low=np.array([self.x_max - 0.1, self.y_min, self.angle_min]),
                           high=np.array([self.x_max, self.y_max, self.angle_max]), fix_dimensions=[2],
                           dtype=np.float32),
            RectangularSet(low=np.array([self.x_min, self.y_min, self.angle_min]),
                           high=np.array([self.x_max, self.y_min + 0.1, self.angle_max]), fix_dimensions=[2],
                           dtype=np.float32),
            RectangularSet(low=np.array([self.x_min, self.y_max - 0.1, self.angle_min]),
                           high=np.array([self.x_max, self.y_max, self.angle_max]), fix_dimensions=[2],
                           dtype=np.float32),
            # RectangularSet(low=np.array([self.x_min, self.y_min, self.angle_min]), high=np.array([self.x_max, self.y_max, self.angle_min+0.1]), dtype=np.float32),
            # RectangularSet(low=np.array([self.x_min, self.y_min, self.angle_max - 0.1]), high=np.array([self.x_max, self.y_max, self.angle_max]), dtype=np.float32),
            #
            # RectangularSet(low=np.array([self.x_min, self.y_min, self.angle_min]), high=np.array([self.x_min + 0.1, self.y_min + 0.1, self.angle_max]), dtype=np.float32),
            # RectangularSet(low=np.array([self.x_max - 0.1, self.y_max - 0.1, self.angle_min]), high=np.array([self.x_max, self.y_max, self.angle_max]), dtype=np.float32),
        ])

        self.num_steps_until_reset = 100

        # Set to reset to in training (typically the initial state set, or the whole state space)
        self.reset_space = self.state_space

        super(Dubins, self).__init__()

    @partial(jit, static_argnums=(0,))
    def step_base(self, state, u, w):
        '''
        Make a step in the dynamics. When defining a new environment, this the function that should be modified.
        '''

        u = jnp.clip(u, self.min_torque, self.max_torque)

        if self.THETA_WRAPPING:
            theta_range = (self.angle_max - self.angle_min)
            theta_next = (state[2] + self.delta * 5 * (u[0] + w[0]) + (-self.angle_min)) % theta_range - (
                -self.angle_min)
        else:
            theta_next = jnp.clip(state[2] + self.delta * 5 * (u[0] + w[0]),
                                  self.angle_min, self.angle_max)
        x = state[0] + self.delta * 10 * u[1] * jnp.cos(theta_next * (2 * jnp.pi) / self.full_circle)
        y = state[1] + self.delta * 10 * u[1] * jnp.sin(theta_next * (2 * jnp.pi) / self.full_circle)

        # Lower bound state
        state = jnp.array([x, y, theta_next])
        state = jnp.clip(state, self.state_space.low, self.state_space.high)

        return state

    @partial(jit, static_argnums=(0,))
    def step_noise_set(self, state, u, w_lb, w_ub):
        ''' Make step with dynamics for a set of noise values '''

        state_mean, epsilon = self.propagate_additive_noise_box(state, u, w_lb, w_ub)

        return state_mean, epsilon

    @partial(jit, static_argnums=(0,))
    def step_train(self, state, key, u, steps_since_reset):

        # Split RNG key
        key, subkey = jax.random.split(key)

        # Sample noise value
        noise = self.sample_triangular_noise_jax(subkey)

        goal_reached = self.target_space.jax_contains(jnp.array([state]))[0]
        fail = self.unsafe_space.jax_contains(jnp.array([state]))[0]
        bonus = 50 * goal_reached - 5 * fail
        costs = - 0 * 1 - bonus + jnp.sqrt((state[0] - self.target_space.center[0]) ** 2 +
                                           (state[1] - self.target_space.center[1]) ** 2)

        # Propagate dynamics
        state = self.step_base(state, u, noise)

        steps_since_reset += 1

        terminated = goal_reached + fail
        truncated = (steps_since_reset >= self.num_steps_until_reset)
        done = terminated | truncated
        state, key, steps_since_reset = self._maybe_reset(state, key, steps_since_reset, done)

        return state, key, steps_since_reset, -costs, terminated, truncated, {}

    def integrate_noise(self, w_lb, w_ub):
        prob_lb, prob_ub = self.integrate_noise_triangular(w_lb, w_ub)

        return prob_lb, prob_ub

    def step(self, u):
        '''
        Step in the gymnasium environment (only used for policy initialization with PPO).
        '''

        assert self.state is not None, "Call reset before using step method."

        u = jnp.clip(u, self.min_torque, self.max_torque)
        w = self.sample_triangular_noise_numpy()

        if self.THETA_WRAPPING:
            theta_range = (self.angle_max - self.angle_min)
            theta_next = (self.state[2] + self.delta * 5 * (u[0] + w[0]) + (-self.angle_min)) % theta_range - (
                -self.angle_min)
        else:
            theta_next = jnp.clip(self.state[2] + self.delta * 5 * (u[0] + w[0]),
                                  self.angle_min, self.angle_max)
        x = self.state[0] + self.delta * 10 * u[1] * np.cos(theta_next * (2 * np.pi) / self.full_circle)
        y = self.state[1] + self.delta * 10 * u[1] * np.sin(theta_next * (2 * np.pi) / self.full_circle)

        # Lower/upper bound state
        self.state = jnp.array([x, y, theta_next])
        self.state = jnp.clip(self.state, self.state_space.low, self.state_space.high)

        self.last_u = u  # for rendering

        fail = bool(
            self.unsafe_space.contains(np.array([self.state]), return_indices=True) +
            self.state_space.not_contains(np.array([self.state]), return_indices=True)
        )
        goal_reached = np.all(self.state >= self.target_space.low) * np.all(self.state <= self.target_space.high)
        terminated = fail

        if fail:
            costs = 5
        elif goal_reached:
            costs = -5
        else:
            costs = -1 + np.sqrt((self.state[0] - self.target_space.center[0]) ** 2 +
                                 (self.state[1] - self.target_space.center[1]) ** 2)

        return np.array(self.state, dtype=np.float32), -costs, terminated, False, {}
