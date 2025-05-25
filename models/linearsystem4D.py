from functools import partial

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
from gymnasium import spaces
from jax import jit

from core.commons import RectangularSet, MultiRectangularSet
from models.base_class import BaseEnvironment, compute_lipschitz_linear


class LinearSystem4D(BaseEnvironment, gym.Env):
    def __init__(self, args=False):

        self.variable_names = ['x1', 'v1', 'x2', 'v2']
        self.plot_dim = [0, 2]

        self.max_torque = np.array([0.5, 0.5])

        self.A = np.array([
            [1, 0.045, 0, 0],
            [0, 0.9, 0, 0],
            [0, 0, 1, 0.045],
            [0, 0, 0, 0.9],
        ])
        self.B = np.array([
            [0.45, 0],
            [0.5, 0],
            [0, 0.45],
            [0, 0.5],
        ])
        self.W = np.diag([0.01, 0.005, 0.01, 0.005])

        # Compute Lipschitz constants (for linear dynamics)
        self.lipschitz_f_l1, self.lipschitz_f_linfty, self.lipschitz_f_l1_A, self.lipschitz_f_linfty_A, self.lipschitz_f_l1_B, self.lipschitz_f_linfty_B = \
            compute_lipschitz_linear(self.A, self.B)

        # This will throw a warning in tests/envs/test_envs in utils/env_checker.py as the space is not symmetric
        #   or normalised as max_torque == 2 by default. Ignoring the issue here as the default settings are too old
        #   to update to follow the openai gym api
        self.action_space = spaces.Box(
            low=-self.max_torque, high=self.max_torque, shape=(len(self.max_torque),), dtype=np.float32
        )

        # Set support of noise distribution (which is triangular, zero-centered)
        high = np.array([1, 1, 1, 1], dtype=np.float32)
        self.noise_space = spaces.Box(low=-high, high=high, dtype=np.float32)

        if args and args.layout == 1:
            print('- Use layout from DynAbs (abstraction-based controller synthesis)')

            # Set observation / state space
            low = np.array([-1, -0.5, -1, -0.5], dtype=np.float32)
            high = np.array([0, 0.5, 1, 0.5], dtype=np.float32)
            self.state_space = RectangularSet(low=low, high=high, dtype=np.float32)

            # Set target state set
            self.target_space = RectangularSet(low=np.array([-0.75, -0.5, 0.5, -0.5]), high=np.array([-0.5, 0.5, 0.75, 0.5]), fix_dimensions=[1, 3], dtype=np.float32)

            # Set initial state set
            self.init_space = RectangularSet(low=np.array([-0.9, -0.1, -0.9, -0.1]), high=np.array([-0.8, 0.1, -0.8, 0.1]), fix_dimensions=[1, 3], dtype=np.float32)

            # Set unsafe state set
            self.unsafe_space = RectangularSet(low=np.array([-1, -0.5, -0.25, -0.5]), high=np.array([-0.5, 0.5, 0, 0.5]), fix_dimensions=[1, 3], dtype=np.float32)

            self.init_unsafe_dist = 0.55

        else:
            print('- Use standard layout')

            # Set observation / state space
            high = np.array([1.5, 1.5, 1.5, 1.5], dtype=np.float32)
            self.state_space = RectangularSet(low=-high, high=high, dtype=np.float32)

            # Set target state set
            self.target_space = RectangularSet(low=np.array([-0.2, -0.2, -0.2, -0.2]), high=np.array([0.2, 0.2, 0.2, 0.2]), dtype=np.float32)

            # Set initial state set
            self.init_space = MultiRectangularSet([
                RectangularSet(low=np.array([-0.25, -0.1, -0.25, -0.1]), high=np.array([-0.20, 0.1, -0.20, 0.1]), dtype=np.float32),
                RectangularSet(low=np.array([0.20, -0.1, 0.20, -0.1]), high=np.array([0.25, 0.1, 0.25, 0.1]), dtype=np.float32)
            ])

            # Set unsafe state set
            self.unsafe_space = MultiRectangularSet([
                RectangularSet(low=np.array([-1.5, -1.5, -1.5, -1.5]), high=np.array([-1.4, 0, -1.4, 0]), dtype=np.float32),
                RectangularSet(low=np.array([1.4, 0, 1.4, 0]), high=np.array([1.5, 1.5, 1.5, 1.5]), dtype=np.float32)
            ])

            self.init_unsafe_dist = 1.15 + 1.15

        self.num_steps_until_reset = 100

        # Set to reset to in training (typically the initial state set, or the whole state space)
        self.reset_space = self.state_space

        super(LinearSystem4D, self).__init__()

    @partial(jit, static_argnums=(0,))
    def step_base(self, state, u, w):
        '''
        Make a step in the dynamics. When defining a new environment, this the function that should be modified.
        '''

        u = jnp.clip(u, -self.max_torque, self.max_torque)
        state = jnp.matmul(self.A, state) + jnp.matmul(self.B, u) + jnp.matmul(self.W, w)
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
        fail = self.unsafe_space.jax_contains(jnp.array([state]))[0] + \
               self.state_space.jax_not_contains(jnp.array([state]))[0]
        costs = 10 * (state[0] - self.target_space.center[0]) ** 2 + (state[1] - self.target_space.center[1]) ** 2 + \
                10 * (state[2] - self.target_space.center[2]) ** 2 + (state[3] - self.target_space.center[3]) ** 2 + 10 * fail

        # Propagate dynamics
        state = self.step_base(state, u, noise)

        steps_since_reset += 1

        terminated = False  # fail
        truncated = (steps_since_reset >= self.num_steps_until_reset)
        done = terminated | truncated
        state, key, steps_since_reset = self._maybe_reset(state, key, steps_since_reset, done)

        return state, key, steps_since_reset, -costs, terminated, truncated, {}

    def integrate_noise(self, w_lb, w_ub):
        prob_lb, prob_ub = self.integrate_noise_triangular(w_lb, w_ub)

        return prob_lb, prob_ub

    def step(self, u):
        '''
        Step in the gymnasium environment (only used for policy initialization with StableBaselines3).
        '''

        assert self.state is not None, "Call reset before using step method."

        u = np.clip(u, -self.max_torque, self.max_torque)
        w = self.sample_triangular_noise_numpy()
        self.state = self.A @ self.state + self.B @ u + self.W @ w
        self.state = np.clip(self.state, self.state_space.low, self.state_space.high)
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
            costs = -1 + np.sqrt((self.state[0] - self.target_space.center[0]) ** 2 + (self.state[2] - self.target_space.center[2]) ** 2)

        return np.array(self.state, dtype=np.float32), -costs, terminated, False, {}
