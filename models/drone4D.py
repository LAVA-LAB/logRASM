from functools import partial

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
from gymnasium import spaces
from jax import jit

from core.commons import RectangularSet, MultiRectangularSet
from models.base_class import BaseEnvironment, compute_lipschitz_jacobian


class Drone4D(BaseEnvironment, gym.Env):
    def __init__(self, args=False):

        self.variable_names = ['x1', 'v1', 'x2', 'v2']
        self.plot_dim = [0, 2]

        self.max_torque = np.array([0.5, 0.5], dtype=np.float32)
        self.tau = 0.5

        # Nonlinear damping constants
        self.damping1 = 0.02
        self.damping2 = 0.01

        # Wind force
        self.wind = -0.1

        self.W = np.array([0.01, 0.01])

        # Set observation / state space
        low = np.array([-0.5, -0.5, -0.5, -0.5], dtype=np.float32)
        high = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
        self.state_space = RectangularSet(low=low, high=high, dtype=np.float32)

        # This will throw a warning in tests/envs/test_envs in utils/env_checker.py as the space is not symmetric
        #   or normalised as max_torque == 2 by default. Ignoring the issue here as the default settings are too old
        #   to update to follow the openai gym api
        self.action_space = spaces.Box(
            low=-self.max_torque, high=self.max_torque, shape=(len(self.max_torque),), dtype=np.float32
        )

        # Set support of noise distribution (which is triangular, zero-centered)
        high = np.array([1, 1], dtype=np.float32)
        self.noise_space = spaces.Box(low=-high, high=high, dtype=np.float32)

        # Set target set
        self.target_space = RectangularSet(low=np.array([0.3, -0.5, 0.3, -0.5]), high=np.array([0.6, 0.5, 0.6, 0.5]), fix_dimensions=[1, 3], dtype=np.float32)

        # Set initial state set
        self.init_space = RectangularSet(low=np.array([-0.45, -0.1, -0.45, 0.25]), high=np.array([-0.35, 0.1, -0.35, 0.35]), fix_dimensions=[1, 3], dtype=np.float32)

        # Set unsafe state set
        self.unsafe_space = MultiRectangularSet([
            RectangularSet(low=np.array([0.2, -0.5, -0.5, -0.5]), high=np.array([0.5, 0.5, -0.3, 0.5]), fix_dimensions=[1, 3], dtype=np.float32),
            RectangularSet(low=np.array([0.0, -0.5, -0.5, -0.5]), high=np.array([0.2, 0.5, -0.1, 0.5]), fix_dimensions=[1, 3], dtype=np.float32),
            RectangularSet(low=np.array([-0.5, -0.5, 0.4, -0.5]), high=np.array([0, 0.5, 0.5, 0.5]), fix_dimensions=[1, 3], dtype=np.float32)
        ])

        self.init_unsafe_dist = 0.35

        # Compute all partial derivatives
        dx1dx1 = 1
        dx1dv1 = self.tau
        dx1du0 = self.tau ** 2 / 2
        dv1dx1 = 0
        dv1dv1 = 1 + self.tau * (-self.damping1 * 2 * max(np.abs(self.state_space.low[1]), self.state_space.high[1]) ** 2)
        dv1du0 = self.tau
        dx2dx2 = 1
        dx2dv2 = self.tau
        dx2du1 = self.tau ** 2 / 2
        dv2dx2 = 0
        dv2dv2 = 1 + self.tau * (-self.damping2 * 2 * max(np.abs(self.state_space.low[3]), self.state_space.high[3]) ** 2)
        dv2dx1 = self.tau * self.wind * np.pi
        dv2du1 = self.tau

        # Set Jacobian
        J = np.array([[dx1dx1, dx1dv1, 0, 0],
                      [dv1dx1, dv1dv1, 0, 0],
                      [0, 0, dx2dx2, dx2dv2],
                      [dv2dx1, 0, dv2dx2, dv2dv2]])
        G = np.array([[dx1du0, 0],
                      [dv1du0, 0],
                      [0, dx2du1],
                      [0, dv2du1]])

        # Compute Lipschitz constants from the Jacobians J and G
        self.lipschitz_f_l1, _, self.lipschitz_f_l1_A, _, self.lipschitz_f_l1_B, _ = compute_lipschitz_jacobian(J, G)

        self.num_steps_until_reset = 100

        # Set to reset to in training (typically the initial state set, or the whole state space)
        self.reset_space = self.state_space

        super(Drone4D, self).__init__()

    @partial(jit, static_argnums=(0,))
    def step_base(self, state, u, w):
        '''
        Make a step in the dynamics. When defining a new environment, this the function that should be modified.
        '''

        u = jnp.clip(u, -self.max_torque, self.max_torque)
        x1, v1, x2, v2 = state

        # Dynamics (first x2 then x1)
        x2 = x2 + self.tau * v2 + self.tau ** 2 / 2 * u[1]
        v2 = v2 + self.tau * (-self.damping2 * v2 ** 3 + u[1] + self.wind * jnp.sin(jnp.pi * x1)) + self.W[1] * w[1]
        x1 = x1 + self.tau * v1 + self.tau ** 2 / 2 * u[0]
        v1 = v1 + self.tau * (-self.damping1 * v1 ** 3 + u[0]) + self.W[0] * w[0]

        # Add noise and clip
        state = jnp.array([x1, v1, x2, v2])
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
        costs = jnp.sqrt((state[0] - self.target_space.center[0]) ** 2 + (state[2] - self.target_space.center[2]) ** 2) \
                + 10 * fail - 10 * goal_reached

        # Propagate dynamics
        state = self.step_base(state, u, noise)

        steps_since_reset += 1

        terminated = fail
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

        x1, v1, x2, v2 = self.state

        # Dynamics (first x2 then x1)
        x2 = x2 + self.tau * v2 + self.tau ** 2 / 2 * u[1]
        v2 = v2 + self.tau * (-self.damping2 * v2 ** 3 + u[1] + self.wind * np.sin(np.pi * x1)) + self.W[1] * w[1]
        x1 = x1 + self.tau * v1 + self.tau ** 2 / 2 * u[0]
        v1 = v1 + self.tau * (-self.damping1 * v1 ** 3 + u[0]) + self.W[0] * w[0]

        # Add noise and clip
        self.state = np.array([x1, v1, x2, v2])
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
