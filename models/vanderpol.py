from functools import partial

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
from gymnasium import spaces
from jax import jit

from core.commons import RectangularSet, MultiRectangularSet
from models.base_class import BaseEnvironment


# python run.py --model VanDerPol --probability_bound 0.9 --pretrain_method PPO_JAX --pretrain_total_steps 1000000 --mesh_loss 0.0001 --exp_certificate --plot_intermediate --mesh_verify_grid_init 0.01 --expDecr_multiplier 10 --epochs 100 --eps_decrease 0.01


class Vanderpol(BaseEnvironment, gym.Env):
    def __init__(self, args=False):

        self.variable_names = ['position', 'velocity']
        self.plot_dim = [0, 1]

        self.max_force = np.array([1])

        # Pendulum parameters
        self.delta = 0.1

        # TODO: Correct Lipschitz computation
        self.lipschitz_f_l1_A = 8
        self.lipschitz_f_l1_B = 1
        self.lipschitz_f_l1 = max(self.lipschitz_f_l1_A, self.lipschitz_f_l1_B)

        # Set observation / state space
        high = np.array([5, 5], dtype=np.float32)
        self.state_space = RectangularSet(low=-high, high=high, dtype=np.float32)

        # This will throw a warning in tests/envs/test_envs in utils/env_checker.py as the space is not symmetric
        #   or normalised as max_force == 2 by default. Ignoring the issue here as the default settings are too old
        #   to update to follow the openai gym api
        self.action_space = spaces.Box(
            low=-self.max_force, high=self.max_force, shape=(len(self.max_force),), dtype=np.float32
        )

        # Set support of noise distribution (which is triangular, zero-centered)
        high = np.array([0.1], dtype=np.float32)
        self.noise_space = spaces.Box(low=-high, high=high, dtype=np.float32)

        # Set target set
        # self.target_space = RectangularSet(low=np.array([-1.4, -2.9]),
        #                                    high=np.array([-0.7, -2.0]),
        #                                    dtype=np.float32)
        self.target_space = RectangularSet(low=np.array([-2.5, -1.5]),
                                           high=np.array([-1, 1]),
                                           dtype=np.float32)

        # Set initial state set
        self.init_space = RectangularSet(low=np.array([0.7, 2.0]), high=np.array([1.4, 2.9]), dtype=np.float32)
        # self.init_space = RectangularSet(low=np.array([-2, -2]), high=np.array([2, 2]), dtype=np.float32)

        # Set unsafe state set
        self.unsafe_space = MultiRectangularSet([
            RectangularSet(low=np.array([-5, -5]), high=np.array([-4, 5]), dtype=np.float32),
            RectangularSet(low=np.array([-5, 4]), high=np.array([5, 5]), dtype=np.float32),
            RectangularSet(low=np.array([-5, -5]), high=np.array([5, -4]), dtype=np.float32),
            RectangularSet(low=np.array([4, -5]), high=np.array([5, 5]), dtype=np.float32),
        ])

        self.init_unsafe_dist = 1.1

        self.num_steps_until_reset = 100

        # Set to reset to in training (typically the initial state set, or the whole state space)
        self.reset_space = self.state_space

        super(Vanderpol, self).__init__()

    @partial(jit, static_argnums=(0,))
    def step_base(self, state, u, w):
        '''
        Make a step in the dynamics. When defining a new environment, this the function that should be modified.
        '''

        u = jnp.clip(u, -self.max_force, self.max_force)

        x0 = state[0] + state[1] * self.delta + w[0]
        x1 = state[1] + (-state[0] + (1 - state[0] ** 2) * state[1]) * self.delta + u[0] + w[1]

        # Lower bound state
        state = jnp.clip(jnp.array([x0, x1]), self.state_space.low, self.state_space.high)

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
        costs = 1 - 100 * goal_reached + 100 * fail

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
        Step in the gymnasium environment (only used for policy initialization with PPO).
        '''

        assert self.state is not None, "Call reset before using step method."

        u = np.clip(u, -self.max_force, self.max_force)
        w = self.sample_triangular_noise_numpy()

        x0 = self.state[0] + self.state[1] * self.delta + w[0]
        x1 = self.state[1] + (-self.state[0] + (1 - self.state[0] ** 2) * self.state[1]) * self.delta + u + w[1]

        # Clip state
        self.state = np.clip(np.array([x0, x1]), self.state_space.low, self.state_space.high)

        fail = bool(
            self.unsafe_space.contains(np.array([self.state]), return_indices=True)
        )
        goal_reached = np.all(self.state >= self.target_space.low) * np.all(self.state <= self.target_space.high)
        terminated = fail

        if fail:
            costs = 100
        elif goal_reached:
            costs = -100
        else:
            costs = 1

        return np.array(self.state, dtype=np.float32), -costs, terminated, False, {}
