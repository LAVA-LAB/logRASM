from functools import partial

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
from gymnasium import spaces
from jax import jit

from core.commons import RectangularSet, MultiRectangularSet
from models.base_class import BaseEnvironment


def angle_normalize(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi


class MountainCar(BaseEnvironment, gym.Env):
    metadata = {
        "render_modes": [],
        "render_fps": 30,
    }

    def __init__(self, args=False):

        self.variable_names = ['position', 'velocity']

        # Time discretization step
        self.delta = 10

        self.min_action = np.array([-1.0])
        self.max_action = np.array([1.0])
        self.min_position = -1.2 - 0.2
        self.max_position = 0.6
        self.v_scaling = 45  # Scaled from the original, which is 0.07
        self.max_speed = 0.07 * self.v_scaling
        self.goal_position = (
            0.45  # was 0.5 in gym, 0.45 in Arnaud de Broissia's version
        )
        self.power = 0.0015
        self.constant1 = 0.0025
        self.friction = 1

        self.low_state = np.array(
            [self.min_position, -self.max_speed], dtype=np.float32
        )
        self.high_state = np.array(
            [self.max_position, self.max_speed], dtype=np.float32
        )

        self.state_dim = len(self.low_state)
        self.plot_dim = [0, 1]

        # Entries of Jacobian; row 1
        self.dpdp = 1 + self.constant1 * self.delta ** 2 * 3 * np.sin(0.5 * np.pi)  # max[sin(3*p_k)] = 1
        self.dpdv = self.delta * self.friction / self.v_scaling
        self.dpdu = self.delta ** 2 * self.power

        # Entries of Jacobian; row 2
        self.dvdp = self.constant1 * self.delta * 3 * self.v_scaling  # max[sin(3*p_k)] = 1
        self.dvdv = self.friction
        self.dvdu = self.delta * self.power * self.v_scaling

        self.lipschitz_f_l1_A = max(np.abs(self.dpdp) + np.abs(self.dvdp), np.abs(self.dpdv) + np.abs(self.dvdv))
        self.lipschitz_f_l1_B = np.abs(self.dpdu) + np.abs(self.dvdu)
        self.lipschitz_f_l1 = max(self.lipschitz_f_l1_A, self.lipschitz_f_l1_B)

        # This will throw a warning in tests/envs/test_envs in utils/env_checker.py as the space is not symmetric
        #   or normalised as max_torque == 2 by default. Ignoring the issue here as the default settings are too old
        #   to update to follow the openai gym api
        self.action_space = spaces.Box(
            low=self.min_action, high=self.max_action, shape=(1,), dtype=np.float32
        )

        # Set observation / state space
        self.state_space = RectangularSet(low=self.low_state, high=self.high_state, dtype=np.float32)

        # Set support of noise distribution (which is triangular, zero-centered)
        high = np.array([1], dtype=np.float32)
        self.noise_space = spaces.Box(low=-high, high=high, dtype=np.float32)
        self.noise_dim = len(high)

        # Set target set
        self.target_space = RectangularSet(low=np.array([self.goal_position, -self.max_speed - 0.1]),
                                           high=np.array([self.max_position + 0.1, self.max_speed + 0.1]),
                                           dtype=np.float32)

        self.min_position_init = -0.6
        self.max_position_init = -0.4
        self.init_space = RectangularSet(low=np.array([self.min_position_init, -0.1]),
                                         high=np.array([self.max_position_init, 0.1]), dtype=np.float32)

        self.min_position_unsafe = -1.4
        self.max_position_unsafe = -1.2
        self.unsafe_space = MultiRectangularSet([
            RectangularSet(low=np.array([self.min_position_unsafe, -self.max_speed]), high=np.array([self.max_position_unsafe, self.max_speed]),
                           dtype=np.float32),
        ])

        self.init_unsafe_dist = np.abs(- self.max_position_unsafe + self.min_position_init)

        self.num_steps_until_reset = 100

        # Vectorized step, but only with different noise values
        self.vstep_noise_set = jax.vmap(self.step_noise_set, in_axes=(None, None, 0, 0), out_axes=(0, 0))

        # Set to reset to in training (typically the initial state set, or the whole state space)
        self.reset_space = self.state_space  # self.init_space

        super(MountainCar, self).__init__()

    def step(self, u):
        '''
        Step in the gymnasium environment (only used for policy initialization with StableBaselines3).
        '''

        assert self.state is not None, "Call reset before using step method."

        force = np.clip(u, self.min_action, self.max_action)
        w = self.sample_triangular_noise_numpy()

        position = self.state[0]
        velocity = self.state[1]

        velocity = self.friction * velocity + self.delta * (force[0] * self.power - self.constant1 * np.cos(3 * position)) * self.v_scaling + 0.01 * w[0]
        velocity = np.clip(velocity, -self.max_speed, self.max_speed)
        position += self.delta * velocity / self.v_scaling
        position = np.clip(position, self.min_position, self.max_position)

        # Put together state
        self.state = np.array([position, velocity])

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

    @partial(jit, static_argnums=(0,))
    def step_base(self, state, u, w):
        '''
        Make a step in the dynamics. When defining a new environment, this the function that should be modified.
        '''

        force = jnp.clip(u, self.min_action, self.max_action)

        position = state[0]
        velocity = state[1]

        velocity = self.friction * velocity + self.delta * (force[0] * self.power - self.constant1 * jnp.cos(3 * position)) * self.v_scaling + 0.01 * w[0]
        velocity = jnp.clip(velocity, -self.max_speed, self.max_speed)
        position += self.delta * velocity / self.v_scaling
        position = jnp.clip(position, self.min_position, self.max_position)

        # Put together state
        state = jnp.array([position, velocity])

        return state

    @partial(jit, static_argnums=(0,))
    def step_noise_set(self, state, u, w_lb, w_ub):
        ''' Make step with dynamics for a set of noise values.
        Propagate state under lower/upper bound of the noise (note: this works because the noise is additive) '''

        # Propogate dynamics for both the lower bound and upper bound of the noise
        # (note: this works because the noise is additive)
        state_lb = self.step_base(state, u, w_lb)
        state_ub = self.step_base(state, u, w_ub)

        # Compute the mean and the epsilon (difference between mean and ub/lb)
        state_mean = (state_ub + state_lb) / 2
        epsilon = (state_ub - state_lb) / 2

        return state_mean, epsilon

    def integrate_noise(self, w_lb, w_ub):
        prob_lb, prob_ub = self.integrate_noise_triangular(w_lb, w_ub)

        return prob_lb, prob_ub

    @partial(jit, static_argnums=(0,))
    def step_train(self, state, key, u, steps_since_reset):

        # Split RNG key
        key, subkey = jax.random.split(key)

        # Sample noise value
        noise = self.sample_triangular_noise_jax(subkey)

        force = jnp.clip(u, self.min_action, self.max_action)

        goal_reached = self.target_space.jax_contains(jnp.array([state]))[0]
        fail = self.unsafe_space.jax_contains(jnp.array([state]))[0]
        # costs = 0.1 * (force[0] ** 2) - 100 * goal_reached + 100 * fail
        costs = 1 - 100 * goal_reached + 100 * fail

        # Propagate dynamics
        state = self.step_base(state, u, noise)

        steps_since_reset += 1

        terminated = False  # fail
        truncated = (steps_since_reset >= self.num_steps_until_reset)
        done = terminated | truncated
        state, key, steps_since_reset = self._maybe_reset(state, key, steps_since_reset, done)

        return state, key, steps_since_reset, -costs, terminated, truncated, {}
