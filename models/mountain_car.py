from functools import partial

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
from gymnasium import spaces
from jax import jit

from core.commons import RectangularSet, MultiRectangularSet
from models.base_class import BaseEnvironment, compute_lipschitz_jacobian


# python run.py --model MyMountainCar --probability_bound 0.9999 --pretrain_method PPO_JAX --pretrain_total_steps 10000000 --mesh_loss 0.0001 --exp_certificate --plot_intermediate --mesh_verify_grid_init 0.001 --expDecr_multiplier 10 --epochs 100 --load_ckpt 'ckpt/ppo_jax_MyMountainCar_seed=1_2025-05-23_14-00-47' --eps_decrease 0.01 --normalize_loss

def angle_normalize(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi


class MountainCar(BaseEnvironment, gym.Env):
    def __init__(self, args=False):

        self.variable_names = ['position', 'velocity']
        self.plot_dim = [0, 1]

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

        # Entries of Jacobian; row 1
        dpdp = 1 + self.constant1 * self.delta ** 2 * 3 * np.sin(0.5 * np.pi)  # max[sin(3*p_k)] = 1
        dpdv = self.delta * self.friction / self.v_scaling
        dpdu = self.delta ** 2 * self.power

        # Entries of Jacobian; row 2
        dvdp = self.constant1 * self.delta * 3 * self.v_scaling  # max[sin(3*p_k)] = 1
        dvdv = self.friction
        dvdu = self.delta * self.power * self.v_scaling

        # Set Jacobian
        J = np.array([[dpdp, dpdv],
                      [dvdp, dvdv]])
        G = np.array([[dpdu],
                      [dvdu]])

        # Compute Lipschitz constants from the Jacobians J and G
        self.lipschitz_f_l1, _, self.lipschitz_f_l1_A, _, self.lipschitz_f_l1_B, _ = compute_lipschitz_jacobian(J, G)

        # Set observation / state space
        self.state_space = RectangularSet(low=self.low_state, high=self.high_state, dtype=np.float32)

        # This will throw a warning in tests/envs/test_envs in utils/env_checker.py as the space is not symmetric
        #   or normalised as max_torque == 2 by default. Ignoring the issue here as the default settings are too old
        #   to update to follow the openai gym api
        self.action_space = spaces.Box(
            low=self.min_action, high=self.max_action, shape=(1,), dtype=np.float32
        )

        # Set support of noise distribution (which is triangular, zero-centered)
        high = np.array([1], dtype=np.float32)
        self.noise_space = spaces.Box(low=-high, high=high, dtype=np.float32)

        # Set target set
        self.target_space = RectangularSet(low=np.array([self.goal_position, -self.max_speed - 0.1]),
                                           high=np.array([self.max_position + 0.1, self.max_speed + 0.1]),
                                           dtype=np.float32)

        # Set initial state set
        self.min_position_init = -0.6
        self.max_position_init = -0.4
        self.init_space = RectangularSet(low=np.array([self.min_position_init, -0.1]),
                                         high=np.array([self.max_position_init, 0.1]), dtype=np.float32)

        # Set unsafe state set
        self.min_position_unsafe = -1.4
        self.max_position_unsafe = -1.2
        self.unsafe_space = MultiRectangularSet([
            RectangularSet(low=np.array([self.min_position_unsafe, -self.max_speed]), high=np.array([self.max_position_unsafe, self.max_speed]),
                           dtype=np.float32),
        ])

        self.init_unsafe_dist = np.abs(- self.max_position_unsafe + self.min_position_init)

        self.num_steps_until_reset = 100

        # Set to reset to in training (typically the initial state set, or the whole state space)
        self.reset_space = self.state_space  # self.init_space

        super(MountainCar, self).__init__()

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
        ''' Make step with dynamics for a set of noise values '''

        state_mean, epsilon = self.propagate_additive_noise_box(state, u, w_lb, w_ub)

        return state_mean, epsilon

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

    def integrate_noise(self, w_lb, w_ub):
        prob_lb, prob_ub = self.integrate_noise_triangular(w_lb, w_ub)

        return prob_lb, prob_ub

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
