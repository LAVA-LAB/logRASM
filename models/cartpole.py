from functools import partial

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
from gymnasium import spaces
from jax import jit

from core.commons import RectangularSet, MultiRectangularSet
from models.base_class import BaseEnvironment


#  python run.py --model Cartpole --probability_bound 0.9 --pretrain_method PPO_JAX --pretrain_total_steps 1000000 --mesh_loss 0.0001 --exp_certificate --plot_intermediate --mesh_verify_grid_init 0.1 --expDecr_multiplier 10 --epochs 100 --eps_decrease 0.01
# python run.py --model Cartpole --probability_bound 0.9 --pretrain_method PPO --pretrain_total_steps 100000 --hidden_layers 3 --mesh_loss 0.001 --mesh_verify_grid_init 0.05 --refine_threshold 50000000 --verify_threshold 600000000 --noise_partition_cells 2 --max_refine_factor 2 --epochs 100
# python run.py --model Cartpole --probability_bound 0.9 --pretrain_method PPO_JAX --pretrain_total_steps 10000000 --hidden_layers 3 --mesh_loss 0.01 --mesh_verify_grid_init 0.03 --refine_threshold 50000000 --verify_threshold 600000000 --noise_partition_cells 2 --max_refine_factor 2 --epochs 100 --plot_intermediate --policy_patching
# python run.py --model Cartpole --probability_bound 0.9 --pretrain_method TRPO --pretrain_total_steps 1000000 --mesh_loss 0.01 --exp_certificate --certificate_hidden_layers 3 --certificate_neurons_per_layer 128 --policy_hidden_layers 2 --policy_neurons_per_layer 64 --mesh_verify_grid_init 0.03 --refine_threshold 50000000 --verify_threshold 600000000 --noise_partition_cells 2 --max_refine_factor 2 --epochs 100 --plot_intermediate --policy_patching --Policy_learning_rate 5e-7

class Cartpole(BaseEnvironment, gym.Env):
    def __init__(self, args=False):

        # NOTES:
        # - Try init_space as reset space
        # - Compare different RL algorithms
        # - Number of steps, 1e6 should be enough?
        # - For SB3: Give reward of 1 for hitting the target

        self.variable_names = ['position', 'velocity', 'angle', 'angular speed']

        # This multiplier can be used to make the step sizes slightly bigger
        tau_multiplier = 10

        ### Below are the original parameters of the Cartpole problem as the Gymnasium Python package
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masspole + self.masscart
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = self.masspole * self.length
        self.force_mag = 10.0
        self.tau = 0.02 * tau_multiplier  # seconds between state updates

        self.theta_unsafe_radians = 12 * (2 * np.pi / 360)  # Angle at which to fail
        self.theta_goal_radians = 1 * (2 * np.pi / 360)  # Goal angle
        self.x_threshold = 2.4  # Position at which to fail
        ### End of original parameters

        # Initial state set
        self.x_init = 0.2
        self.dx_init = 0.2
        self.theta_init = 4 * (2 * np.pi / 360)  # Initial angle
        self.dtheta_init = 0.01

        # State space limits
        self.x_max = self.x_threshold + 0.01
        self.dx_max = 2  # The original benchmark has infty; we need a bounded state space and choose this as maximum
        self.theta_max = self.theta_unsafe_radians + 0.01
        self.dtheta_max = 0.5  # The original benchmark has infty; we need a bounded state space and choose this as maximum

        self.dx_max0 = 0.2
        self.dtheta_max0 = 2

        self.high_state = np.array([self.x_max, self.dx_max, self.theta_max, self.dtheta_max], dtype=np.float32)
        self.low_state = -self.high_state

        self.state_dim = len(self.low_state)
        self.plot_dim = [0, 2]

        # TODO: Compute Lipschitz constant
        # Entries of Jacobian; row 1
        # self.dpdp = 1 + self.constant1 * self.delta ** 2 * 3 * np.sin(0.5 * np.pi)  # max[sin(3*p_k)] = 1
        # self.dpdv = self.delta * self.friction / self.v_scaling
        # self.dpdu = self.delta ** 2 * self.power
        #
        # # Entries of Jacobian; row 2
        # self.dvdp = self.constant1 * self.delta * 3 * self.v_scaling  # max[sin(3*p_k)] = 1
        # self.dvdv = self.friction
        # self.dvdu = self.delta * self.power * self.v_scaling

        self.lipschitz_f_l1_A = 1  # max(np.abs(self.dpdp) + np.abs(self.dvdp), np.abs(self.dpdv) + np.abs(self.dvdv))
        self.lipschitz_f_l1_B = 1  # np.abs(self.dpdu) + np.abs(self.dvdu)
        self.lipschitz_f_l1 = max(self.lipschitz_f_l1_A, self.lipschitz_f_l1_B)

        # Set observation / state space
        self.state_space = RectangularSet(low=self.low_state, high=self.high_state, dtype=np.float32)
        print('- State space:')
        print('-- LB: ', self.state_space.low)
        print('-- UB: ', self.state_space.high)

        # This will throw a warning in tests/envs/test_envs in utils/env_checker.py as the space is not symmetric
        #   or normalised as max_torque == 2 by default. Ignoring the issue here as the default settings are too old
        #   to update to follow the openai gym api
        self.min_action = -np.array([self.force_mag])
        self.max_action = np.array([self.force_mag])
        self.action_space = spaces.Box(
            low=self.min_action, high=self.max_action, shape=(1,), dtype=np.float32
        )
        print('- Action space:')
        print('-- LB: ', self.action_space.low)
        print('-- UB: ', self.action_space.high)

        # Set support of noise distribution (which is triangular, zero-centered)
        high = np.array([1], dtype=np.float32)
        self.noise_space = spaces.Box(low=-high, high=high, dtype=np.float32)
        self.noise_dim = len(high)

        # Set target set
        self.target_space = RectangularSet(low=np.array([-self.x_max, -0.4, -self.theta_goal_radians, -0.4]),
                                           high=np.array([self.x_max, 0.4, self.theta_goal_radians, 0.4]),
                                           dtype=np.float32)
        print('- Target set:')
        print('-- LB: ', self.target_space.low)
        print('-- UB: ', self.target_space.high)

        self.init_space = RectangularSet(low=np.array([-self.x_init, -self.dx_init, -self.theta_init, -self.dtheta_init]),
                                         high=np.array([self.x_init, self.dx_init, self.theta_init, self.dtheta_init]), dtype=np.float32)
        print('- Init set:')
        print('-- LB: ', self.init_space.low)
        print('-- UB: ', self.init_space.high)

        self.unsafe_space = MultiRectangularSet([
            # Min/max position
            RectangularSet(low=np.array([-self.x_max, self.low_state[1], self.low_state[2], self.low_state[3]]),
                           high=np.array([-self.x_threshold, self.high_state[1], self.high_state[2], self.high_state[3]]),
                           dtype=np.float32),
            RectangularSet(low=np.array([self.x_threshold, self.low_state[1], self.low_state[2], self.low_state[3]]),
                           high=np.array([self.x_max, self.high_state[1], self.high_state[2], self.high_state[3]]),
                           dtype=np.float32),
            # Min/max angle
            RectangularSet(low=np.array([self.low_state[0], self.low_state[1], self.theta_unsafe_radians, self.low_state[3]]),
                           high=np.array([self.high_state[0], self.high_state[1], self.theta_max, self.high_state[3]]),
                           dtype=np.float32),
            RectangularSet(low=np.array([self.low_state[0], self.low_state[1], -self.theta_max, self.low_state[3]]),
                           high=np.array([self.high_state[0], self.high_state[1], -self.theta_unsafe_radians, self.high_state[3]]),
                           dtype=np.float32),
        ])

        for i, set in enumerate(self.unsafe_space.sets):
            print(f'- Unsafe set {i}:')
            print('-- LB: ', set.low)
            print('-- UB: ', set.high)

        self.init_unsafe_dist = np.abs(self.x_threshold - self.x_init)

        self.num_steps_until_reset = 500

        # Vectorized step, but only with different noise values
        self.vstep_noise_set = jax.vmap(self.step_noise_set, in_axes=(None, None, 0, 0), out_axes=(0, 0))

        # Set to reset to in training (typically the initial state set, or the whole state space)
        self.reset_space = self.state_space

        super(Cartpole, self).__init__()

    @partial(jit, static_argnums=(0,))
    def sample_noise(self, key, size=None):
        return jax.random.triangular(key, self.noise_space.low * jnp.ones(self.noise_dim), jnp.zeros(self.noise_dim),
                                     self.noise_space.high * jnp.ones(self.noise_dim))

    def sample_noise_numpy(self, size=None):
        return np.random.triangular(self.noise_space.low * np.ones(self.noise_dim),
                                    np.zeros(self.noise_dim),
                                    self.noise_space.high * np.ones(self.noise_dim),
                                    size)

    def step(self, u):
        '''
        Step in the gymnasium environment (only used for policy initialization with StableBaselines3).
        '''

        assert self.state is not None, "Call reset before using step method."

        force = np.clip(u, -self.force_mag, self.force_mag)[0]
        w = self.sample_noise_numpy()

        x, x_dot, theta, theta_dot = self.state
        costheta = np.cos(theta)
        sintheta = np.sin(theta)

        # For the interested reader:
        # https://coneural.org/florian/papers/05_cart_pole.pdf
        temp = (
                       force + self.polemass_length * np.square(theta_dot) * sintheta
               ) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
                self.length
                * (4.0 / 3.0 - self.masspole * np.square(costheta) / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        # Dynamics
        x_dot = x_dot + self.tau * xacc
        x = x + self.tau * x_dot
        theta_dot = theta_dot + self.tau * thetaacc
        theta = theta + self.tau * theta_dot

        # x = x + self.tau * x_dot
        # x_dot = x_dot + self.tau * (xacc + 0 * w[0])
        # theta = theta + self.tau * theta_dot
        # theta_dot = theta_dot + self.tau * thetaacc

        # Put together state
        self.state = np.clip(np.array([x, x_dot, theta, theta_dot]), self.state_space.low, self.state_space.high)

        fail = bool(
            self.unsafe_space.contains(np.array([self.state]), return_indices=True)
        )
        goal_reached = np.all(self.state >= self.target_space.low) * np.all(self.state <= self.target_space.high)
        terminated = fail

        if fail:
            costs = 100
        elif goal_reached:
            costs = -10
        else:
            costs = 0

        return np.array(self.state, dtype=np.float32), -costs, terminated, False, {}

    @partial(jit, static_argnums=(0,))
    def step_base(self, state, u, w):
        '''
        Make a step in the dynamics. When defining a new environment, this the function that should be modified.
        '''

        force = jnp.clip(u, -self.force_mag, self.force_mag)[0]

        x, x_dot, theta, theta_dot = state

        costheta = jnp.cos(theta)
        sintheta = jnp.sin(theta)

        # For the interested reader:
        # https://coneural.org/florian/papers/05_cart_pole.pdf
        temp = (
                       force + self.polemass_length * jnp.square(theta_dot) * sintheta
               ) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
                self.length
                * (4.0 / 3.0 - self.masspole * jnp.square(costheta) / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        # Dynamics
        x_dot = x_dot + self.tau * xacc
        x = x + self.tau * x_dot
        theta_dot = theta_dot + self.tau * thetaacc
        theta = theta + self.tau * theta_dot

        # x = x + self.tau * x_dot
        # x_dot = x_dot + self.tau * (xacc + 0 * w[0])
        # theta = theta + self.tau * theta_dot
        # theta_dot = theta_dot + self.tau * thetaacc

        # Put together state
        state = jnp.clip(jnp.array([x, x_dot, theta, theta_dot]), self.state_space.low, self.state_space.high)

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
        noise = self.sample_noise(subkey, size=(2,))

        force = jnp.clip(u, self.min_action, self.max_action)

        goal_reached = self.target_space.jax_contains(jnp.array([state]))[0]
        fail = self.unsafe_space.jax_contains(jnp.array([state]))[0]
        # costs = 0.1 * (force[0] ** 2) - 100 * goal_reached + 100 * fail
        costs = 0 - 10 * goal_reached + 100 * fail

        # Propagate dynamics
        state = self.step_base(state, u, noise)

        steps_since_reset += 1

        terminated = fail
        truncated = (steps_since_reset >= self.num_steps_until_reset)
        done = terminated | truncated
        state, key, steps_since_reset = self._maybe_reset(state, key, steps_since_reset, done)

        return state, key, steps_since_reset, -costs, terminated, truncated, {}
