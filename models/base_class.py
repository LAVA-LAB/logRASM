from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from gymnasium import spaces
from jax import jit
from scipy.stats import triang


class BaseEnvironment:

    def __init__(self):
        # Define vectorized functions
        self.vreset = jax.vmap(self.reset_jax, in_axes=0, out_axes=0)
        self.vstep = jax.vmap(self.step_train, in_axes=0, out_axes=0)
        self.vstep_base = jax.vmap(self.step_base, in_axes=0, out_axes=0)
        self.vstep_noise_batch = jax.vmap(self.step_noise_key, in_axes=(None, 0, None), out_axes=0)

        self.state_dim = len(self.state_space.low)
        self.action_dim = len(self.action_space.low)
        self.noise_dim = len(self.noise_space.low)

        # Vectorized step, but only with different noise values
        self.vstep_noise_set = jax.vmap(self.step_noise_set, in_axes=(None, None, 0, 0), out_axes=(0, 0))

        # Initialize as gym environment
        self.initialize_gym_env()

        if hasattr(self, 'lipschitz_f_l1_A'):
            print(f'- Lipschitz constant of dynamics w.r.t. state variables: {np.round(self.lipschitz_f_l1_A, 3)}')
        if hasattr(self, 'lipschitz_f_l1_B'):
            print(f'- Lipschitz constant of dynamics w.r.t. input variables: {np.round(self.lipschitz_f_l1_B, 3)}')
        print(f'- Overall Lipschitz constant of dynamics: {np.round(self.lipschitz_f_l1, 3)}')

    def initialize_gym_env(self):
        # Initialize state
        self.state = None
        self.steps_beyond_terminated = None

        # Observation space is only used in the gym version of the environment
        self.observation_space = spaces.Box(low=self.reset_space.low, high=self.reset_space.high, dtype=np.float32)

    @partial(jit, static_argnums=(0,))
    def sample_triangular_noise_jax(self, key):
        return jax.random.triangular(key, self.noise_space.low * jnp.ones(self.noise_dim), jnp.zeros(self.noise_dim),
                                     self.noise_space.high * jnp.ones(self.noise_dim))

    def sample_triangular_noise_numpy(self, size=None):
        return np.random.triangular(self.noise_space.low * np.ones(self.noise_dim),
                                    np.zeros(self.noise_dim),
                                    self.noise_space.high * np.ones(self.noise_dim),
                                    size)

    @partial(jit, static_argnums=(0,))
    def step_noise_key(self, state, key, u):
        # Split RNG key
        key, subkey = jax.random.split(key)

        # Sample noise value
        noise = self.sample_triangular_noise_jax(subkey)

        # Propagate dynamics
        state = self.step_base(state, u, noise)

        return state, key

    def _maybe_reset(self, state, key, steps_since_reset, done):
        return jax.lax.cond(done, self._reset, lambda key: (state, key, steps_since_reset), key)

    def _reset(self, key):
        high = self.reset_space.high
        low = self.reset_space.low

        key, subkey = jax.random.split(key)
        new_state = jax.random.uniform(subkey, minval=low,
                                       maxval=high, shape=(self.state_dim,))

        steps_since_reset = 0

        return new_state, key, steps_since_reset

    def reset(self, seed=None, options=None):
        ''' Reset function for pytorch / gymnasium environment '''

        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Sample state uniformly from observation space
        self.state = np.random.uniform(low=self.observation_space.low, high=self.observation_space.high)
        self.last_u = None

        return self.state, {}

    @partial(jit, static_argnums=(0,))
    def reset_jax(self, key):
        state, key, steps_since_reset = self._reset(key)

        return state, key, steps_since_reset

    def integrate_noise_triangular(self, w_lb, w_ub):
        ''' Integrate noise distribution in the box [w_lb, w_ub]. '''

        # For triangular distribution, integration is simple, because we can integrate each dimension individually and
        # multiply the resulting probabilities
        probs = np.ones(len(w_lb))

        # Triangular cdf increases from loc to (loc + c*scale), and decreases from (loc+c*scale) to (loc + scale)
        # So, 0 <= c <= 1.
        loc = self.noise_space.low
        c = 0.5  # Noise distribution is zero-centered, so c=0.5 by default
        scale = self.noise_space.high - self.noise_space.low

        for d in range(self.noise_space.shape[0]):
            probs *= triang.cdf(w_ub[:, d], c, loc=loc[d], scale=scale[d]) - triang.cdf(w_lb[:, d], c, loc=loc[d],
                                                                                        scale=scale[d])

        # In this case, the noise integration is exact, but we still return an upper and lower bound
        prob_ub = probs
        prob_lb = probs

        return prob_lb, prob_ub

    @partial(jit, static_argnums=(0,))
    def propagate_additive_noise_box(self, state, u, w_lb, w_ub):
        ''' Make step with dynamics for a set of noise values '''

        # Propogate dynamics for both the lower bound and upper bound of the noise
        # (note: this works because the noise is additive)
        state_lb = self.step_base(state, u, w_lb)
        state_ub = self.step_base(state, u, w_ub)

        # Compute the mean and the epsilon (difference between mean and ub/lb)
        state_mean = (state_ub + state_lb) / 2
        epsilon = (state_ub - state_lb) / 2

        return state_mean, epsilon


def compute_lipschitz_linear(A, B):
    '''
    Compute Lipschitz constants of the dynamics (with respect to x and u) for linear dynamics

    :param A: Dynamics matrix
    :param B: Control input matrix
    :return:
        - lipschitz_f_l1: L_1 Lipschitz constant of dynamics (in both x and u)
        - lipschitz_f_linfty: L_infty Lipschitz constant of dynamics (in both x and u)
        - lipschitz_f_l1_A: L_1 Lipschitz constant of dynamics (in only x)
        - lipschitz_f_linfty_A: L_infty Lipschitz constant of dynamics (in only x)
        - lipschitz_f_l1_B: L_1 Lipschitz constant of dynamics (in only u)
        - lipschitz_f_linfty_B: L_infty Lipschitz constant of dynamics (in only u)
    '''

    lipschitz_f_l1 = float(np.max(np.sum(np.hstack((A, B)), axis=0)))
    lipschitz_f_linfty = float(np.max(np.sum(np.hstack((A, B)), axis=1)))

    lipschitz_f_l1_A = float(np.max(np.sum(A, axis=0)))
    lipschitz_f_linfty_A = float(np.max(np.sum(A, axis=1)))
    lipschitz_f_l1_B = float(np.max(np.sum(B, axis=0)))
    lipschitz_f_linfty_B = float(np.max(np.sum(B, axis=1)))

    return lipschitz_f_l1, lipschitz_f_linfty, lipschitz_f_l1_A, lipschitz_f_linfty_A, lipschitz_f_l1_B, lipschitz_f_linfty_B


def compute_lipschitz_jacobian(J, G):
    '''
    Compute Lipschitz Jacobian of the dynamics (with respect to x and u) for nonlinear dynamics

    :param J: Jacobian of the dynamics in x
    :param G: Jacobian of the dynamics in u
    :return:
        - lipschitz_f_l1: L_1 Lipschitz constant of dynamics (in both x and u)
        - lipschitz_f_linfty: currently not implemented
        - lipschitz_f_l1_A: L_1 Lipschitz constant of dynamics (in only x)
        - lipschitz_f_linfty_A: currently not implemented
        - lipschitz_f_l1_B: L_1 Lipschitz constant of dynamics (in only u)
        - lipschitz_f_linfty_B: currently not implemented
    '''

    # Lipschitz constants is computed as the maximum sum of columns (after taking absolute values)
    lipschitz_f_l1_A = np.max(np.sum(np.abs(J), axis=0))
    lipschitz_f_l1_B = np.max(np.sum(np.abs(G), axis=0))
    lipschitz_f_l1 = max(lipschitz_f_l1_A, lipschitz_f_l1_B)

    return lipschitz_f_l1, float('nan'), lipschitz_f_l1_A, float('nan'), lipschitz_f_l1_B, float('nan')
