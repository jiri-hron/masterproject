from __future__ import print_function

from __builtin__ import staticmethod

import numpy as np
import theano
import theano.tensor as T

from theano.sandbox.rng_mrg import MRG_RandomStreams
from theano.sandbox.cuda.dnn import dnn_available

from utils import identity_map, log_sum_exp, eps, floatX_arr

floatX = theano.config.floatX


def get_name(parent_name, name):
    return parent_name + '_' + name


class HiddenLayer(object):

    def __init__(self, seed, input, n_in, n_out, W=None, b=None,
                 activation=identity_map, name=None):
        if name is None:
            name = self.__class__.__name__
        self.name = name

        if seed is not None:
            # self.srng.seed(seed)
            np.random.seed(seed)

        if activation is None:
            activation = identity_map

        if W is None:
            # glorot uniform
            W = np.asarray(np.random.uniform(
                low=-np.sqrt(6. / (n_in + n_out)),
                high=np.sqrt(6. / (n_in + n_out)),
                size=(n_in, n_out)
            ), dtype=theano.config.floatX)
            if activation == theano.tensor.nnet.sigmoid:
                W *= 4

        if b is None:
            b = np.zeros((n_out,), dtype=theano.config.floatX)

        self.W = theano.shared(value=W, name='W')
        self.b = theano.shared(value=b, name='b')
        self.params = [self.W, self.b]
        self.activation = activation

        self.input = input
        self.output = self.predict(self.input)

        # parameters of the model
        self.params = [self.W, self.b]

    def predict(self, input):
        lin_output = T.dot(input, self.W) + self.b
        return self.activation(lin_output)

    def __str__(self):
        return self.name


class GaussLayer(object):

    def __init__(self, input, n_in, n_out, M=None, m=None,
                 sigma_W_params=None, sigma_W=1e-3, tune_sigma_W=True,
                 sigma_b_params=None, sigma_b=1e-6, tune_sigma_b=True,
                 l_W=1e-6, l_b=1e-6, activation=identity_map,
                 approx_cols=False, diag_noise=True, b_is_deterministic=True,
                 bias_term=True, name=None, seed=None):
        self.srng = MRG_RandomStreams()
        if seed is not None:
            self.srng.seed(seed)

        self.name = self.__class__.__name__ if name is None else name
        self.approx_cols = approx_cols
        self.diag_noise = diag_noise

        self.n_in = n_in
        self.n_out = n_out
        self.k = n_out if self.approx_cols else n_in
        self.d = n_in if self.approx_cols else n_out

        if seed is not None:
            self.srng.set_rstate(seed)
            np.random.seed(seed)

        if not bias_term:
            b_is_deterministic = True

        if b_is_deterministic:
            sigma_b = 0.0
            tune_sigma_b = False

        self.tune_sigma_W = tune_sigma_W
        self.tune_sigma_b = tune_sigma_b
        self.b_is_deterministic = b_is_deterministic
        self.bias_term = bias_term

        if M is None:
            # glorot uniform
            M = np.asarray(np.random.uniform(
                low=-np.sqrt(6. / (self.k + self.d)),
                high=np.sqrt(6. / (self.k + self.d)),
                size=(self.k, self.d)
            ), dtype=theano.config.floatX)
            if activation == theano.tensor.nnet.sigmoid:
                M *= 4
        else:
            assert M.shape == (self.k, self.d), 'bad M shape'

        if m is None:
            m = np.zeros((self.n_out,), dtype=theano.config.floatX)
        else:
            assert m.shape == (self.n_out,), 'bad m shape'

        self.activation = identity_map if activation is None else activation

        self.M = theano.shared(value=M, name=get_name(self.name, 'M'))
        self.m = theano.shared(value=m, name=get_name(self.name, 'm'))

        if self.diag_noise:
            sigma_W_shape = (self.k, self.d)
            sigma_b_shape = (self.n_out,)
        else:
            sigma_W_shape = (self.k, 1)
            sigma_b_shape = ()

        if sigma_W_params is None:
            sigma_W = 1e-3 if sigma_W is None else sigma_W
            sigma_W_params = (
                np.sqrt(sigma_W) * np.ones(shape=sigma_W_shape) *
                (
                    1 + tune_sigma_W * np.random.uniform(
                        low=-np.sqrt(sigma_W) / 10.0,
                        high=np.sqrt(sigma_W) / 10.0,
                        size=sigma_W_shape
                    )
                )
            ).astype(floatX)
        else:
            assert sigma_W_params.shape == sigma_W_shape, \
                'bad sigma_W_params shape'

        if sigma_b_params is None:
            sigma_b = 1e-6 if sigma_b is None else sigma_b
            sigma_b_params = (
                np.sqrt(sigma_b) * np.ones(shape=sigma_b_shape) *
                (
                    1 + tune_sigma_b * np.random.uniform(
                        low=-np.sqrt(sigma_b),
                        high=np.sqrt(sigma_b),
                        size=sigma_b_shape
                    )
                )
            ).astype(floatX)
        else:
            assert sigma_b_params.shape == sigma_b_shape, \
                'bad sigma_b_params shape'

        self.sigma_W_params = theano.shared(
            sigma_W_params, name=get_name(self.name, 'sigma_W_params')
        )
        self.sigma_b_params = theano.shared(
            sigma_b_params, name=get_name(self.name, 'sigma_b_params')
        )

        # ensure positive
        self.Sigma_W = T.pow(self.sigma_W_params, 2)
        self.Sigma_b = T.pow(self.sigma_b_params, 2)

        if not self.diag_noise:
            # self.Sigma_W = self.Sigma_W.dimshuffle(0, 'x')
            self.Sigma_W = T.addbroadcast(self.Sigma_W, 1)

        self.l_W = l_W
        self.l_b = l_b

        if sigma_W > 0.0:
            noise_W = (
                self.srng.normal(size=(self.k, self.d), std=1.0) * self.Sigma_W
            )
        else:
            noise_W = T.constant(0.0, dtype=floatX)

        if sigma_b > 0.0:
            noise_b = (
                self.srng.normal(size=(self.n_out,), std=1.0) * self.Sigma_b
            )
        else:
            noise_b = T.constant(sigma_b, dtype=floatX)

        self.noise_W = T.as_tensor_variable(
            noise_W, name=get_name(self.name, 'noise_W')
        )
        self.noise_b = T.as_tensor_variable(
            noise_b, name=get_name(self.name, 'noise_b')
        )

        self.W = T.as_tensor_variable(
            self.M + self.noise_W,
            name=get_name(self.name, 'W')
        )
        self.b = T.as_tensor_variable(
            self.m + self.noise_b,
            name=get_name(self.name, 'b')
        )

        if approx_cols:
            self.W = T.transpose(self.W)

        self.input = input
        self.output = self.predict(self.input)

        # parameters of the model
        self.disc_params = []
        self.grad_params = [self.M]
        if self.bias_term:
            self.grad_params += [self.M]
        if tune_sigma_W:
            self.grad_params += [self.sigma_W_params]
        if tune_sigma_b:
            self.grad_params += [self.sigma_b_params]

        # TODO: inspect how to define an interface in python
        # the below are defined for compatibility with StochLayer
        self.disc_params = []
        self.disc_grads = {}
        self.update_wrappers = {}

        if self.b_is_deterministic:
            self.kl = self.kl_W()
        else:
            self.kl = self.kl_W() + self.kl_b()

        self.effect_kl = self.effect_kl_W() + self.effect_kl_b()

    def predict(self, input):
        lin_output = T.dot(input, self.W) + self.b
        return self.activation(lin_output)

    def kl_W(self):
        """
        Calculates the KL-divergence between the current posterior over the
        'm', scaled by corresponding length-scale and no. of data points
        """
        one = T.constant(1.0, dtype=floatX)
        half = T.constant(0.5, dtype=floatX)
        n_weigts = T.constant(self.k, dtype=floatX)
        dim = T.constant(self.d, dtype=floatX)
        log_l_W = T.constant(np.log(self.l_W), dtype=floatX)
        l_W2 = T.constant(self.l_W ** 2, dtype=floatX)

        mean_contrib = half * l_W2 * T.sum(self.M ** 2)
        var_contrib = (one if self.diag_noise else dim) * T.sum(
            half * l_W2 * (self.Sigma_W ** 2) -
            T.log(self.Sigma_W + eps)  # eps to avoid T.log(0.0)
        )
        const_contrib = -(n_weigts * dim * (half + log_l_W))

        return mean_contrib + var_contrib + const_contrib

    def effect_kl_W(self):
        """
        Calculates the part of KL(q(W) || p(W)) dependent on actively
        optimised parameters.
        """
        half = T.constant(0.5, dtype=floatX)
        l_W2 = T.constant(self.l_W ** 2, dtype=floatX)

        ret = half * l_W2 * T.sum(self.M ** 2)

        if self.tune_sigma_W:
            one = T.constant(1.0, dtype=floatX)
            dim = T.cast(self.d, dtype=floatX)

            ret += (one if self.diag_noise else dim) * T.sum(
                half * l_W2 * (self.Sigma_W ** 2) -
                T.log(self.Sigma_W + eps)  # eps to avoid T.log(0.0)
            )

        return ret

    def kl_b(self):
        """
        Calculates the KL-divergence between the current posterior over the
        'm', scaled by corresponding length-scale and no. of data points.
        """
        if self.b_is_deterministic or not self.bias_term:
            return T.constant(0.0, dtype=floatX)
        # if KL over individual paramaters is summed over layers, this will
        # ensure that the contribution from layer with deterministic b is
        # not counted

        one = T.constant(1.0, dtype=floatX)
        half = T.constant(0.5, dtype=floatX)
        dim = T.constant(self.n_out, dtype=floatX)
        log_l_b = T.constant(np.log(self.l_b), dtype=floatX)
        l_b2 = T.constant(self.l_b ** 2, dtype=floatX)

        mean_contrib = half * l_b2 * T.sum(self.m ** 2)
        covar_contrib = (one if self.diag_noise else dim) * T.sum(
            half * l_b2 * (self.Sigma_b ** 2) -
            T.log(self.Sigma_b + eps)  # eps to avoid T.log(0.0)
        )
        const_contrib = -(dim * (half + log_l_b))
        # no multiplication by number of bias vectors -> there's only one

        return mean_contrib + covar_contrib + const_contrib

    def effect_kl_b(self):
        """
        Calculates the part of KL(q(b) || p(b)) dependent on actively
        optimised parameters.
        """
        if self.b_is_deterministic or not self.bias_term:
            return T.constant(0.0, dtype=floatX)

        half = T.constant(0.5, dtype=floatX)
        l_b2 = T.constant(self.l_b ** 2, dtype=floatX)

        ret = half * l_b2 * T.sum(self.m ** 2)

        if self.tune_sigma_b:
            one = T.constant(1.0, dtype=floatX)
            dim = T.constant(self.n_out, dtype=floatX)
            ret += (one if self.diag_noise else dim) * T.sum(
                half * l_b2 * (self.Sigma_b ** 2) -
                T.log(self.Sigma_b + eps)  # eps to avoid T.log(0.0)
            )

        return ret

    def get_param_dictionary(self):
        return {
            'M': np.array(self.M.eval()),
            'm': np.array(self.m.eval()),
            'sigma_W_params': np.array(self.sigma_W_params.eval()),
            'sigma_b_params': np.array(self.sigma_b_params.eval())
        }

    def __str__(self):
        return self.name


class DropoutLayer(object):
    """
    Standard implementation of the Dropout layer; no scaling of the output.
    """

    def __init__(self, input, dropout=0.5, seed=None, name=None):
        if name is None:
            name = self.__class__.__name__
        self.name = name

        self.srng = MRG_RandomStreams()
        if seed is not None:
            self.srng.set_rstate(seed)

        self.input = input
        self.dropout = dropout

        self.output = self.predict(input)

    def predict(self, input):
        retain_prob = 1 - self.dropout

        # use nonsymbolic shape for dropout mask if possible
        input_shape = self.input.shape
        if any(s is None for s in input_shape):
            input_shape = input.shape

        mask = self.srng.binomial(input_shape, p=retain_prob,
                                  dtype=theano.config.floatX)

        return input * T.cast(mask, dtype=theano.config.floatX)

    def __str__(self):
        return self.name


class StochLayer(object):
    """
    Implementation of the proposed stochastic layer where the posterior over
    weights (of a Bayesian Neural network) is approximated via a Mixture of
    Gaussian variables.

    The mixture components are isotropic Gaussians with equal variance (model
    hyperparameter) - if this variance is set to zero, the approximating
    distribution becomes categorical. This variance is also shared with
    approximate posterior over the bias vector (which is a Gaussian).

    There are two main modes: 'untied' where each posterior is approximated by
    a different Mixture of Gaussians model, and 'tied' where posterior over
    each weight has its own set of mixture component probabilities, but the
    mixture mean and covariance parameters are shared.

    The tunable parameters are A (matrix of parameters used to calculate the
    mixture component probabilities for each vector in the weight matrix). M
    (the component means), and b (mean vector of the bias posterior). The
    hyperparameters are sigma (standard deviation of the Gaussian components
    and of the approximate posterior over b), number of mixture components,
    number of weights being approximated, length scales for the prior over
    W and b (if applicable), and the activation function.

    The layer implements masking of some of its parameters; it is therefore
    possible to only optimise a subset of the parameters.
    """

    @staticmethod
    def init_params(A, M, m, sigmoid_act, k, d, t, n_out,
                    sigma_W, tune_sigma_W, sigma_W_params,
                    sigma_b, tune_sigma_b, sigma_b_params,
                    A_scale, tie_weights, diag_noise,
                    dropout, dropout_prob):
        """
        Initialises the parameters A, M, and m to either the supplied values
        (if available) or randomly.
        """
        M_shape = (t, d) if tie_weights else (k, t, d)
        if M is None:
            M = np.asarray(
                np.random.uniform(
                    low=-np.sqrt(6.0 / (t + d)),
                    high=np.sqrt(6.0 / (t + d)),
                    size=M_shape),
                dtype=floatX)
            if sigmoid_act:
                M *= 4
            if dropout:
                if tie_weights:
                    M[-1] = 0.0
                else:
                    M[:, -1] = 0.0
        else:
            assert M.shape == M_shape, \
                'bad M shape; k=%d, t=%d, shape=%s, tie_weights=%s' % \
                (k, t, str(M.shape), str(tie_weights))

        if m is None:
            m = np.zeros((n_out,), dtype=floatX)
        else:
            assert m.shape == (n_out,), \
                'bad m shape, should (n_out=%d) vs. was len(m)=%d' % \
                (n_out, len(m))

        if A is None:
            # doing np.log of the sampled values, i.e. to initialise to
            # the sampled distribution leads to having very low probs of some
            # components, which subequently causes the T.log(self.Pi) operation
            # in update_A to return -Infty, which in further calculations
            # causes appearance of NaN values
            # -> don't initialise to too small values
            if dropout:
                A = np.tile(
                    [[np.log(1 - dropout_prob), np.log(dropout_prob)]], (k, 1)
                ).astype(floatX)
            else:
                A = floatX_arr(
                    np.random.dirichlet(tuple([A_scale] * t), k)
                )
        else:
            assert A.shape == (k, t), \
                'bad A shape; should (k=%d, t=%d) vs. was (%d, %d)' % \
                (k, t, A.shape[0], A.shape[1])

        # initialise sigma_W_params
        if diag_noise:
            sigma_W_shape = (t, d) if tie_weights else (k, t, d)
            sigma_b_shape = (n_out,)
        else:
            sigma_W_shape = (t, 1) if tie_weights else (k, t)
            sigma_b_shape = ()

        if sigma_W_params is None:
            sigma_W = 1e-3 if sigma_W is None else sigma_W
            sigma_W_params = (
                np.sqrt(sigma_W) * np.ones(shape=sigma_W_shape) *
                (
                    1 + tune_sigma_W * np.random.uniform(
                        low=-np.sqrt(sigma_W) / 10.0,
                        high=np.sqrt(sigma_W) / 10.0,
                        size=sigma_W_shape
                    )
                )
            ).astype(floatX)
        # the uniform noise is only introduced if sigmas are to be optimised
        # hence the multiplication by tune_sigma
        else:
            assert sigma_W_params.shape == sigma_W_shape, \
                'bad sigma_W_params shape'

        # initialise sigma_b_params
        if sigma_b_params is None:
            sigma_b = 1e-6 if sigma_b is None else sigma_b
            sigma_b_params = (
                np.sqrt(sigma_b) * np.ones(shape=sigma_b_shape) *
                (
                    1 + tune_sigma_b * np.random.uniform(
                        low=-np.sqrt(sigma_b) / 10.0,
                        high=np.sqrt(sigma_b) / 10.0,
                        size=sigma_b_shape
                    )
                )
            ).astype(floatX)
        else:
            assert sigma_b_params.shape == sigma_b_shape, \
                'bad sigma_b_params shape'

        return A, M, m, sigma_W_params, sigma_b_params

    @staticmethod
    def init_masks(A_mask, M_mask, m_mask, k, d, t, n_out,
                   tune_A, tie_weights, dropout):
        """
        Initialises the masks of A and M to either the supplied values (if
        available) or randomly.
        """
        if M_mask is None:
            shape = (t, d) if tie_weights else (k, t, d)
            M_mask = np.ones(shape, dtype=floatX)
            if dropout:
                if tie_weights:
                    M_mask[-1] = 0.0
                else:
                    M_mask[:, -1] = 0.0
        if A_mask is None:
            array_constr = np.ones if tune_A else np.zeros
            A_mask = array_constr((k, t), dtype=floatX)
        if m_mask is None:
            m_mask = np.ones((n_out,), dtype=floatX)

        assert (M_mask.shape == (t, d) and tie_weights) or \
               (M_mask.shape == (k, t, d) and not tie_weights), \
            'bad M_mask shape'
        assert A_mask.shape == (k, t), 'bad A_mask shape'
        assert m_mask.shape == (n_out,), 'bad m_mask shape'

        return A_mask, M_mask, m_mask

    @staticmethod
    def setup_shared_vars(name, M, m, A, M_mask, m_mask, A_mask,
                          sigma_W_params, sigma_b_params):
        M = theano.shared(
            M, name=get_name(name, 'M'), allow_downcast=True
        )
        m = theano.shared(
            m, name=get_name(name, 'm'), allow_downcast=True
        )
        A = theano.shared(
            A, name=get_name(name, 'A'), allow_downcast=True
        )
        M_mask = theano.shared(
            M_mask, name=get_name(name, 'M_mask'),
            allow_downcast=True
        )
        m_mask = theano.shared(
            m_mask, name=get_name(name, 'm_mask')
        )
        A_mask = theano.shared(
            A_mask, name=get_name(name, 'A_mask'),
            allow_downcast=True
        )
        sigma_W_params = theano.shared(
            sigma_W_params, name=get_name(name, 'sigma_W_params')
        )
        sigma_b_params = theano.shared(
            sigma_b_params, name=get_name(name, 'sigma_b_params')
        )

        return M, m, A, M_mask, m_mask, A_mask, sigma_W_params, sigma_b_params

    @staticmethod
    def setup_constants(name, l_W, l_b):
        l_W = T.constant(l_W, name=get_name(name, 'l_W'), dtype=floatX)
        l_b = T.constant(l_b, name=get_name(name, 'l_b'), dtype=floatX)

        return l_W, l_b

    @staticmethod
    def setup_Pi_Z(name, A, srng, k, t):
        Pi = T.as_tensor_variable(T.nnet.softmax(A), name=get_name(name, 'Pi'))

        # T.log(Pi) is faster if cuDNN is present, otherwise use the more
        # stable A - log_sum_exp(self.A) to avoid calculation on CPU
        log_Pi = T.as_tensor_variable(
            T.log(Pi + eps) if dnn_available()
            # adding the constant to prevent underflows
            else A - log_sum_exp(A, axis=1),
            name=get_name(name, 'log_Pi')
        )

        if t == 1:
            # faster when the outcome is deterministic
            Z = T.zeros(k, dtype='uint16')
        else:
            # approx. 50% faster than using the above srng.multinomial
            unif = srng.uniform(size=(k, ), dtype=floatX)
            Z = (T.gt(unif[:, None], T.cumsum(Pi, axis=1))
                 .astype(floatX).sum(axis=1).astype('uint16'))
            # prevent errors due to the float sum
            Z = T.minimum(Z, t - 1)

        Z = T.as_tensor_variable(Z, name=get_name(name, 'Z'))
        Z_onehot = T.as_tensor_variable(
            T.extra_ops.to_one_hot(Z, t, dtype=floatX),
            name=get_name(name, 'Z_onehot')
        )

        return Pi, log_Pi, Z, Z_onehot

    @staticmethod
    def setup_W_b_samplers(
            name, Z, M, m, srng, k, d, n_out, tie_weights, approx_cols,
            sigma_W, sigma_W_params, sigma_b, sigma_b_params, diag_noise
    ):

        Sigma_W = T.pow(sigma_W_params, 2)  # ensure positive
        Sigma_b = T.pow(sigma_b_params, 2)  # ensure positive

        # M and Sigma_W have different shapes based on the tie_weights settings
        if tie_weights:
            idx_sampled = Z
        else:
            idx_sampled = (T.arange(k, dtype='uint16'), Z)

        sampled_means = M[idx_sampled]
        sampled_sigmas_W = Sigma_W[idx_sampled]
        sampled_sigmas_b = Sigma_b

        # broadcastable dim needed for below elemwise product with gauss noise
        if not diag_noise:
            if tie_weights:
                sampled_sigmas_W = T.addbroadcast(sampled_sigmas_W, 1)
            else:
                sampled_sigmas_W = sampled_sigmas_W.dimshuffle(0, 'x')

        # skip random mixture component noise generation if sigma == 0
        if sigma_W > 0.0:
            noise_W = srng.normal(size=(k, d), std=1.0) * sampled_sigmas_W
        else:
            noise_W = T.constant(0.0, dtype=floatX)

        if sigma_b > 0.0:
            noise_b = srng.normal(size=(n_out,), std=1.0) * sampled_sigmas_b
        else:
            noise_b = T.constant(0.0, dtype=floatX)

        # cast as tensor variables
        noise_W = T.as_tensor_variable(noise_W, name=get_name(name, 'noise_W'))
        noise_b = T.as_tensor_variable(noise_b, name=get_name(name, 'noise_b'))

        W = T.as_tensor_variable(
            sampled_means + noise_W, name=get_name(name, 'W')
        )
        b = T.as_tensor_variable(
            m + noise_b, name=get_name(name, 'b')
        )

        # the rest of the changes is ensured by setting k,d based on approx_cols
        if approx_cols:
            W = T.transpose(W)

        return W, b, Sigma_W, Sigma_b

    def __init__(self, input, n_in, n_out, n_cat,
                 M=None, M_mask=None, A=None, A_mask=None, m=None, m_mask=None,
                 sigma_W_params=None, sigma_b_params=None, diag_noise=True,
                 sigma_W=1e-3, tune_sigma_W=True,
                 sigma_b=1e-6, tune_sigma_b=True,
                 W_len_scale=1e-6, b_len_scale=1e-6,
                 A_scale=1.0, tune_A=True, dropout=False, dropout_prob=0.5,
                 activation=identity_map, tie_weights=False,
                 approx_cols=False, b_is_deterministic=False,
                 bias_term=True, name=None, seed=None):
        """
        Constructs the Stochastic Layer.

        :param n_in: number of inputs
        :type n_in: int
        :param n_out: number of outputs (i.e. hidden units)
        :type n_out: int
        :param n_cat: number of required mixture components
        :type n_cat: int
        :param input: the input 2D matrix (one input per row)
        :type input:  [theano TensorVariable]
        :param A: the matrix of alpha parameters (one vector of alphas
            corresponding to a particular weight per row); shape should be
            (n_in, n_categories)
        :type A: numpy.ndarray
        :param M: the array of mean parameters for each of the mixture
            component; shape should be (n_cat, n_out) tie_weights=True,
            (n_in, n_cat, n_out) if tie_weights=False
        :type M: numpy.ndarray
        :param m: vector of initial bias parameter Gaussian means (expected
            an array with shape (n_out,) )
        :type m: numpy.ndarray
        :param sigma_W: standard deviation for all mixture components and bias
            parameter Gaussians
        :type sigma_W: float or int
        :param seed: seed for the random number generator in 'numpy' and in
                'MRG_RandomStreams'
        :type seed: int
        :param activation: the activation function used
        :type activation: [theano expression]
        :param M_mask: mask preventing updates of particular values in the
            'M' matrix; must be of same shape as M
        :type M_mask: numpy.ndarray
        :param A_mask: mask preventing updates of particular values in the
            'A' matrix; must be of same shape as A
        :type A_mask: numpy.ndarray
        :param tune_A: should the mixture component probabilities be optimised
        :type tune_A: bool
        :param: should the mixture component Gaussians be tied for all weights
        :type: bool
        """

        self.srng = MRG_RandomStreams()
        if seed is not None:
            self.srng.set_rstate(seed)

        if dropout:
            assert (not tie_weights) and n_cat == 2 and \
                   0.0 < dropout_prob < 1.0 and (not approx_cols)
            tune_A = False

        self.name = self.__class__.__name__ if name is None else name
        self.approx_cols = approx_cols
        self.tie_weights = tie_weights
        self.tune_A = tune_A
        self.dropout = dropout
        self.dropout_prob = dropout_prob

        self.n_in = n_in
        self.n_out = n_out
        self.k = n_out if self.approx_cols else n_in
        self.d = n_in if self.approx_cols else n_out
        self.t = n_cat

        self.activation = identity_map if activation is None else activation

        if not bias_term:
            b_is_deterministic = True

        if b_is_deterministic:
            sigma_b = 0.0
            tune_sigma_b = False

        self.tune_sigma_W = tune_sigma_W
        self.tune_sigma_b = tune_sigma_b
        self.b_is_deterministic = b_is_deterministic
        self.bias_term = bias_term
        self.diag_noise = diag_noise

        # initialise parameters and masks which prevent parameter updates
        sigmoid_act = self.activation == theano.tensor.nnet.sigmoid
        A, M, m, sigma_W_params, sigma_b_params = StochLayer.init_params(
            A=A, M=M, m=m, sigmoid_act=sigmoid_act, diag_noise=self.diag_noise,
            k=self.k, d=self.d, t=self.t, n_out=n_out,
            A_scale=A_scale, tie_weights=self.tie_weights,
            sigma_W=sigma_W, sigma_W_params=sigma_W_params,
            tune_sigma_W=tune_sigma_W,
            sigma_b=sigma_b, sigma_b_params=sigma_b_params,
            tune_sigma_b=tune_sigma_b,
            dropout=self.dropout, dropout_prob=self.dropout_prob
        )
        A_mask, M_mask, m_mask = StochLayer.init_masks(
            A_mask=A_mask, M_mask=M_mask, m_mask=m_mask,
            k=self.k, d=self.d, t=self.t, n_out=self.n_out,
            tune_A=self.tune_A, tie_weights=self.tie_weights,
            dropout=self.dropout
        )

        # check if either of the parameter matrices is fixed to all zeros
        self.no_M_update = np.sum(M_mask.astype('int')) == 0
        self.no_A_update = (
            (not self.tune_A) or (np.sum(A_mask.astype('int')) == 0)
        )
        self.no_m_update = (
            (not self.bias_term) or np.sum(m_mask.astype('int')) == 0
        )

        # set up shared variables
        self.M, self.m, self.A, self.M_mask, self.m_mask, self.A_mask, \
        self.sigma_W_params, self.sigma_b_params = (
            StochLayer.setup_shared_vars(
                name=self.name, M=M, m=m, A=A,
                M_mask=M_mask, m_mask=m_mask, A_mask=A_mask,
                sigma_W_params=sigma_W_params,
                sigma_b_params=sigma_b_params
            )
        )

        # set up constants
        self.l_W, self.l_b = StochLayer.setup_constants(
            name=self.name, l_W=W_len_scale, l_b=b_len_scale
        )

        # set up variables for sampling mixture components
        self.Pi, self.log_Pi, self.Z, self.Z_onehot = StochLayer.setup_Pi_Z(
            name=self.name, A=self.A, k=self.k, t=self.t, srng=self.srng
        )

        # set up variables for sampling W and b
        self.W, self.b, self.Sigma_W, self.Sigma_b = (
            StochLayer.setup_W_b_samplers(
                name=self.name, Z=self.Z, M=self.M, m=self.m,
                k=self.k, d=self.d, n_out=self.n_out, srng=self.srng,
                tie_weights=tie_weights, approx_cols=approx_cols,
                sigma_W=sigma_W, sigma_W_params=self.sigma_W_params,
                sigma_b=sigma_b, sigma_b_params=self.sigma_b_params,
                diag_noise=self.diag_noise
            )
        )
        # remember that Sigma_W and Simga_b are square roots of resp covariances

        half = T.constant(0.5, dtype=floatX)
        if self.tie_weights:
            self.M_squared_norms = T.sum(self.M ** 2, axis=1).dimshuffle('x', 0)

            self.lw2_sigma2_div_2_min_log_sigma = T.sum(
                half * (self.l_W ** 2) * (self.Sigma_W.T ** 2) -
                T.log(self.Sigma_W.T + eps),  # eps to avoid T.log(0.0)
                axis=0, keepdims=True  # N: 1st dimension must be broadcastable
            )
        else:
            self.M_squared_norms = T.sum(self.M ** 2, axis=2)

            if self.diag_noise:
                self.lw2_sigma2_div_2_min_log_sigma = T.sum(
                    half * (self.l_W ** 2) * (self.Sigma_W ** 2) -
                    T.log(self.Sigma_W + eps),  # eps to avoid T.log(0.0)
                    axis=2
                )
            else:
                self.lw2_sigma2_div_2_min_log_sigma = (
                    half * (self.l_W ** 2) * (self.Sigma_W ** 2) -
                    T.log(self.Sigma_W + eps)  # eps to avoid T.log(0.0)
                )

        self.input = input
        self.output = self.predict(self.input)

        # 'grad_params' -> AutoDiff can be used to calculate gradients
        self.grad_params = [self.M]
        if self.bias_term:
            self.grad_params += [self.m]
        if tune_sigma_W:
            self.grad_params += [self.sigma_W_params]
        if tune_sigma_b:
            self.grad_params += [self.sigma_b_params]

        # 'disc_params' -> AutoDiff cannot be used
        self.disc_params = [self.A]

        if self.b_is_deterministic or not self.bias_term:
            self.kl = self.kl_W()
        else:
            self.kl = self.kl_W() + self.kl_b()

        self.effect_kl = self.effect_kl_W() + self.effect_kl_b()

        # function(s) calculating gradients based on current mean likelihood
        self.disc_grads = {
            self.A: self.gradient_A
        }

        # functions through which all proposed updates to parameters are passed
        # mostly used to prevent updates of parameters based on masks
        self.update_wrappers = {
            self.M: self.update_M,
            self.m: self.update_m,
            self.A: self.update_A
        }

    def predict(self, input):
        """
        Calculate the layer's output
        """
        # W = T.dot(self.Z, self.M) + self.noise_W

        return self.activation(T.dot(input, self.W) + self.b)

    def kl_W(self):
        """
        Calculates the approximation to the KL-divergencce between the current
        approx distribution and the prior over W, up to an additive constant.
        """
        one = T.constant(1.0, dtype=floatX)
        half = T.constant(0.5, dtype=floatX)
        n_weights = T.constant(self.k, dtype=floatX)
        dim = T.constant(self.d, dtype=floatX)
        const = T.constant(0.5 * (np.log(2 * np.pi) + 1), dtype=floatX)

        mean_contrib = (
            half * (self.l_W ** 2) * T.sum(self.Pi * self.M_squared_norms)
        )
        var_contrib = (
            (one if self.diag_noise else dim) *
            T.sum(self.Pi * self.lw2_sigma2_div_2_min_log_sigma)
        )
        const_contrib = (
            T.sum(self.Pi * self.log_Pi) -
            n_weights * dim * (T.log(self.l_W + eps) + const)
            # eps to avoid T.log(0.0)
        )

        return mean_contrib + var_contrib  + const_contrib

    def effect_kl_W(self):
        """"
        Calculates the part of KL(q(W) || p(W)) that is dependent on actively
        optimised parameters.
        """
        ret = T.constant(0.0, dtype=floatX)
        one = T.constant(1.0, dtype=floatX)

        if not self.no_M_update:
            half = T.constant(0.5, dtype=floatX)
            ret += (
                half * (self.l_W ** 2) * T.sum(self.Pi * self.M_squared_norms)
            )

        if self.tune_sigma_W:
            dim = T.constant(self.d, dtype=floatX)
            ret += (
                (one if self.diag_noise else dim) *
                T.sum(self.Pi * self.lw2_sigma2_div_2_min_log_sigma)
            )

        return ret

    def kl_b(self):
        """
        Calculates the KL-divergence between the current posterior over b
        up to an additive constant.
        """
        if self.b_is_deterministic or not self.bias_term:
            return T.constant(0.0, dtype=floatX)
        # if KL over individual paramaters is summed over layers, this will
        # ensure that the contribution from layer with deterministic b is
        # not counted

        one = T.constant(1.0, dtype=floatX)
        half = T.constant(0.5, dtype=floatX)
        dim = T.constant(self.n_out, dtype=floatX)

        mean_contrib = half * (self.l_b ** 2) * T.sum(self.m ** 2)
        covar_contrib = (one if self.diag_noise else dim) * T.sum(
            half * (self.l_b ** 2) * (self.Sigma_b ** 2) -
            T.log(self.Sigma_b + eps)  # eps to avoid T.log(0.0)
        )
        const_contrib = (-dim) * (half + T.log(self.l_b + eps))
        # no multiplication by number of bias vectors -> there's only one

        return mean_contrib + covar_contrib + const_contrib

    def effect_kl_b(self):
        """"
        Calculates the part of KL(q(b) || p(b)) that is dependent on actively
        optimised parameters.
        """
        ret = T.constant(0.0, dtype=floatX)

        if self.b_is_deterministic or not self.bias_term:
            return ret

        half = T.constant(0.5, dtype=floatX)

        if not self.no_m_update:
            ret += half * (self.l_b ** 2) * T.sum(self.m ** 2)

        if self.tune_sigma_b:
            one = T.constant(1.0, dtype=floatX)
            dim = T.constant(self.n_out, dtype=floatX)
            ret += (one if self.diag_noise else dim) * T.sum(
                half * (self.l_b ** 2) * (self.Sigma_b ** 2) -
                T.log(self.Sigma_b + eps)  # eps to avoid T.log(0.0)
            )

        return ret

    def gradient_A(self, mean_log_likelihood, n_data,
                   ll_const_ctrl_var, ll_model_ctrl_var):
        """
        Calculates the update to the 'A' matrix based on current likelihood
        """
        one = T.constant(1.0, dtype=floatX)
        half = T.constant(0.5, dtype=floatX)
        d = T.constant(self.d, dtype=floatX)

        if self.no_A_update:
            return T.zeros_like(self.A)  # more efficient to avoid computation

        likelihood_contrib = (
            (self.Z_onehot - self.Pi) * (
                mean_log_likelihood - ll_const_ctrl_var - ll_model_ctrl_var
            )
        )
        products_kl = self.Pi * (
            self.log_Pi + (
                half * (self.l_W ** 2) * self.M_squared_norms +
                (one if self.diag_noise else d) *
                self.lw2_sigma2_div_2_min_log_sigma
            )
        )
        kl_contrib = (
            products_kl - (
                self.Pi * T.sum(products_kl, axis=1).dimshuffle(0, 'x')
            )
        )
        kl_contrib /= T.constant(n_data, name='n_data', dtype=floatX)
        # unlike the likelihood, the KL term must be divided by the number
        # of datapoints in the dataset (no SVI applied on the KL term)

        # minimising the negative log-lower bound divided by number of samples
        return -(likelihood_contrib - kl_contrib)

    def update_m(self, updated_value):
        """
        Updates the m vector.
        """
        if self.no_m_update:
            return self.m  # more efficient to avoid all computations

        return updated_value * self.m_mask

    def update_M(self, updated_value):
        """
        Updates the M array.
        """
        if self.no_M_update:
            return self.M  # more efficient to avoid all computations

        return updated_value * self.M_mask

    def update_A(self, updated_value):
        """
        Updates the A matrix.
        """
        if self.no_A_update:
            return self.A  # more efficient to avoid all computations

        return updated_value * self.A_mask

    def get_param_dictionary(self):
        return {
            'M': np.array(self.M.eval()),
            'm': np.array(self.m.eval()),
            'A': np.array(self.A.eval()),
            'sigma_W_params': np.array(self.sigma_W_params.eval()),
            'sigma_b_params': np.array(self.sigma_b_params.eval())
        }

        # TODO: reimplement with respect to posterior sigmas if needed
        # def log_prob_W(self):
        #     two = T.constant(2.0, dtype=floatX)
        #     pi = T.constant(np.pi, dtype=floatX)
        #     return (
        #         -(self.k * self.d)/two * T.log(two * pi * self.sigma_sq + eps) -
        #         T.sum((self.W - self.sampled_means)**2) / (two * self.sigma_sq)
        #     )
        #
        # def log_prob_b(self):
        #     two = T.constant(2.0, dtype=floatX)
        #     pi = T.constant(np.pi, dtype=floatX)
        #     return (
        #         -self.d/two * T.log(two * pi * self.sigma_sq + eps) -
        #         T.sum((self.b - self.m) ** 2) / (two * self.sigma_sq)
        #     )

    def __str__(self):
        return self.name


class SoftmaxLayer(object):
    """
    Softmax activation layer. Doesn't contain weight parameters of its own;
    if needed, these should be added as a separate layer with linear activation.
    """

    def __init__(self, input, name=None):
        if name is None:
            name = self.__class__.__name__
        self.name = name

        self.input = input

        self.p_y_given_x = self.predict(self.input)
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

    def predict(self, input):
        """
        Calculate the layer's output
        """
        return T.nnet.softmax(input)

    def log_likelihood(self, y):
        """
        Calculate the log likelihood
        """

        # eps to avoid T.log(0.0)
        log_p_y_given_x = T.log(self.p_y_given_x + eps)

        return log_p_y_given_x[T.arange(y.shape[0], dtype='uint16'), y]

    def mean_log_likelihood(self, y):
        """
        Calculate the mean log likelihood
        """
        return T.mean(self.log_likelihood(y), dtype=floatX)

    def errors(self, y):
        """
        Calculate the mean misclassification error
        """

        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y), dtype=floatX)
        else:
            raise NotImplementedError()

    def __str__(self):
        return self.name
