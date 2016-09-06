from __future__ import print_function

import pickle

import numpy as np
import theano
import theano.tensor as T

# from theano.compile.nanguardmode import NanGuardMode
from lasagne.updates import adam

from layers import GaussLayer, SoftmaxLayer

floatX = theano.config.floatX


def init_from_file(path, n_in=28**2, n_out=10, seed=None):
    with open(path, 'rb') as f:
        params = pickle.load(f)

    # TODO: save info to the pickled file and replace the below inferred values

    n_layers = len(params)
    n_hidden_layers = n_layers - 1
    n_units_in = params['h1']['m'].shape[0]
    n_units_hidden = params['h2']['m'].shape[0] \
        if 'h2' in params.keys() else n_units_in

    diag_noise = params['h1']['sigma_b_params'].shape != ()
    approx_cols = params['h1']['M'].shape[1] == n_in

    # assumed it was trained to using as initilisation for deep GPs
    divide_1st_layer_by_its_n_out = False  # gauss_io in (INPUT, IN_OUT)
    b_out_deterministic = True  # always

    M_lst = []
    m_lst = []
    sigma_W_params_lst = []
    sigma_b_params_lst = []

    # acess by name to prevent changes of order
    keys = ['h' + str(i) for i in xrange(1, n_hidden_layers + 1)] + ['out']
    assert set(keys) == set(params.keys()), 'invalid grad_params supplied'
    for key in keys:
        layer_params = params[key]
        M_lst += [layer_params['M']]
        m_lst += [layer_params['m']]
        sigma_W_params_lst += [layer_params['sigma_W_params']]
        sigma_b_params_lst += [layer_params['sigma_b_params']]

    # define symbolic variables
    index = T.lscalar('index')
    X = T.matrix('X', dtype=floatX)
    y = T.ivector('y')

    model = GaussBNN(
        X=X, n_in=n_in, n_out=n_out, n_hidden_layers=n_hidden_layers,
        n_units_in=n_units_in, n_units_hidden=n_units_hidden,
        M_lst=M_lst, m_lst=m_lst,
        sigma_W_params_lst=sigma_W_params_lst,
        sigma_b_params_lst=sigma_b_params_lst,
        diag_noise=diag_noise, approx_cols=approx_cols,
        divide_1st_layer_by_its_n_out=divide_1st_layer_by_its_n_out,
        b_out_deterministic=b_out_deterministic, seed=seed
    )

    return model, X, y, index


class GaussBNN(object):

    def __init__(self, X, n_in, n_out, n_hidden_layers,
                 n_units_in, n_units_hidden,
                 M_lst=None, m_lst=None,
                 sigma_W_params_lst=None, sigma_b_params_lst=None,
                 sigma_W=1e-3, tune_sigma_W=True,
                 sigma_b=1e-6, tune_sigma_b=True,
                 l_W=1e-6, l_b=1e-6,
                 diag_noise=True, approx_cols=False,
                 divide_1st_layer_by_its_n_out=False,
                 b_out_deterministic=False, seed=None):
        assert n_hidden_layers > 0, 'n_layers must be positive'

        n_layers = n_hidden_layers + 1

        M_lst = [None] * (n_layers) if M_lst is None else M_lst
        m_lst = [None] * (n_layers) if m_lst is None else m_lst

        if sigma_W_params_lst is None:
            sigma_W_params_lst = [None] * (n_layers)
        if sigma_b_params_lst is None:
            sigma_b_params_lst = [None] * (n_layers)

        assert \
            len(M_lst) ==  len(m_lst) == len(sigma_W_params_lst) == \
            len(sigma_b_params_lst) == n_layers, \
            'length of all lists must be hte same and equal to ' \
            '(n_layers + 1) where the +1 is for the output layer mapping'

        # set seed to ensure each layer is init differently (cf. seed += 1)
        seed = np.random.randint(int(1e6)) if seed is None else seed
        np.random.seed(seed)

        def activation(x):
            return T.nnet.relu(x, alpha=0.1)

        self.in_layer = GaussLayer(
            input=X, n_in=n_in, n_out=n_units_in,
            M=M_lst[0], m=m_lst[0],
            sigma_W=sigma_W, tune_sigma_W=tune_sigma_W,
            sigma_W_params=sigma_W_params_lst[0],
            sigma_b=sigma_b, tune_sigma_b=tune_sigma_b,
            sigma_b_params=sigma_b_params_lst[0],
            l_W=l_W, l_b=l_b, diag_noise=diag_noise,
            activation=activation, approx_cols=approx_cols,
            seed=seed, name='h1'
        )
        self.layers = [self.in_layer]
        seed += 1

        # specific settings necessary for initialisation of deep GPs
        if divide_1st_layer_by_its_n_out:
            sqrt_n_out = T.constant(self.in_layer.n_out ** 0.5, dtype=floatX)
            self.in_layer.output /= sqrt_n_out

        # the first hidden layer was already set up above
        for i in xrange(1, n_hidden_layers):
            prev_layer = self.layers[-1]
            layer = GaussLayer(
                input=prev_layer.output,
                n_in=prev_layer.n_out, n_out=n_units_hidden,
                M=M_lst[i], m=m_lst[i],
                sigma_W=sigma_W, tune_sigma_W=tune_sigma_W,
                sigma_W_params=sigma_W_params_lst[i],
                sigma_b=sigma_b, tune_sigma_b=tune_sigma_b,
                sigma_b_params=sigma_b_params_lst[i],
                l_W=l_W, l_b=l_b, diag_noise=diag_noise,
                activation=activation, name='h' + str(i + 1),
                approx_cols=approx_cols, seed=seed
            )
            self.layers += [layer]
            seed += 1

        # initialised separately because of the necessary linear activation
        prev_layer = self.layers[-1]
        self.out_layer = GaussLayer(
            input=prev_layer.output, n_in=prev_layer.n_out, n_out=n_out,
            M=M_lst[-1], m=m_lst[-1],
            sigma_W=sigma_W, tune_sigma_W=tune_sigma_W,
            sigma_W_params=sigma_W_params_lst[-1],
            sigma_b=sigma_b, tune_sigma_b=tune_sigma_b,
            sigma_b_params=sigma_b_params_lst[-1],
            l_W=l_W, l_b=l_b, diag_noise=diag_noise,
            b_is_deterministic=b_out_deterministic,
            approx_cols=approx_cols, name='out', seed=seed
        )
        self.layers += [self.out_layer]

        self.softmax = SoftmaxLayer(
            input=self.out_layer.output, name='softmax'
        )

        self.params = reduce(
            lambda x, y: x + y, [layer.grad_params for layer in self.layers]
        )

        self.input = X

        self.p_y_given_x = self.softmax.p_y_given_x
        self.y_pred = self.softmax.y_pred
        self.mean_log_likelihood = self.softmax.mean_log_likelihood
        self.errors = self.softmax.errors

        # self.kl_W = T.sum([layer.kl_W() for layer in self.layers])
        # self.kl_b = T.sum([layer.kl_b() for layer in self.layers])
        # self.kl = self.kl_W + self.kl_b

        self.effect_kl_W = T.sum([layer.effect_kl_W() for layer in self.layers])
        self.effect_kl_b = T.sum([layer.effect_kl_b() for layer in self.layers])
        self.effect_kl = self.effect_kl_W + self.effect_kl_b

    def get_param_dictionary(self):
        """
        Used to save model -- returns an array of all tunable parameters.
        """
        ret = {}
        for layer in self.layers:
            ret[layer.name] = layer.get_param_dictionary()

        return ret


def prepare_functions(model, X, index, y, X_test, X_train, y_train,
                      batch_size, l_rate):
    n_data_const = T.constant(
        X_train.shape[0].eval(), name='n_data', dtype=floatX
    )

    mean_log_likelihood = model.mean_log_likelihood(y)

    # scaled_kl_W = model.kl_W / n_data_const
    # scaled_kl_b = model.kl_b / n_data_const
    # scaled_kl = scaled_kl_W + scaled_kl_b
    effect_scaled_kl_W  = model.effect_kl_W / n_data_const
    effect_scaled_kl_b = model.effect_kl_b / n_data_const
    effect_scaled_kl = effect_scaled_kl_W + effect_scaled_kl_b

    # cost = -(mean_log_likelihood - scaled_kl)
    cost = -(mean_log_likelihood - effect_scaled_kl)

    params = model.params
    updates = adam(cost, params, learning_rate=l_rate)

    print('... compiling functions')

    # monitor cost and the individual components of it
    # outputs = [cost, mean_log_likelihood, scaled_kl_W, scaled_kl_b]
    outputs = (
        [cost, mean_log_likelihood, effect_scaled_kl_W, effect_scaled_kl_b]
    )

    train = theano.function(
        inputs=[index], outputs=outputs, updates=updates,
        givens={
            X: X_train[index * batch_size:(index + 1) * batch_size],
            y: y_train[index * batch_size:(index + 1) * batch_size]
        }
        # mode=NanGuardMode(nan_is_error=True, inf_is_error=True)
    )
    test_predict = theano.function(
        [index], model.p_y_given_x,
        givens={X: X_test[index * batch_size:(index + 1) * batch_size]}
        # mode=NanGuardMode(nan_is_error=True, inf_is_error=True)
    )

    return train, test_predict
