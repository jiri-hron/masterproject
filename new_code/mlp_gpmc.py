from __future__ import print_function

import pickle
from collections import OrderedDict

import numpy as np
import theano
import theano.tensor as T

# from theano.compile.nanguardmode import NanGuardMode
from lasagne.updates import adam  # , apply_nesterov_momentum

from layers import HiddenLayer, GaussLayer, StochLayer, SoftmaxLayer
from logistic_sgd import LogisticRegression
from utils import floatX_arr, eps

floatX = theano.config.floatX


class EnumGaussIO(object):
    NONE = 0
    INPUT = 1
    OUTPUT = 2
    IN_OUT = 3

    str2idx = {'NONE': NONE, 'INPUT': INPUT, 'OUTPUT': OUTPUT, 'IN_OUT': IN_OUT}
    idx2str = {idx: string for string, idx in str2idx.iteritems()}


class GPMC(object):
    """
    Implements four layer MLP-like architecture with softmax output
    """

    def __init__(self, input, n_in, n_out, n_hidden_layers,
                 n_hidden_lst, n_cat_lst,
                 M_lst=None, M_mask_lst=None,
                 A_lst=None, A_mask_lst=None,
                 m_lst=None, m_mask_lst=None,
                 sigma_W_params_lst=None, sigma_b_params_lst=None,
                 sigma_W=1e-3, sigma_b=1e-6,
                 tune_sigma_W_io=False, tune_sigma_b_io=False,
                 tune_sigma_W=True, tune_sigma_b=True,
                 l_W=1e-6, l_b=1e-6, A_scale=1.0, tune_A=True,
                 gauss_io=EnumGaussIO.IN_OUT, tie_weights=True,
                 classification=True, approx_cols=False,
                 output_bias_term=True, diag_noise=True,
                 dropout=False, dropout_prob=0.5, seed=None):

        assert n_hidden_layers > 0, 'n_layers must be positive'

        n_layers = n_hidden_layers + 1  # +1 for the output layer mapping

        M_lst = [None] * (n_layers) if M_lst is None else M_lst
        A_lst = [None] * (n_layers) if A_lst is None else A_lst
        M_mask_lst = [None] * (n_layers) if M_mask_lst is None else M_mask_lst
        A_mask_lst = [None] * (n_layers) if A_mask_lst is None else A_mask_lst
        m_lst = [None] * (n_layers) if m_lst is None else m_lst
        m_mask_lst = [None] * (n_layers) if m_mask_lst is None else m_mask_lst

        if sigma_W_params_lst is None:
            sigma_W_params_lst = [None] * (n_layers)
        if sigma_b_params_lst is None:
            sigma_b_params_lst = [None] * (n_layers)

        assert \
            len(n_hidden_lst) == len(n_cat_lst) == len(M_lst) == \
            len(M_mask_lst) == len(A_lst) == len(A_mask_lst) == \
            len(m_lst) == len(m_mask_lst) == len(sigma_W_params_lst) == \
            len(sigma_b_params_lst) == n_layers, \
            'length of all lists must be hte same and equal to ' \
            '(n_layers + 1) where the +1 is for the output layer mapping'

        # set seed to ensure each layer is init differently (cf. seed += 1)
        seed = np.random.randint(int(1e6)) if seed is None else seed
        np.random.seed(seed)

        def activation(x):
            return T.nnet.relu(x, alpha=0.0)
            # return T.nnet.relu(x, alpha=0.1)
            # return T.constant(np.sqrt(2), dtype=floatX) * T.cos(x)
            # return x
            # return (
            #     x +
            #     x**2 / T.constant(2.0, dtype=floatX)
            # #     # x**3 / T.constant(3.0, dtype=floatX) +
            # #     # x**4 / T.constant(4.0, dtype=floatX) +
            # #     # x**5 / T.constant(5.0, dtype=floatX) +
            # #     # x**6 / T.constant(6.0, dtype=floatX) +
            # #     # x**7 / T.constant(7.0, dtype=floatX) +
            # #     # x**8 / T.constant(8.0, dtype=floatX)
            # )
            # return T.tanh(x)
            # return T.nnet.sigmoid(x)
        # TODO: delete

        if gauss_io in (EnumGaussIO.INPUT, EnumGaussIO.IN_OUT):
            self.in_layer = GaussLayer(
                input=input, n_in=n_in, n_out=n_hidden_lst[0],
                M=M_lst[0], m=m_lst[0],
                sigma_W_params=sigma_W_params_lst[0],
                sigma_W=sigma_W, tune_sigma_W=tune_sigma_W_io,
                sigma_b_params=sigma_b_params_lst[0],
                sigma_b=sigma_b, tune_sigma_b=tune_sigma_b_io,
                l_W=l_W, l_b=l_b, activation=activation,
                approx_cols=approx_cols, diag_noise=diag_noise,
                name='h1', seed=seed
            )
        else:
            self.in_layer = StochLayer(
                input=input, n_in=n_in, n_out=n_hidden_lst[0],
                n_cat=n_cat_lst[0],
                M=M_lst[0], M_mask=M_mask_lst[0],
                A=A_lst[0], A_mask=A_mask_lst[0],
                m=m_lst[0], m_mask=m_mask_lst[0],
                sigma_W_params=sigma_W_params_lst[0],
                sigma_b_params=sigma_b_params_lst[0],
                sigma_W=sigma_W, tune_sigma_W=tune_sigma_W_io,
                sigma_b=sigma_b, tune_sigma_b=tune_sigma_b_io,
                W_len_scale=l_W, b_len_scale=l_b,
                A_scale=A_scale, tune_A=tune_A,
                activation=activation, tie_weights=tie_weights,
                approx_cols=approx_cols, diag_noise=diag_noise,
                dropout=dropout, dropout_prob=dropout_prob,
                name='h1', seed=seed
            )
            # the 1st layer corresponds to the kernel approx with n_out samples
            sqrt_n_out = T.constant(self.in_layer.n_out ** 0.5, dtype=floatX)
            self.in_layer.output /= sqrt_n_out
        self.layers = [self.in_layer]
        seed += 1  # seeds change so layers with same shape have diff init vals

        # the first hidden layer was already set-up above, hence the -1
        for i in xrange(1, n_layers - 1):
            prev_layer = self.layers[-1]
            layer = StochLayer(
                input=prev_layer.output, n_in=prev_layer.n_out,
                n_out=n_hidden_lst[i], n_cat=n_cat_lst[i],
                M=M_lst[i], M_mask=M_mask_lst[i],
                A=A_lst[i], A_mask=A_mask_lst[i],
                m=m_lst[i], m_mask=m_mask_lst[i],
                sigma_W_params=sigma_W_params_lst[i],
                sigma_b_params=sigma_b_params_lst[i],
                sigma_W=sigma_W, tune_sigma_W=tune_sigma_W,
                sigma_b=sigma_b, tune_sigma_b=tune_sigma_b,
                W_len_scale=l_W, b_len_scale=l_b,
                A_scale=A_scale, tune_A=tune_A,
                activation=activation, tie_weights=tie_weights,
                approx_cols=approx_cols, diag_noise=diag_noise,
                dropout=dropout, dropout_prob=dropout_prob,
                name='h' + str(i+1), seed=seed
            )
            self.layers += [layer]
            seed += 1

        # the output weights shouldn't be divided by sqrt(1/K), hence also no
        prev_layer = self.layers[-1]
        if gauss_io in (EnumGaussIO.OUTPUT, EnumGaussIO.IN_OUT):
            self.out_layer = GaussLayer(
                input=prev_layer.output,
                n_in=prev_layer.n_out, n_out=n_out,
                M=M_lst[-1], m=m_lst[-1],
                sigma_W_params=sigma_W_params_lst[-1],
                sigma_W=sigma_W, tune_sigma_W=tune_sigma_W_io,
                sigma_b_params=sigma_b_params_lst[-1],
                sigma_b=sigma_b, tune_sigma_b=tune_sigma_b_io,
                l_W=l_W, l_b=l_b, approx_cols=approx_cols,
                diag_noise=diag_noise, b_is_deterministic=True,
                bias_term=output_bias_term, name='out', seed=seed
            )
        else:
            self.out_layer = StochLayer(
                input=prev_layer.output,
                n_in=prev_layer.n_out, n_out=n_out, n_cat=n_cat_lst[-1],
                M=M_lst[-1], M_mask=M_mask_lst[-1],
                A=A_lst[-1], A_mask=A_mask_lst[-1],
                m=m_lst[-1], m_mask=m_lst[-1],
                sigma_W_params=sigma_W_params_lst[-1],
                sigma_b_params=sigma_b_params_lst[-1],
                sigma_W=sigma_W, tune_sigma_W=tune_sigma_W_io,
                sigma_b=sigma_b, tune_sigma_b=tune_sigma_W_io,
                A_scale=A_scale, tune_A=tune_A,
                W_len_scale=l_W, b_len_scale=l_b,
                tie_weights=tie_weights, approx_cols=approx_cols,
                diag_noise=diag_noise, b_is_deterministic=True,
                dropout=dropout, dropout_prob=dropout_prob,
                bias_term=output_bias_term, name='out', seed=seed
            )
        self.layers += [self.out_layer]
        # sigma_b must stay zero because the output bias vector is a parameter
        # of the generative model

        self.input = input
        self.approx_cols = approx_cols

        if classification:
            self.softmax_layer = SoftmaxLayer(
                input=self.out_layer.output
            )

            self.p_y_given_x = self.softmax_layer.p_y_given_x
            self.y_pred = self.softmax_layer.y_pred
            self.log_likelihood = self.softmax_layer.log_likelihood
            self.mean_log_likelihood = self.softmax_layer.mean_log_likelihood
            self.errors = self.softmax_layer.errors
        else:
            self.y_pred = self.out_layer.output.flatten()

        # # kl_M and kl_m were added to track them separetely
        # self.kl_W = T.sum([layer.kl_W() for layer in self.layers])
        # self.kl_b = T.sum([layer.kl_b() for layer in self.layers])
        # self.kl = self.kl_W + self.kl_b

        # kl_M and kl_m were added to track them separetely
        self.effect_kl_W = T.sum([layer.effect_kl_W() for layer in self.layers])
        self.effect_kl_b = T.sum([layer.effect_kl_b() for layer in self.layers])
        self.effect_kl = self.effect_kl_W + self.effect_kl_b

        # collect all parameters
        self.grad_params = reduce(
            lambda x, y: x + y, [layer.grad_params for layer in self.layers]
        )
        self.disc_params = reduce(
            lambda x, y: x + y, [layer.disc_params for layer in self.layers]
        )

        # collect all gradient functions for disconnected parameters
        self.disc_grads = dict(
            reduce(
                lambda x, y: x + y,
                [layer.disc_grads.items() for layer in self.layers]
            )
        )

        # collect all wrappers (mostly used to prevent updates of particular
        # parameters based on user supplied update masks)
        self.update_wrappers = dict(
            reduce(
                lambda x, y: x + y,
                [layer.update_wrappers.items() for layer in self.layers]
            )
        )

    def get_param_dictionary(self):
        """
        Used to save model -- returns an array of all tunable parameters.
        """
        ret = {}
        for layer in self.layers:
            ret[layer.name] = layer.get_param_dictionary()

        return ret


def prepare_log_reg(X, y):

    # fdir = os.path.dirname(os.path.realpath('__file__'))

    # load the saved weights
    # with open(os.path.join(fdir, 'log_reg_params.pkl'), 'rb') as f:
    with open('log_reg_params.pkl', 'rb') as f:
        params = pickle.load(f)

    n_in, n_out = params['W'].shape

    model = LogisticRegression(input=X, n_in=n_in, n_out=n_out,
                               W=params['W'], b=params['b'])

    return model


def prepare_functions(model, index, X, y, X_test, X_train, y_train,
                      l_rate, batch_size, n_train_samples, n_in,
                      ll_net_hidden, ll_track_decay, ll_add_log_reg=False,
                      classification=True, tune_tau=False,
                      tau=theano.shared(floatX_arr(1.0), name='tau')):

    n_data = X_train.shape[0].eval()
    n_data_const = T.constant(n_data, name='n_data', dtype=floatX)

    # define cost function
    if classification:
        # log_likelihood = model.log_likelihood(y)
        mean_log_likelihood = model.mean_log_likelihood(y)
    else:
        half = T.constant(0.5, dtype=floatX)
        two_pi = T.constant(2 * np.pi, dtype=floatX)
        d = T.constant(1.0, dtype=floatX) \
            if y.ndim == 1 else T.cast(y.shape[1], dtype=floatX)

        mean_log_likelihood = (
            -half * d * T.log(two_pi / tau + eps) -
            half * tau * T.mean(T.pow(y - model.y_pred, 2))
        )

    # scaled_kl_W = model.kl_W / n_data_const
    # scaled_kl_b = model.kl_b / n_data_const
    # scaled_kl = scaled_kl_W + scaled_kl_b
    effect_scaled_kl_W = model.effect_kl_W / n_data_const
    effect_scaled_kl_b = model.effect_kl_b / n_data_const
    effect_scaled_kl = effect_scaled_kl_W + effect_scaled_kl_b

    # cost = -(mean_log_likelihood - scaled_kl)
    cost = -(mean_log_likelihood - effect_scaled_kl)

    # set-up iteration counter
    t_prev = theano.shared(floatX_arr(0.0), name='t_prev')
    t = t_prev + 1

    # track the running mean of the mean log likelihood
    run_mean_ll = theano.shared(floatX_arr(0.0), name='run_mean_ll')

    # set up constant control variate
    ll_const_ctrl_var = run_mean_ll

    # set up model-based control variate
    nnet = OneLayerMLP(input=X, n_in=n_in, n_out=1, n_hidden=ll_net_hidden)

    ll_model_ctrl_var = T.mean(nnet.output)
    if ll_add_log_reg:
        assert classification

        log_reg = prepare_log_reg(X=X, y=y)
        ll_model_ctrl_var += log_reg.mean_log_likelihood(y)

    # the nnet has to be optimised
    cost_nnet = T.pow(
        -mean_log_likelihood - ll_const_ctrl_var - ll_model_ctrl_var, 2
    )
    # cost_nnet = T.mean(
    #     (log_likelihood  - ll_const_ctrl_var -
    #      nnet.output - ll_add_log_reg.log_likelihood(y)) ** 2
    # )
    grads_nnet = [
        (param, grad) for param, grad in
        zip(nnet.params, T.grad(cost_nnet, nnet.params))
        ]

    # collect parameters
    params = model.grad_params + model.disc_params + nnet.params

    # define gradients
    diff_grads = [
        (param, grad) for param, grad in
        zip(model.grad_params, T.grad(cost, model.grad_params))
        ]
    disc_grads = [
        (param, calc_grad(mean_log_likelihood=mean_log_likelihood,
                          n_data=n_data,
                          ll_const_ctrl_var=ll_const_ctrl_var,
                          ll_model_ctrl_var=ll_model_ctrl_var))
        for param, calc_grad in model.disc_grads.iteritems()
        ]

    if tune_tau:
        params += [tau]
        diff_grads += [(tau, T.grad(cost, tau))]

    grads = dict(diff_grads + disc_grads + grads_nnet)

    print('... compiling functions')

    # monitor cost and the individual components of it
    # outputs = [cost, mean_log_likelihood, scaled_kl_W, scaled_kl_b]
    outputs = (
        [cost, mean_log_likelihood, effect_scaled_kl_W, effect_scaled_kl_b]
    )

    # define the givens
    givens_train = {
        X: X_train[index * batch_size:(index + 1) * batch_size],
        y: y_train[index * batch_size:(index + 1) * batch_size]
    }

    assert n_train_samples > 0, 'n_train_samples non-positive'
    if n_train_samples == 1:
        # define all updates of parameters in one step
        updates = adam(grads.values(), grads.keys(), learning_rate=l_rate,
                       beta1=0.9, beta2=0.99)

        # single sample update for the running mean
        update_run_mean_ll = (
            ll_track_decay * run_mean_ll +
            (1 - ll_track_decay) * mean_log_likelihood
        )
        update_run_mean_ll /= (1 - ll_track_decay ** t)
        # correction for the bias of the running mean (cf. Adam paper)

        updates = (
            OrderedDict([(param, model.update_wrappers[param](updates[param]))
                         if param in model.update_wrappers.keys()
                         else (param, updates[param])
                         for param in updates.keys()] +
                        [(run_mean_ll, update_run_mean_ll), (t_prev, t)])
        )

        # define the train function
        train = theano.function(
            inputs=[index], outputs=outputs, updates=updates,
            givens=givens_train
            # mode=NanGuardMode(nan_is_error=True, inf_is_error=True)
        )
    else:
        # average of gradients is accumulated in 'accumulators' for each param
        accumulators = OrderedDict()
        updates_batch = OrderedDict()
        const_n_samples = T.constant(n_train_samples, name='n_train_samples',
                                     dtype=floatX)

        for param in params:
            val = param.get_value(borrow=True)
            acc = theano.shared(np.zeros(val.shape, dtype=val.dtype),
                                broadcastable=param.broadcastable,
                                name=param.name + '_acc')
            accumulators[param] = acc
            updates_batch[acc] = acc + grads[param] / const_n_samples

        # accumulate the mean of mean log-likelihoods
        acc_mean_ll = theano.shared(floatX_arr(0.0), name='mean_ll_acc')
        updates_batch[acc_mean_ll] = (
            acc_mean_ll + mean_log_likelihood / const_n_samples
        )

        # samples gradient and calculate statistics at given point
        train_batch = theano.function(
            inputs=[index], outputs=outputs, updates=updates_batch,
            givens=givens_train
            # mode=NanGuardMode(nan_is_error=True, inf_is_error=True)
        )

        # multi-sample update for the running mean
        update_run_mean_ll = (
            ll_track_decay * run_mean_ll + (1-ll_track_decay) * acc_mean_ll
        )
        update_run_mean_ll /= (1 - ll_track_decay**t)
        # correction for the bias of the running mean (cf. Adam paper)

        # calculate the parameter updates
        updates_update = adam(accumulators.values(), accumulators.keys(),
                              learning_rate=l_rate)
        updates_update = OrderedDict(
            [(param, model.update_wrappers[param](updates_update[param]))
             if param in model.update_wrappers.keys()
             else (param, updates_update[param])
             for param in updates_update.keys()] +
            [(run_mean_ll, update_run_mean_ll)] +
            [(accumulators[par], T.zeros_like(accumulators[par]))
             for par in accumulators.keys()] +
            [(acc_mean_ll, T.zeros_like(acc_mean_ll))] +
            # resets the values of accumulators at the end
            [(t_prev, t)]
        )

        # called after the avg. gradients were accumulated to update
        # the grad_params
        train_update = theano.function(
            inputs=[], outputs=[], updates=updates_update
            # mode=NanGuardMode(nan_is_error=True, inf_is_error=True)
        )

        # define the train function
        def train(idx):
            stats_batch = np.array(
                [train_batch(idx) for _ in xrange(n_train_samples)]
            )
            train_update()
            return np.mean(stats_batch, axis=0).tolist()

    # predict function
    if classification:
        test_predict = theano.function(
            [index], model.p_y_given_x,
            givens={X: X_test[index * batch_size:(index + 1) * batch_size]}
            # mode=NanGuardMode(nan_is_error=True, inf_is_error=True)
        )
    else:
        test_predict = theano.function(
            [index], model.y_pred,
            givens={X: X_test[index * batch_size:(index + 1) * batch_size]}
            # mode=NanGuardMode(nan_is_error=True, inf_is_error=True)
        )

    return train, test_predict


class OneLayerMLP(object):

    def __init__(self, input, n_in, n_out, n_hidden, seed=None):

        activation = lambda x: T.nnet.relu(x, alpha=0.01)

        self.hidden = HiddenLayer(input=input, n_in=n_in, n_out=n_hidden,
                                  activation=activation, seed=seed)
        self.linear = HiddenLayer(input=self.hidden.output, n_in=n_hidden,
                                  n_out=n_out, seed=seed)

        self.params = self.hidden.params + self.linear.params

        self.input = input
        self.output = self.linear.output.flatten()
