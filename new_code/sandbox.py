from __future__ import print_function

import os
import argparse
import pickle

import numpy as np
import theano
import theano.tensor as T
import matplotlib.pyplot as plt

from utils import prepare_mnist_data, train_model
from mlp_gpmc import EnumGaussIO, prepare_functions
from mlp_gpmc_tied import prepare_model

out_dir = os.getcwd()

floatX = theano.config.floatX


def softmax(x, axis=1):
    x_norm = x - np.max(x, axis=axis, keepdims=True)
    return np.exp(x_norm) / np.sum(np.exp(x_norm), axis=axis, keepdims=True)


def ard_stats_unique(A, const=0.99):
    pi = softmax(A)
    idx = np.max(pi, axis=1) >= const
    n_fixed = pi[idx].shape[0]
    n_used = len(set(np.argmax(pi[idx], axis=1)))
    prop_fixed_weights = 1.0 if pi.shape[0] == 0 else (
        (n_fixed / float(pi.shape[0]))
    )
    prop_used_unique = 1.0 if pi[idx].shape[0] == 0 else (
        (n_used / float(pi[idx].shape[0]))
    )
    print('number of weights with fixed cluster membership: %d' % n_fixed)
    print('weights used: %d' % n_used)
    print('prop. of the above: %f' % prop_fixed_weights)
    print('prop. used categories to the no. of fixed weights: %f' %
          prop_used_unique)

    return prop_fixed_weights, n_used, prop_used_unique


def ard_stats_all(A, const=0.01):
    pi = softmax(A)
    n_used = np.unique(np.nonzero(pi >= const)[1]).shape[0]
    proportion = n_used / float(pi.shape[1])
    print('total number of categories: %d' % pi.shape[1])
    print('categories used: %d' % n_used)
    print('proportion: %f' % proportion)

    return n_used, proportion


# # argmax vs. dot test
# def main():
#     k = 1000
#     t = 1500
#     d = 800
#
#     srng = MRG_RandomStreams()
#     srng.seed(1234)
#
#     Pi = np.asarray(np.random.dirichlet([1.0] * t, k), dtype=floatX)
#     Pi = theano.shared(Pi, name='Pi')
#     Z = T.as_tensor_variable(srng.multinomial(pvals=Pi, dtype=floatX), name='Z')
#     # Z = T.as_tensor_variable(srng.multinomial(pvals=Pi, dtype='uint16'), name='Z')
#
#     W = np.asarray(np.random.uniform(size=(t, d)), dtype=floatX)
#     W = theano.shared(W, name='W')
#
#     fn = theano.function([], T.mean(T.dot(Z, W)))
#     # fn = theano.function([], T.mean(W[T.argmax(Z, axis=1)]))
#
#     rst = 0
#     for _ in xrange(1000):
#         rst += fn()
#
#     print(fn())


def main(n_hidden_layers=2, n_hidden=100, n_cat_in=50, n_cat_hidden=50,
         n_iter=50, tune_sigma_W=True, tune_sigma_b=True,
         ll_track_decay=0.25, ll_net_hidden=256, ll_add_log_reg=False,
         diag_noise=True, approx_cols=True, l_rate=0.01, plot=False):

    batch_size = 256

    sigma_W = 1e-3
    sigma_b = 1e-6
    A_scale = 1.0
    tune_A = True

    gauss_io = EnumGaussIO.IN_OUT
    init_params_path = None
    seed = 1234

    test_at_iter = 10
    n_train_samples = 1
    n_test_samples = 10
    n_in = 28**2
    n_out = 10

    # load data
    X_train, y_train, X_test, y_test, n_train_batches, n_test_batches = (
        prepare_mnist_data(batch_size=batch_size)
    )

    # define symbolic variables
    index = T.lscalar('index')
    X = T.matrix('X', dtype=floatX)
    y = T.ivector('y')

    model = prepare_model(
        X=X, n_hidden_layers=n_hidden_layers, n_in=n_in, n_out=n_out,
        n_hidden=n_hidden, n_cat_hidden=n_cat_hidden, n_cat_in=n_cat_in,
        sigma_W=sigma_W, sigma_b=sigma_b,
        tune_sigma_W=tune_sigma_W, tune_sigma_b=tune_sigma_b,
        diag_noise=diag_noise, A_scale=A_scale, tune_A=tune_A,
        tune_noise_io=False, only_init_weights=True,
        approx_cols=approx_cols, gauss_io=gauss_io,
        init_params_path=init_params_path, seed=seed
    )

    # compile theano functions
    train, test_predict = prepare_functions(
        model=model, index=index, X=X, y=y, n_in=n_in,
        X_train=X_train, X_test=X_test, y_train=y_train,
        l_rate=l_rate, batch_size=batch_size,
        n_train_samples=n_train_samples,
        ll_track_decay=ll_track_decay, ll_net_hidden=ll_net_hidden,
        ll_add_log_reg=ll_add_log_reg
    )

    # train the model
    cost_stats, ll_stats, kl_W_stats, kl_b_stats, test_err_stats = train_model(
        train=train, test_predict=test_predict, y_test=y_test,
        batch_size=batch_size, test_at_iter=test_at_iter,
        n_iter=n_iter, n_test_batches=n_test_batches,
        n_test_samples=n_test_samples, n_train_batches=n_train_batches
    )

    fname = (
        'gpmc_sandbox_'
        '%dn_%dk_%df_%dt_%di_%sw_%sb_%sd_%sc_%.2el_%.2ey_%de_%sr'
        '.save' %
        (n_hidden_layers, n_hidden, n_cat_in, n_cat_hidden, n_iter,
        str(tune_sigma_W), str(tune_sigma_b), str(diag_noise), str(approx_cols),
        l_rate, ll_track_decay, ll_net_hidden, str(ll_add_log_reg))
    )

    out_file = os.path.join(out_dir, fname)
    with open(out_file, 'wb') as f:
        pickle.dump(model.get_param_dictionary(), f)
        pickle.dump([
            ('cost', cost_stats), ('ll', ll_stats),
            ('kl_W', kl_W_stats), ('kl_b', kl_b_stats),
            ('error', test_err_stats)
        ], f)
    if plot:
        fig, ax = plt.subplots(2, 2)

        ax[0, 0].plot(cost_stats)
        ax[0, 0].set_title('cost')
        ax[0, 1].plot(ll_stats)
        ax[0, 1].set_title('mean log likelihood')
        ax[1, 0].plot(kl_W_stats)
        ax[1, 0].set_title('1/N KL(q(W) || p(W)) + C')
        ax[1, 1].plot(kl_b_stats)
        ax[1, 1].set_title('1/N KL(q(b) || p(b)) + C')

        fig, ax = plt.subplots(len(model.layers), 2)

        for i, layer in enumerate(model.layers):
            params = layer.get_param_dictionary()
            ax[i, 0].hist(params['sigma_W_params'].flatten() ** 2)
            ax[i, 0].set_title('h%d sigma_W' % (i+1))
            ax[i, 1].hist(params['sigma_b_params'].flatten() ** 2)
            ax[i, 1].set_title('h%d sigma_b' % (i+1))

        As = [layer.get_param_dictionary()['A'] for layer in model.layers
              if layer.__class__.__name__ == 'StochLayer']

        prop_fixed = np.zeros(len(As))
        prop_stat_unique = np.zeros(len(As))
        prop_stat_all = np.zeros(len(As))

        fig, ax = plt.subplots(1, len(As))

        for i, A in enumerate(As):
            prop_fixed[i], _, prop_stat_unique[i] = ard_stats_unique(A)
            _, prop_stat_all[i] = ard_stats_all(A)

            pi = softmax(A, axis=1)

            if len(As) == 1:
                ax.hist(np.sum(pi >= 0.01, axis=1))
            else:
                ax[i].hist(np.sum(pi >= 0.01, axis=1))

        plt.show()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Run simulation of GPMC approximation with tied weights'
    )

    parser.add_argument('-n', '--n_hidden_layers', type=int, required=True)
    parser.add_argument('-k', '--n_hidden', type=int, required=True)
    parser.add_argument('-f', '--n_cat_in', type=int, required=True)
    parser.add_argument('-t', '--n_cat_hidden', type=int, required=True)
    parser.add_argument('-i', '--n_iter', type=int, required=True)
    parser.add_argument('-w', '--tune_sigma_w', type=str, required=True)
    parser.add_argument('-b', '--tune_sigma_b', type=str, required=True)
    parser.add_argument('-d', '--diag_noise', type=str, required=True)
    parser.add_argument('-c', '--approx_cols', type=str, required=True)
    parser.add_argument('-l', '--l_rate', type=float, required=True)
    parser.add_argument('-y', '--ll_track_decay', type=float, required=True)
    parser.add_argument('-e', '--ll_net_hidden', type=int, required=True)
    parser.add_argument('-r', '--ll_log_reg', type=str, required=True)
    parser.add_argument('-p', '--plot', type=str, required=True)

    args = parser.parse_args()

    assert args.tune_sigma_w is None or args.tune_sigma_w in ('True', 'False')
    assert args.tune_sigma_b is None or args.tune_sigma_b in ('True', 'False')
    assert args.approx_cols is None or args.approx_cols in ('True', 'False')
    assert args.diag_noise is None or args.diag_noise in ('True', 'False')
    assert args.ll_log_reg is None or args.ll_log_reg in ('True', 'False')
    assert args.plot is None or args.plot in ('True', 'False')

    n_hidden_layers = args.n_hidden_layers
    n_hidden = args.n_hidden
    n_cat_in = args.n_cat_in
    n_cat_hidden = args.n_cat_hidden
    n_iter = args.n_iter
    tune_sigma_W = args.tune_sigma_w == 'True'
    tune_sigma_b = args.tune_sigma_b == 'True'
    diag_noise = args.diag_noise == 'True'
    approx_cols = args.approx_cols == 'True'
    l_rate = args.l_rate
    ll_track_decay = args.ll_track_decay
    ll_net_hidden = args.ll_net_hidden
    ll_add_log_reg = args.ll_log_reg == 'True'
    plot = args.plot == 'True'

    main(n_hidden_layers=n_hidden_layers, n_hidden=n_hidden,
         n_cat_in=n_cat_in, n_cat_hidden=n_cat_hidden, n_iter=n_iter,
         tune_sigma_W=tune_sigma_W, tune_sigma_b=tune_sigma_b,
         diag_noise=diag_noise, approx_cols=approx_cols, l_rate=l_rate,
         ll_track_decay=ll_track_decay, ll_net_hidden=ll_net_hidden,
         ll_add_log_reg=ll_add_log_reg, plot=plot)
