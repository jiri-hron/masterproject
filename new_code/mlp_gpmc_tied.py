from __future__ import print_function

import os
import argparse
import pickle
import re
import matplotlib.pyplot as plt

import numpy as np
import theano
import theano.tensor as T

from itertools import compress
from utils import prepare_mnist_data, prepare_cifar10_data, train_model
from mlp_gpmc import GPMC, EnumGaussIO, prepare_functions

out_dir = os.getcwd()
# os.chdir('/home/ucabjh1/gitHub/machineLearning/bayesian_dp/mog_approx/code')

floatX = theano.config.floatX


def prepare_model(X, n_in, n_out,
                  n_hidden, n_hidden_layers, n_cat_in, n_cat_hidden,
                  A_scale, tune_A, sigma_W, sigma_b, tune_sigma_W, tune_sigma_b,
                  diag_noise, tune_noise_io, gauss_io, approx_cols,
                  init_params_path, only_init_weights, seed):

    print('... building the model')

    # for now, the number of units is same for all hidden layers
    n_hidden_lst = [n_hidden] * (n_hidden_layers + 1)

    # the number of hidden units is same for but the 1st layer, and we are
    # approximating rows of the weight matrices which means their number is
    # dependent on number of inputs to the layer; the only layer with different
    # number of inputs is thus the first hidden, therefore it might need to
    # have a different number of inputs
    n_cat_lst = [n_cat_in] + [n_cat_hidden] * n_hidden_layers

    if init_params_path is not None:
        with open(init_params_path, 'rb') as f:
            params = pickle.load(f)

        M_lst = []
        m_lst = []
        sigma_W_params_lst = []
        sigma_b_params_lst = []

        # acess by name to prevent changes of order
        keys = ['h' + str(i) for i in xrange(1,n_hidden_layers+1)] + ['out']
        assert set(keys) == set(params.keys()), 'invalid grad_params supplied'
        for i, key in enumerate(keys):
            layer_params = params[key]

            M = layer_params['M']
            m = layer_params['m']
            sigma_W_params = layer_params['sigma_W_params']
            sigma_b_params = layer_params['sigma_b_params']

            t = n_cat_lst[i]
            if key == 'h1' and gauss_io in (EnumGaussIO.IN_OUT,
                                            EnumGaussIO.INPUT):
                t = n_hidden_lst[0] if approx_cols else n_in
            if key == 'out' and gauss_io in (EnumGaussIO.IN_OUT,
                                             EnumGaussIO.OUTPUT):
                t = n_out if approx_cols else n_hidden_lst[-1]

            if M.shape[0] > t:
                # will be the same for sigma_W_params
                idx = np.random.choice(range(M.shape[0]), t, replace=False)
                M = M[idx]
                sigma_W_params = sigma_W_params[idx]

            if m.shape[0] > n_hidden_lst[i]:
                idx = np.random.choice(range(m.shape[0]), n_hidden_lst[i],
                                       replace=False)
                m = m[idx]

            M_lst += [M]
            m_lst += [m]
            sigma_W_params_lst += [sigma_W_params]
            sigma_b_params_lst += [sigma_b_params]
    else:
        M_lst = None
        m_lst = None
        sigma_W_params_lst = None
        sigma_b_params_lst = None

    if only_init_weights:
        sigma_W_params_lst = None
        sigma_b_params_lst = None

    model = GPMC(
        input=X, n_in=n_in, n_out=n_out, n_hidden_layers=n_hidden_layers,
        n_hidden_lst=n_hidden_lst, n_cat_lst=n_cat_lst,
        M_lst=M_lst, m_lst=m_lst, sigma_W_params_lst=sigma_W_params_lst,
        sigma_b_params_lst=sigma_b_params_lst,
        sigma_W=sigma_W, sigma_b=sigma_b,
        tune_sigma_W=tune_sigma_W, tune_sigma_b=tune_sigma_b,
        tune_sigma_W_io=tune_noise_io, tune_sigma_b_io=tune_noise_io,
        l_W=1e-6, l_b=1e-6, diag_noise=diag_noise,
        A_scale=A_scale, tune_A=tune_A, gauss_io=gauss_io,
        tie_weights=True, approx_cols=approx_cols, seed=seed
    )

    return model


def test_mlp(l_rate=0.01, n_iter=20, batch_size=256,
             n_hidden_layers=4, n_hidden=512, n_cat_in=28**2, n_cat_hidden=512,
             n_train_samples=1, n_test_samples=10, test_at_iter=5,
             sigma_W=1e-3, sigma_b=1e-6, tune_sigma_W=True, tune_sigma_b=True,
             diag_noise=True, tune_noise_io=True, A_scale=1.0, tune_A=True,
             gauss_io=EnumGaussIO.IN_OUT, approx_cols=False,
             ll_track_decay=0.25, ll_net_hidden=256, only_init_weights=True,
             init_params_path=None, plot=False, seed=1234, mnist=True):

    # load data
    if mnist:
        n_in = 28 * 28
        n_out = 10

        X_train, y_train, X_test, y_test, n_train_batches, n_test_batches = (
            prepare_mnist_data(batch_size=batch_size)
        )
    else:  # CIFAR-10
        n_in = 3 * 32 * 32
        n_out = 10

        X_train, y_train, X_test, y_test, n_train_batches, n_test_batches = (
            prepare_cifar10_data(batch_size=batch_size)
        )

    # define symbolic variables
    index = T.lscalar('index')
    X = T.matrix('X', dtype=floatX)
    y = T.ivector('y')

    # create model
    model = prepare_model(
        X=X, n_in=n_in, n_out=n_out, n_hidden_layers=n_hidden_layers,
        n_hidden=n_hidden, n_cat_hidden=n_cat_hidden, n_cat_in=n_cat_in,
        sigma_W=sigma_W, sigma_b=sigma_b,
        tune_sigma_W=tune_sigma_W, tune_sigma_b=tune_sigma_b,
        diag_noise=diag_noise, tune_noise_io=tune_noise_io, gauss_io=gauss_io,
        A_scale=A_scale, tune_A=tune_A, approx_cols=approx_cols,
        init_params_path=init_params_path, only_init_weights=only_init_weights,
        seed=seed
    )

    # compile theano functions
    train, test_predict = prepare_functions(
        model=model, index=index, X=X, y=y, n_in=n_in,
        X_train=X_train, X_test=X_test, y_train=y_train,
        l_rate=l_rate, batch_size=batch_size,
        n_train_samples=n_train_samples,
        ll_track_decay=ll_track_decay,
        ll_net_hidden=ll_net_hidden
    )

    # train the model
    cost_stats, ll_stats, kl_W_stats, kl_b_stats, test_err_stats = train_model(
        train=train, test_predict=test_predict, y_test=y_test,
        batch_size=batch_size, test_at_iter=test_at_iter,
        n_iter=n_iter, n_test_batches=n_test_batches,
        n_test_samples=n_test_samples, n_train_batches=n_train_batches
    )

    fname = (
        'gpmc_mnist_%dh_%dc1_%dch_%ds_%.2elr_gaussIO_%s_tuneA_%s'
        '_cols_%s_diag_%s.save' %
        (n_hidden, n_cat_in, n_cat_hidden, n_train_samples, l_rate,
         EnumGaussIO.idx2str[gauss_io], str(tune_A), str(approx_cols),
         str(diag_noise))
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

        plt.show()


def main():

    parser = argparse.ArgumentParser(
        description='Run simulation of GPMC approximation with tied weights'
    )

    parser.add_argument('-n', '--n_hidden_layers', type=int, required=False)
    parser.add_argument('-k', '--n_hidden', type=int, required=False)
    parser.add_argument('-f', '--n_cat_in', type=int, required=False)
    parser.add_argument('-t', '--n_cat_hidden', type=int, required=False)
    parser.add_argument('-d', '--diag_noise', type=str, required=False)
    parser.add_argument('-o', '--tune_noise_io', type=str, required=False)
    parser.add_argument('-w', '--sigma_w', type=float, required=False)
    parser.add_argument('-b', '--sigma_b', type=float, required=False)
    parser.add_argument('-m', '--tune_sigma_w', type=str, required=False)
    parser.add_argument('-e', '--tune_sigma_b', type=str, required=False)
    parser.add_argument('-s', '--n_train_samples', type=int, required=False)
    parser.add_argument('-l', '--l_rate', type=float, required=False)
    parser.add_argument('-i', '--n_iter', type=int, required=False)
    parser.add_argument('-g', '--gauss_io', type=str, required=False)
    parser.add_argument('-c', '--approx_cols', type=str, required=False)
    parser.add_argument('-a', '--tune_A', type=str, required=False)
    parser.add_argument('-p', '--plot', type=str, required=False)
    parser.add_argument('-z', '--init_params_path', type=str, required=False)
    parser.add_argument('-q', '--only_init_weights', type=str, required=False)
    parser.add_argument('-j', '--mnist', type=str, required=False)

    args = parser.parse_args()

    assert args.gauss_io is None or args.gauss_io in EnumGaussIO.str2idx.keys()
    assert args.approx_cols is None or args.approx_cols in ('True', 'False')
    assert args.tune_A is None or args.tune_A in ('True', 'False')
    assert args.plot is None or args.plot in ('True', 'False')
    assert args.diag_noise is None or args.diag_noise in ('True', 'False')
    assert args.tune_noise_io is None or args.tune_noise_io in ('True', 'False')
    assert args.tune_sigma_w is None or args.tune_sigma_w in ('True', 'False')
    assert args.tune_sigma_b is None or args.tune_sigma_b in ('True', 'False')
    assert args.mnist is None or args.mnist in ('True', 'False')
    assert args.only_init_weights is None or \
           args.only_init_weights in ('True', 'False')

    n_hidden = 512 if args.n_hidden is None else args.n_hidden
    n_cat_in = 28**2 if args.n_cat_in is None else args.n_cat_in
    n_cat_hidden = 512 if args.n_cat_hidden is None else args.n_cat_hidden
    diag_noise = True if args.diag_noise is None else args.diag_noise == 'True'
    sigma_W = 1e-3 if args.sigma_w is None else args.sigma_w
    sigma_b = 1e-6 if args.sigma_b is None else args.sigma_b
    l_rate = 0.01 if args.l_rate is None else args.l_rate
    n_iter = 20 if args.n_iter is None else args.n_iter
    tune_A = True if args.tune_A is None else args.tune_A == 'True'
    plot = False if args.plot is None else args.plot == 'True'
    mnist = True if args.mnist is None else args.mnist == 'True'
    n_hidden_layers = 4 \
        if args.n_hidden_layers is None else args.n_hidden_layers
    n_train_samples = 1 \
        if args.n_train_samples is None else args.n_train_samples
    gauss_io = EnumGaussIO.IN_OUT \
        if args.gauss_io is None else EnumGaussIO.str2idx[args.gauss_io]
    tune_noise_io = False \
        if args.tune_noise_io is None else args.tune_noise_io == 'True'
    approx_cols = False if args.approx_cols is None \
        else args.approx_cols == 'True'
    init_params_path = None \
        if args.init_params_path is None else args.init_params_path
    tune_sigma_W = True \
        if args.tune_sigma_w is None else args.tune_sigma_w == 'True'
    tune_sigma_b = True \
        if args.tune_sigma_b is None else args.tune_sigma_b == 'True'
    only_init_weights = True \
        if args.only_init_weights is None else args.only_init_weights

    # automatically pick an appropriate file for initialisation if a directory
    # containing saved weights was supplied
    if (init_params_path is not None) and os.path.isdir(init_params_path):
        fnames = os.listdir(init_params_path)
        fnames = filter(lambda x: x.endswith('.save'), fnames)

        candidates = {}
        for fname in fnames:
            split_fname = fname.split('_')
            split_fname[-1] = split_fname[-1][:split_fname[-1].index('.save')]

            tmp_n_hidden_layers = int(
                filter(lambda x: re.match(r'[0-9]+n', x), split_fname)[0][:-1]
            )
            tmp_n_units_in = int(
                filter(lambda x: re.match(r'[0-9]+u', x), split_fname)[0][:-1]
            )
            tmp_n_units_hidden = int(
                filter(lambda x: re.match(r'[0-9]+k', x), split_fname)[0][:-1]
            )
            tmp_lrate = float(
                filter(lambda x: re.match(r'\d\.\d{2}e-\d{2}l', x),
                       split_fname)[0][:-1]
            )
            tmp_approx_cols = (
                split_fname[split_fname.index('cols') + 1] == 'True'
            )
            tmp_diag_noise = (
                split_fname[split_fname.index('diag') + 1] == 'True'
            )

            bool_cond = (
                tmp_n_hidden_layers == n_hidden_layers and
                (
                    (tmp_n_units_in >= n_cat_in and approx_cols) or
                    (tmp_n_units_hidden >= n_cat_in and not approx_cols)
                ) and
                tmp_n_units_hidden >= n_cat_hidden and
                tmp_approx_cols == approx_cols and
                tmp_n_units_in == tmp_n_units_hidden == n_hidden and
                tmp_diag_noise == diag_noise
            )
            if bool_cond:
                candidates[fname] = {
                    'u': tmp_n_units_in,
                    'k': tmp_n_units_hidden,
                    'l': tmp_lrate
                }

        if len(candidates) == 0:
            raise RuntimeError(
                'init_params_path is a directory but does not contain '
                'a file with suitable parameter set'
            )
        else:  # try to pick the smallest possible model
            dict_sum_units = dict(
                map(
                    lambda (k, v): (k, v['u'] + v['k']),
                    candidates.iteritems()
                )
            )
            min_sum = min(dict_sum_units.values())
            idx_min = [v == min_sum for v in dict_sum_units.values()]
            final_candidates = {
                k: v
                for (k, v) in candidates.iteritems()
                if k in compress(dict_sum_units.keys(), idx_min)
                }

            fname = final_candidates.keys()[0]

            # if there's more than one candiate, pick the one with lowest
            # learning rate (if still more than one, pick at random)
            if len(final_candidates) > 1:
                min_lrate = min([v['l'] for v in final_candidates.itervalues()])
                final_candidates = dict(filter(
                    lambda (k, v): v['l'] == min_lrate,
                    final_candidates.iteritems()
                ))
                idx_picked = np.random.choice(len(final_candidates), 1)[0]
                fname = final_candidates.keys()[idx_picked]

            # update the path to the file
            init_params_path = (
                os.path.join(init_params_path, fname)
            )

    print('Running GPMC approximation with tied weights')
    print('n_hidden_layers: %d' % n_hidden_layers)
    print('n_hidden: %d' % n_hidden)
    print('n_cat_in: %d' % n_cat_in)
    print('n_cat_hidden: %d' % n_cat_hidden)
    print('diag_noise: %s' % str(diag_noise))
    print('tune_noise_io: %s' % str(tune_noise_io))
    print('sigma_W: %.2e' % sigma_W)
    print('sigma_b: %.2e' % sigma_b)
    print('tune_sigma_W: %s' % str(tune_sigma_W))
    print('tune_sigma_b: %s' % str(tune_sigma_b))
    print('n_train_samples: %d' % n_train_samples)
    print('l_rate: %.2e' % l_rate)
    print('n_iter: %d' % n_iter)
    print('gauss_io: %s' % EnumGaussIO.idx2str[gauss_io])
    print('approx_cols: %s' % str(approx_cols))
    print('tune_A: %s' % str(tune_A))
    print('plot: %s' % str(plot))
    print('init_params_path: %s' % str(init_params_path))
    print('only_init_weights: %s' % str(only_init_weights))
    print('mnist: %s' % str(mnist))

    test_mlp(n_iter=n_iter, l_rate=l_rate,
             n_hidden_layers=n_hidden_layers, n_hidden=n_hidden,
             n_cat_in=n_cat_in, n_cat_hidden=n_cat_hidden,
             sigma_W=sigma_W, sigma_b=sigma_b, diag_noise=diag_noise,
             tune_sigma_W=tune_sigma_W, tune_sigma_b=tune_sigma_b,
             tune_A=tune_A, tune_noise_io=tune_noise_io, gauss_io=gauss_io,
             n_train_samples=n_train_samples, approx_cols=approx_cols,
             init_params_path=init_params_path,
             only_init_weights=only_init_weights,
             plot=plot, mnist=mnist)

if __name__ == "__main__":
    main()
