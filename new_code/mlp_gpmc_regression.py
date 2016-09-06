from __future__ import print_function

import os
import argparse
import pickle
import matplotlib.pyplot as plt

import numpy as np
import theano
import theano.tensor as T

from utils import prepare_toy_data, prepare_co2_data,  train_model, floatX_arr
from mlp_gpmc import GPMC, EnumGaussIO, prepare_functions

out_dir = os.getcwd()
# os.chdir('/home/ucabjh1/gitHub/machineLearning/bayesian_dp/mog_approx/code')

floatX = theano.config.floatX


def prepare_model(X, n_in, n_out, n_hidden, n_hidden_layers,
                  n_cat_in, n_cat_hidden, A_scale, tune_A,
                  sigma_W, sigma_b,  tune_sigma_W, tune_sigma_b,
                  l_W, l_b, diag_noise, tune_noise_io, gauss_io,
                  tie_weights, approx_cols, output_bias_term, seed):

    print('... building the model')

    # for now, the number of units is same for all hidden layers
    n_hidden_lst = [n_hidden] * (n_hidden_layers + 1)

    # the number of hidden units is same for but the 1st layer, and we are
    # approximating rows of the weight matrices which means their number is
    # dependent on number of inputs to the layer; the only layer with different
    # number of inputs is thus the first hidden, therefore it might need to
    # have a different number of inputs
    n_cat_lst = [n_cat_in] + [n_cat_hidden] * n_hidden_layers

    model = GPMC(
        input=X, n_in=n_in, n_out=n_out, n_hidden_layers=n_hidden_layers,
        n_hidden_lst=n_hidden_lst, n_cat_lst=n_cat_lst,
        sigma_W=sigma_W, sigma_b=sigma_b,
        tune_sigma_W=tune_sigma_W, tune_sigma_b=tune_sigma_b,
        tune_sigma_W_io=tune_noise_io, tune_sigma_b_io=tune_noise_io,
        l_W=l_W, l_b=l_b, diag_noise=diag_noise,
        A_scale=A_scale, tune_A=tune_A, gauss_io=gauss_io,
        tie_weights=tie_weights, approx_cols=approx_cols,
        classification=False, output_bias_term=output_bias_term, seed=seed
    )

    return model


def test_mlp(tie_weights=True, l_rate=0.01, n_iter=20, batch_size=256,
             n_hidden_layers=2, n_hidden=32, n_cat_in=32, n_cat_hidden=32,
             n_train_samples=1, n_test_samples=10, test_at_iter=5,
             sigma_W=1e-3, sigma_b=1e-6, tune_sigma_W=False, tune_sigma_b=False,
             diag_noise=False, tune_noise_io=False, A_scale=1.0, tune_A=True,
             gauss_io=EnumGaussIO.IN_OUT, approx_cols=True, l_W=1e-0, l_b=1e-0,
             ll_track_decay=0.25, ll_net_hidden=256, tau=400.0, tune_tau=False,
             output_bias_term=True, seed=1234, toy=True,
             plot=False, n_plot_samples=100):

    # load data
    if toy:
        n_in = 1
        n_out = 1
        n_train_points = 10000
        n_valid_points = 1000

        X_train, y_train, X_test, y_test, n_train_batches, n_test_batches = (
            prepare_toy_data(batch_size=batch_size, n_train=n_train_points,
                             n_valid=n_valid_points)
        )
    else:
        n_in = 1
        n_out = 1

        X_train, y_train, X_test, y_test, _, _, \
        n_train_batches, n_test_batches, _= (
            prepare_co2_data(batch_size=batch_size)
        )

    # define symbolic variables
    X = T.matrix('X', dtype=floatX)
    y = T.vector('y', dtype=floatX)
    index = T.lscalar('index')
    tau = theano.shared(floatX_arr(tau), name='tau')

    # create model
    model = prepare_model(
        X=X, n_in=n_in, n_out=n_out, n_hidden_layers=n_hidden_layers,
        n_hidden=n_hidden, n_cat_hidden=n_cat_hidden, n_cat_in=n_cat_in,
        l_W=l_W, l_b=l_b, sigma_W=sigma_W, sigma_b=sigma_b,
        tune_sigma_W=tune_sigma_W, tune_sigma_b=tune_sigma_b,
        diag_noise=diag_noise, tune_noise_io=tune_noise_io, gauss_io=gauss_io,
        A_scale=A_scale, tune_A=tune_A, approx_cols=approx_cols,
        tie_weights=tie_weights, output_bias_term=output_bias_term, seed=seed
    )
    # compile theano functions
    train, test_predict = prepare_functions(
        model=model, index=index, X=X, y=y, n_in=n_in,
        X_train=X_train, X_test=X_test, y_train=y_train,
        l_rate=l_rate, batch_size=batch_size,
        n_train_samples=n_train_samples,
        ll_track_decay=ll_track_decay, ll_net_hidden=ll_net_hidden,
        classification=False, tau=tau, tune_tau=tune_tau
    )

    # predict
    train_plot = theano.function(
        inputs=[], outputs=model.y_pred,
        givens={X: X_train}
    )

    test_plot = theano.function(
        inputs=[], outputs=model.y_pred,
        givens={X: X_test}
    )

    # train the model
    cost_stats, ll_stats, kl_W_stats, kl_b_stats, test_err_stats = train_model(
        train=train, test_predict=test_predict, y_test=y_test,
        batch_size=batch_size, test_at_iter=test_at_iter,
        n_iter=n_iter, n_test_batches=n_test_batches,
        n_test_samples=n_test_samples, n_train_batches=n_train_batches,
        classification=False
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
        pickle.dump(
            dict(
                model.get_param_dictionary().items() +
                [('tau', np.array(tau.eval()))]
            ), f
        )
        pickle.dump([
            ('cost', cost_stats), ('ll', ll_stats),
            ('kl_W', kl_W_stats), ('kl_b', kl_b_stats),
            ('error', test_err_stats)
        ], f)

    if plot:
        tau_val = float(tau.get_value())

        samples_train = np.array([train_plot() for _ in xrange(n_plot_samples)])
        samples_test = np.array([test_plot() for _ in xrange(n_plot_samples)])

        pred_1st_mmnt_train = np.mean(samples_train, axis=0)
        pred_2nd_mmnt_train = np.mean(samples_train ** 2, axis=0)
        pred_1st_mmnt_test = np.mean(samples_test, axis=0)
        pred_2nd_mmnt_test = np.mean(samples_test ** 2, axis=0)

        pred_mean_train = pred_1st_mmnt_train
        pred_var_train = (
            tau_val**(-1) + pred_2nd_mmnt_train - pred_1st_mmnt_train ** 2
        )
        pred_mean_test = pred_1st_mmnt_test
        pred_var_test = (
            tau_val ** (-1) + pred_2nd_mmnt_test - pred_1st_mmnt_test ** 2
        )

        xx_train = np.array(X_train.eval()).flatten()
        idx_sort_train = np.argsort(xx_train)
        xx_train = xx_train[idx_sort_train]
        yy_train = np.array(y_train.eval())[idx_sort_train]
        pred_mean_train = pred_mean_train[idx_sort_train]
        pred_var_train = pred_var_train[idx_sort_train]

        xx_test = np.array(X_test.eval().flatten())
        idx_sort_test = np.argsort(xx_test)
        xx_test = xx_test[idx_sort_test]
        yy_test = y_test[idx_sort_test]  # y_test is already an nd.array
        pred_mean_test = pred_mean_test[idx_sort_test]
        pred_var_test = pred_var_test[idx_sort_test]

        # fig, ax = plt.subplots()
        #
        # ax.errorbar(xx_train, pred_mean_train, yerr=pred_var_train,
        #                ecolor='#dfdce3', elinewidth=3)
        # ax.plot(xx_train, yy_train, '.', alpha=0.3, color='#00c07f')
        # # ax.set_title('train')

        fig, ax = plt.subplots()

        # ax.errorbar(xx_test, pred_mean_test, yerr=pred_var_test,
        #             ecolor='#dfdce3', elinewidth=3)
        ax.plot(xx_test, pred_mean_test, color='#00303f', linewidth=5)
        ax.plot(xx_test, yy_test, '.', alpha=0.3, color='#00c07f')
        # ax.axvline(x=-0.5, color='#cd5554', linestyle='--')
        # ax.axvline(x=0.0, color='#cd5554', linestyle='--')
        ax.axvline(x=0.5, color='#cd5554', linestyle='--')
        # ax.annotate('overlap', xy=(0.085, -0.05), xycoords='axes fraction')
        ax.annotate('interpolation', xy=(0.325, -0.05), xycoords='axes fraction')
        # ax.annotate('overlap', xy=(0.6, -0.05), xycoords='axes fraction')
        ax.annotate('extrapolation', xy=(0.835, -0.05), xycoords='axes fraction')
        # ax.set_title('test')

        fig, ax = plt.subplots(2, 2)

        ax[0, 0].plot(cost_stats)
        ax[0, 0].set_title('cost')
        ax[0, 1].plot(ll_stats)
        ax[0, 1].set_title('mean log likelihood')
        ax[1, 0].plot(kl_W_stats)
        ax[1, 0].set_title('1/N KL(q(W) || p(W)) + C')
        ax[1, 1].plot(kl_b_stats)
        ax[1, 1].set_title('1/N KL(q(b) || p(b)) + C')

        print('final tau: %f' % tau_val)

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
    parser.add_argument('-q', '--tau', type=float, required=False)
    parser.add_argument('-z', '--tune_tau', type=str, required=False)
    parser.add_argument('-y', '--tie_weights', type=str, required=False)

    args = parser.parse_args()

    assert args.gauss_io is None or args.gauss_io in EnumGaussIO.str2idx.keys()
    assert args.approx_cols is None or args.approx_cols in ('True', 'False')
    assert args.tune_A is None or args.tune_A in ('True', 'False')
    assert args.plot is None or args.plot in ('True', 'False')
    assert args.diag_noise is None or args.diag_noise in ('True', 'False')
    assert args.tune_noise_io is None or args.tune_noise_io in ('True', 'False')
    assert args.tune_sigma_w is None or args.tune_sigma_w in ('True', 'False')
    assert args.tune_sigma_b is None or args.tune_sigma_b in ('True', 'False')
    assert args.tune_tau is None or args.tune_tau in ('True', 'False')
    assert args.tie_weights is None or args.tie_weights in ('True', 'False')

    n_hidden = 100 if args.n_hidden is None else args.n_hidden
    n_cat_in = 1 if args.n_cat_in is None else args.n_cat_in
    n_cat_hidden = 1 if args.n_cat_hidden is None else args.n_cat_hidden
    diag_noise = True if args.diag_noise is None else args.diag_noise == 'True'
    sigma_W = 0.001 if args.sigma_w is None else args.sigma_w
    sigma_b = 0.001 if args.sigma_b is None else args.sigma_b
    l_rate = 0.001 if args.l_rate is None else args.l_rate
    n_iter = 20 if args.n_iter is None else args.n_iter
    tune_A = True if args.tune_A is None else args.tune_A == 'True'
    plot = False if args.plot is None else args.plot == 'True'
    tau = 400 if args.tau is None else args.tau
    tune_tau = True if args.tune_tau is None else args.tune_tau == 'True'
    tie_weights = False \
        if args.tie_weights is None else args.tie_weights == 'True'
    n_hidden_layers = 5 \
        if args.n_hidden_layers is None else args.n_hidden_layers
    n_train_samples = 1 \
        if args.n_train_samples is None else args.n_train_samples
    gauss_io = EnumGaussIO.NONE \
        if args.gauss_io is None else EnumGaussIO.str2idx[args.gauss_io]
    tune_noise_io = True \
        if args.tune_noise_io is None else args.tune_noise_io == 'True'
    approx_cols = True if args.approx_cols is None \
        else args.approx_cols == 'True'
    tune_sigma_W = True \
        if args.tune_sigma_w is None else args.tune_sigma_w == 'True'
    tune_sigma_b = True \
        if args.tune_sigma_b is None else args.tune_sigma_b == 'True'

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
    print('tau: %.2e' % tau)
    print('tune_tau: %s' % str(tune_tau))
    print('tie weights: %s' % str(tie_weights))

    test_mlp(tie_weights=tie_weights, n_iter=n_iter, l_rate=l_rate,
             n_hidden_layers=n_hidden_layers, n_hidden=n_hidden,
             n_cat_in=n_cat_in, n_cat_hidden=n_cat_hidden,
             sigma_W=sigma_W, sigma_b=sigma_b, diag_noise=diag_noise,
             tune_sigma_W=tune_sigma_W, tune_sigma_b=tune_sigma_b,
             tune_A=tune_A, tune_noise_io=tune_noise_io, gauss_io=gauss_io,
             n_train_samples=n_train_samples, approx_cols=approx_cols,
             plot=plot, tau=tau, tune_tau=tune_tau)

if __name__ == "__main__":
    main()
