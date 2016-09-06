from __future__ import print_function

import os
import argparse
import pickle
import matplotlib.pyplot as plt

import theano
import theano.tensor as T

from utils import prepare_mnist_data, prepare_cifar10_data, train_model
from mlp_gpmc import GPMC, EnumGaussIO, prepare_functions

out_dir = os.getcwd()
# os.chdir('/home/ucabjh1/gitHub/machineLearning/bayesian_dp/mog_approx/code')

floatX = theano.config.floatX


def prepare_model(X, n_in, n_out, n_cat, n_hidden, n_hidden_layers,
                  sigma_W, tune_sigma_W, sigma_b, tune_sigma_b,
                  tune_noise_io, diag_noise, gauss_io, approx_cols,
                  dropout, dropout_prob, A_scale, tune_A,  seed):

    n_hidden_lst = [n_hidden] * (n_hidden_layers + 1)
    n_cat_lst = [n_cat] * (n_hidden_layers + 1)

    model = GPMC(
        input=X, n_in=n_in, n_out=n_out, n_hidden_layers=n_hidden_layers,
        n_hidden_lst=n_hidden_lst, n_cat_lst=n_cat_lst,
        sigma_W=sigma_W, sigma_b=sigma_b,
        tune_sigma_W=tune_sigma_W, tune_sigma_b=tune_sigma_b,
        tune_sigma_W_io=tune_noise_io, tune_sigma_b_io=tune_noise_io,
        diag_noise=diag_noise, tie_weights=False, approx_cols=approx_cols,
        l_W=1e-6, l_b=1e-6, A_scale=A_scale, tune_A=tune_A,
        dropout=dropout, dropout_prob=dropout_prob,
        gauss_io=gauss_io, seed=seed
    )

    return model


def test_mlp(l_rate=0.001, n_iter=10, batch_size=256, test_at_iter=5,
             n_hidden_layers=4, n_hidden=512, n_cat=2,
             n_train_samples=1, n_test_samples=10,
             sigma_W=1e-3, sigma_b=1e-6, tune_sigma_W=True, tune_sigma_b=True,
             tune_noise_io=True, diag_noise=True, gauss_io=EnumGaussIO.NONE,
             A_scale=1.0, tune_A=False, approx_cols=False,
             ll_track_decay=0.25, ll_net_hidden=256,
             dropout=False, dropout_prob=0.5,
             plot=False, seed=1234, mnist=True):

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

    print('... building the model')

    # define symbolic variables
    index = T.lscalar('index')
    X = T.matrix('X', dtype=floatX)
    y = T.ivector('y')

    # create model
    model = prepare_model(
        X=X, n_in=n_in, n_out=n_out, n_hidden_layers=n_hidden_layers,
        n_cat=n_cat, n_hidden=n_hidden,
        sigma_W=sigma_W, sigma_b=sigma_b,
        tune_sigma_W=tune_sigma_W, tune_sigma_b=tune_sigma_b,
        tune_noise_io=tune_noise_io, diag_noise=diag_noise,
        A_scale=A_scale, tune_A=tune_A, approx_cols=approx_cols,
        dropout=dropout, dropout_prob=dropout_prob,
        gauss_io=gauss_io, seed=seed
    )

    train, test_predict = prepare_functions(
        model=model, index=index, X=X, y=y, n_in=n_in,
        X_train=X_train, X_test=X_test, y_train=y_train,
        l_rate=l_rate, batch_size=batch_size,
        n_train_samples=n_train_samples,
        ll_track_decay=ll_track_decay,
        ll_net_hidden=ll_net_hidden
    )

    cost_stats, ll_stats, kl_W_stats, kl_b_stats, test_err_stats = train_model(
        train=train, test_predict=test_predict, y_test=y_test,
        batch_size=batch_size, test_at_iter=test_at_iter,
        n_iter=n_iter, n_test_batches=n_test_batches,
        n_test_samples=n_test_samples, n_train_batches=n_train_batches
    )

    fname = 'gpmc_untied_%dh_%dc_%ds_%.2elr_gaussIO_%s_tuneA_%s' \
            '_tuneSigmaW_%s_tuneSigmab_%s' \
            '_cols_%s_diag_%s_dropout_%s_prob_%.2e.save' \
            % (n_hidden, n_cat, n_train_samples, l_rate,
               EnumGaussIO.idx2str[gauss_io], str(tune_A),
               str(tune_sigma_W), str(tune_sigma_b),
               str(approx_cols), str(diag_noise),
               str(dropout), dropout_prob)
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
    parser.add_argument('-d', '--diag_noise', type=str, required=False)
    parser.add_argument('-o', '--tune_noise_io', type=str, required=False)
    parser.add_argument('-w', '--sigma_w', type=float, required=False)
    parser.add_argument('-b', '--sigma_b', type=float, required=False)
    parser.add_argument('-t', '--n_cat', type=int, required=False)
    parser.add_argument('-s', '--n_samples', type=int, required=False)
    parser.add_argument('-l', '--l_rate', type=float, required=False)
    parser.add_argument('-i', '--n_iter', type=int, required=False)
    parser.add_argument('-g', '--gauss_io', type=str, required=False)
    parser.add_argument('-c', '--approx_cols', type=str, required=False)
    parser.add_argument('-a', '--tune_A', type=str, required=False)
    parser.add_argument('-m', '--tune_sigma_w', type=str, required=False)
    parser.add_argument('-e', '--tune_sigma_b', type=str, required=False)
    parser.add_argument('-p', '--plot', type=str, required=False)
    parser.add_argument('-j', '--mnist', type=str, required=False)
    parser.add_argument('-z', '--dropout', type=str, required=False)
    parser.add_argument('-y', '--dropout_prob', type=float, required=False)

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
    assert args.dropout is None or args.dropout in ('True', 'False')

    n_hidden = 512 if args.n_hidden is None else args.n_hidden
    n_cat = 2 if args.n_cat is None else args.n_cat
    diag_noise = True if args.diag_noise is None else args.diag_noise == 'True'
    sigma_W = 0.0 if args.sigma_w is None else args.sigma_w
    sigma_b = 0.0 if args.sigma_b is None else args.sigma_b
    n_samples = 1 if args.n_samples is None else args.n_samples
    l_rate = 0.001 if args.l_rate is None else args.l_rate
    n_iter = 20 if args.n_iter is None else args.n_iter
    tune_A = False if args.tune_A is None else args.tune_A == 'True'
    plot = False if args.plot is None else args.plot == 'True'
    mnist = True if args.mnist is None else args.mnist == 'True'
    dropout = False if args.dropout is None else args.dropout == 'True'
    dropout_prob = 0.5 if args.dropout_prob is None else args.dropout_prob
    gauss_io = EnumGaussIO.NONE if args.gauss_io is None \
        else EnumGaussIO.str2idx[args.gauss_io]
    approx_cols = False if args.approx_cols is None \
        else args.approx_cols == 'True'
    n_hidden_layers = 4 \
        if args.n_hidden_layers is None else args.n_hidden_layers
    tune_sigma_W = True \
        if args.tune_sigma_w is None else args.tune_sigma_w == 'True'
    tune_sigma_b = True \
        if args.tune_sigma_b is None else args.tune_sigma_b == 'True'
    tune_noise_io = False \
        if args.tune_noise_io is None else args.tune_noise_io == 'True'

    print('Running GPMC approximation with untied weights')
    print('n_hidden_layers: %d' % n_hidden_layers)
    print('n_hidden: %d' % n_hidden)
    print('n_cat: %d' % n_cat)
    print('tune_noise_io: %s' % str(tune_noise_io))
    print('diag_noise: %s' % str(diag_noise))
    print('sigma_W: %.2e' % sigma_W)
    print('sigma_b: %.2e' % sigma_b)
    print('tune_sigma_W: %s' % str(tune_sigma_W))
    print('tune_sigma_b: %s' % str(tune_sigma_b))
    print('n_samples: %d' % n_samples)
    print('l_rate: %.2e' % l_rate)
    print('n_iter: %d' % n_iter)
    print('gauss_io: %s' % EnumGaussIO.idx2str[gauss_io])
    print('approx_cols: %s' % str(approx_cols))
    print('tune_A: %s' % str(tune_A))
    print('plot: %s' % str(plot))
    print('mnist: %s' % str(mnist))
    print('dropout: %s' % str(dropout))
    print('dropout_prob: %.2e' % dropout_prob)

    test_mlp(n_iter=n_iter, l_rate=l_rate,
             n_hidden=n_hidden, n_hidden_layers=n_hidden_layers,
             n_cat=n_cat, n_train_samples=n_samples,
             sigma_W=sigma_W, sigma_b=sigma_b, diag_noise=diag_noise,
             tune_sigma_W=tune_sigma_W, tune_sigma_b=tune_sigma_b,
             tune_noise_io=tune_noise_io, tune_A=tune_A, gauss_io=gauss_io,
             approx_cols=approx_cols, plot=plot, mnist=mnist,
             dropout=dropout, dropout_prob=dropout_prob)

if __name__ == "__main__":
    main()
