from __future__ import print_function

import os
import argparse
import pickle
import matplotlib.pyplot as plt

import theano
import theano.tensor as T

from mlp_gauss import GaussBNN, prepare_functions
from utils import prepare_mnist_data, prepare_cifar10_data, train_model

out_dir = os.getcwd()
# os.chdir('/home/ucabjh1/gitHub/machineLearning/bayesian_dp/mog_approx/code')

floatX = theano.config.floatX


def prepare_model(X, n_in, n_out, n_hidden_layers, n_units_in, n_units_hidden,
                  sigma_W, tune_sigma_W, sigma_b, tune_sigma_b,
                  W_len_scale, b_len_scale, diag_noise, approx_cols, seed):

    # using as initilisation for deep GPs
    divide_1st_layer_by_its_n_out = False  # gauss_io in (INPUT, IN_OUT)
    b_out_deterministic = True  # always

    model = GaussBNN(
        X=X, n_in=n_in, n_out=n_out, n_hidden_layers=n_hidden_layers,
        n_units_in=n_units_in, n_units_hidden=n_units_hidden,
        sigma_W=sigma_W, tune_sigma_W=tune_sigma_W,
        sigma_b=sigma_b, tune_sigma_b=tune_sigma_b,
        l_W=W_len_scale, l_b=b_len_scale,
        diag_noise=diag_noise, approx_cols=approx_cols,
        divide_1st_layer_by_its_n_out=divide_1st_layer_by_its_n_out,
        b_out_deterministic=b_out_deterministic, seed=seed
    )

    return model


def test_mlp(l_rate=0.01, n_iter=20, batch_size=128,
             n_hidden_layers=4, n_units_in=512, n_units_hidden=512,
             test_at_iter=5, n_test_samples=20,
             W_len_scale=1e-6, b_len_scale=1e-6,
             sigma_W=1e-3, tune_sigma_W=True,
             sigma_b=1e-6, tune_sigma_b=True,
             diag_noise=True, approx_cols=False,
             seed=1234, plot=False, mnist=True):

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
        X=X, n_hidden_layers=n_hidden_layers, n_in=n_in, n_out=n_out,
        n_units_in=n_units_in, n_units_hidden=n_units_hidden,
        sigma_W=sigma_W, tune_sigma_W=tune_sigma_W,
        sigma_b=sigma_b, tune_sigma_b=tune_sigma_b,
        W_len_scale=W_len_scale, b_len_scale=b_len_scale,
        diag_noise=diag_noise, approx_cols=approx_cols, seed=seed
    )

    # compile theano functions
    train, test_predict = prepare_functions(
        model=model, X=X, y=y, index=index,
        X_train=X_train, X_test=X_test, y_train=y_train,
        batch_size=batch_size, l_rate=l_rate
    )

    cost_stats, ll_stats, kl_W_stats, kl_b_stats, test_err = train_model(
        train=train, test_predict=test_predict, y_test=y_test,
        batch_size=batch_size, test_at_iter=test_at_iter,
        n_iter=n_iter, n_test_batches=n_test_batches,
        n_test_samples=n_test_samples, n_train_batches=n_train_batches
    )

    fname = (
        'gaussBNN_mnist_%dn_%du_%dk_%.2ew_%.2eb_%sm_%se_%.2el_%di_%dt'
        '_cols_%s_diag_%s.save' %
        (n_hidden_layers, n_units_in, n_units_hidden, sigma_W, sigma_b,
         str(tune_sigma_W), str(tune_sigma_b), l_rate,
         n_iter, test_at_iter, str(approx_cols), str(diag_noise))
    )
    out_file = os.path.join(out_dir, fname)
    with open(out_file, 'wb') as f:
        pickle.dump(model.get_param_dictionary(), f)
        pickle.dump([cost_stats, ll_stats, kl_b_stats, kl_W_stats, test_err], f)

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

    parser = argparse.ArgumentParser(description='Run simulation of GaussBNN')

    parser.add_argument('-n', '--n_hidden_layers', type=int, required=False)
    parser.add_argument('-u', '--n_units_in', type=int, required=False)
    parser.add_argument('-k', '--n_units_hidden', type=int, required=False)
    parser.add_argument('-d', '--diag_noise', type=str, required=False)
    parser.add_argument('-w', '--sigma_w', type=float, required=False)
    parser.add_argument('-b', '--sigma_b', type=float, required=False)
    parser.add_argument('-m', '--tune_sigma_w', type=str, required=False)
    parser.add_argument('-e', '--tune_sigma_b', type=str, required=False)
    parser.add_argument('-l', '--l_rate', type=float, required=False)
    parser.add_argument('-c', '--approx_cols', type=str, required=False)
    parser.add_argument('-i', '--n_iter', type=int, required=False)
    parser.add_argument('-t', '--test_at_iter', type=int, required=False)
    parser.add_argument('-p', '--plot', type=str, required=False)
    parser.add_argument('-j', '--mnist', type=str, required=False)

    args = parser.parse_args()
    assert args.plot is None or args.plot in ('True', 'False')
    assert args.diag_noise is None or args.diag_noise in ('True', 'False')
    assert args.tune_sigma_w is None or args.tune_sigma_w in ('True', 'False')
    assert args.tune_sigma_b is None or args.tune_sigma_b in ('True', 'False')
    assert args.mnist is None or args.mnist in ('True', 'False')

    n_units_in = 512 if args.n_units_in is None else args.n_units_in
    n_units_hidden = 512 if args.n_units_hidden is None else args.n_units_hidden
    diag_noise = True if args.diag_noise is None else args.diag_noise == 'True'
    sigma_W = 1e-3 if args.sigma_w is None else args.sigma_w
    sigma_b = 1e-6 if args.sigma_b is None else args.sigma_b
    l_rate = 0.01 if args.l_rate is None else args.l_rate
    n_iter = 20 if args.n_iter is None else args.n_iter
    test_at_iter = 5 if args.test_at_iter is None else args.test_at_iter
    plot = False if args.plot is None else args.plot == 'True'
    mnist = True if args.mnist is None else args.mnist == 'True'
    n_hidden_layers = 4 \
        if args.n_hidden_layers is None else args.n_hidden_layers
    approx_cols = False if args.approx_cols is None \
        else args.approx_cols == 'True'
    tune_sigma_W = True \
        if args.tune_sigma_w is None else args.tune_sigma_w == 'True'
    tune_sigma_b = True \
        if args.tune_sigma_b is None else args.tune_sigma_b == 'True'

    print('Running GPMC approximation with tied weights')
    print('n_hidden_layers: %d' % n_hidden_layers)
    print('n_units_in: %d' % n_units_in)
    print('n_units_hidden: %d' % n_units_hidden)
    print('diag_noise: %s' % str(diag_noise))
    print('sigma_W: %.2e' % sigma_W)
    print('sigma_b: %.2e' % sigma_b)
    print('tune_sigma_W: %s' % str(tune_sigma_W))
    print('tune_sigma_b: %s' % str(tune_sigma_b))
    print('n_iter: %d' % n_iter)
    print('test_at_iter: %d' % test_at_iter)
    print('l_rate: %.2e' % l_rate)
    print('approx_cols: %s' % str(approx_cols))
    print('plot: %s' % str(plot))
    print('mnist: %s' % str(mnist))

    test_mlp(n_iter=n_iter, l_rate=l_rate, test_at_iter=test_at_iter,
             n_hidden_layers=n_hidden_layers, n_units_in=n_units_in,
             n_units_hidden=n_units_hidden, sigma_W=sigma_W, sigma_b=sigma_b,
             tune_sigma_W=tune_sigma_W, tune_sigma_b=tune_sigma_b,
             diag_noise=diag_noise, approx_cols=approx_cols,
             plot=plot, mnist=mnist)


if __name__ == '__main__':
    main()
