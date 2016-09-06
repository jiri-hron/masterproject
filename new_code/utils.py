from __future__ import print_function

import os
import pickle
import gzip

import numpy as np
import theano
import theano.tensor as T

from keras.datasets import cifar10

floatX = theano.config.floatX
eps = T.constant(1e-33, name='eps', dtype=floatX)


def log_sum_exp(x, axis=None):
    """
    Standard LogSumExp able to handle large values in the exponents.

    https://en.wikipedia.org/wiki/LogSumExp
    """
    x_max = T.max(x, axis=axis, keepdims=True)
    ret = T.log(T.sum(T.exp(x - x_max) + eps, axis=axis, keepdims=True)) + x_max
    return ret


def merge_dicts(*dict_args):
    """
    Merges multiple dictionaries into one.

    If there are conflicting keys, the key will be mapped to the last passed
    dictionary that contained the key.
    """
    result = {}
    for dictionary in dict_args:
        result.update(dictionary)
    return result


def identity_map(x):
    """
    Returns it's input - used when a function has to be passed but we don't
    want to make any changes to the input.
    """
    return x


def floatX_arr(x):
    """
    Cast as floatX array.
    """
    return np.asarray(x, dtype=floatX)


def load_data(dataset):
    """
    Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset (here MNIST)
    """

    # Download the MNIST dataset if it is not present
    data_dir, data_file = os.path.split(dataset)
    if data_dir == "" and not os.path.isfile(dataset):
        # Check if dataset is in the data directory.
        new_path = os.path.join(
            os.path.split(__file__)[0],
            # "..",
            # "data",
            dataset
        )
        if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
            dataset = new_path

    if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
        from six.moves import urllib
        origin = (
            'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        )
        print('Downloading data from %s' % origin)
        urllib.request.urlretrieve(origin, dataset)

    print('... loading data')

    # Load the dataset
    with gzip.open(dataset, 'rb') as f:
        try:
            train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
        except:
            train_set, valid_set, test_set = pickle.load(f)

    def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(
            np.asarray(data_x, dtype=theano.config.floatX), borrow=borrow
        )
        shared_y = theano.shared(
            np.asarray(data_y, dtype=theano.config.floatX), borrow=borrow
        )
        return shared_x, T.cast(shared_y, 'int32')

    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval


def prepare_toy_data(n_train, n_valid, batch_size):
    n_train_batches = n_train // batch_size if batch_size < n_train else 1
    n_valid_batches = n_valid // batch_size if batch_size < n_valid else 1

    rng = np.random.RandomState(1234)  # always return the same

    n_train_per_int = n_train // 2

    # interpolation on [-0.5, 0.0], extrapolation on [0.5, 1.0]
    # X_train = np.concatenate((
    #     rng.uniform(low=-1.0, high=-0.5, size=n_train_per_int),
    #     rng.uniform(low=0.0, high=0.5, size=n_train - n_train_per_int)
    # )).astype(floatX)
    # X_valid = rng.uniform(low=-1.0, high=0.5, size=n_valid).astype(floatX)
    X_train = np.asarray(rng.uniform(low=-1.0, high=0.5, size=n_train),
                         dtype=floatX)
    X_valid = np.asarray(rng.uniform(low=-1.0, high=1.0, size=n_valid),
                         dtype=floatX)

    y_train = np.asarray(
        # 0.4*np.sin(3 * 2*np.pi*X_train) + 0.05*rng.normal(size=n_train),
        0.4 * np.cos(2 * np.pi * X_train) ** 2 *
        np.sin(2 * np.pi * X_train + 0.1) +
        0.01 * rng.normal(size=n_train),
        dtype=floatX
    )
    y_valid = np.asarray(
        # 0.4*np.sin(3 * 2*np.pi*X_valid) + 0.05*rng.normal(size=n_valid),
        0.4 * np.cos(2 * np.pi * X_valid) ** 2 *
        np.sin(2 * np.pi * X_valid + 0.1) +
        0.01 * rng.normal(size=n_valid),
        dtype=floatX
    )

    X_train = T.shape_padaxis(theano.shared(X_train, name='X_train'), axis=1)
    y_train = theano.shared(y_train, name='y_train')
    X_valid = T.shape_padaxis(theano.shared(X_valid, name='X_valid'), axis=1)
    y_valid = theano.shared(y_valid, name='y_valid')

    # used in evaluation with multiple samples
    y_valid = np.array(y_valid.eval())

    return X_train, y_train, X_valid, y_valid, n_train_batches, n_valid_batches


def prepare_co2_data(batch_size):
    data = np.genfromtxt('co2.csv')
    n = data.shape[0]

    n_pred = 5*12  # predict next five years

    # we will extrapolate into future, hence train and test data are the same
    X_train = np.arange(n, dtype=floatX)
    X_valid = np.arange(n, dtype=floatX)
    X_test = np.arange(n, n + n_pred, dtype=floatX)

    y_train = data.astype(floatX)
    y_valid = data.astype(floatX)

    n_train_batches = n // batch_size if batch_size < n else 1
    n_valid_batches = n // batch_size if batch_size < n else 1
    n_test_batches = n_pred // batch_size if batch_size < n_pred else 1

    X_train = theano.shared(X_train, name='X_train')
    y_train = theano.shared(y_train, name='y_train')
    X_valid = theano.shared(X_valid, name='X_valid')
    y_valid = theano.shared(y_valid, name='y_valid')
    X_test = theano.shared(X_test, name='X_test')
    y_test = None

    # used in evaluation with multiple samples
    y_valid = np.array(y_valid.eval())

    return (
        X_train, y_train, X_valid, y_valid, X_test, y_test,
        n_train_batches, n_valid_batches, n_test_batches
    )


def prepare_cifar10_data(batch_size):

    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    y_train = y_train.flatten()
    y_test = y_test.flatten()

    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)

    rng = np.random.RandomState(1234)  # always return the same
    idx = rng.permutation(range(X_train.shape[0]))

    # TODO: change for the final test error estimate
    X_valid = X_train[idx][-5000:].astype(floatX)
    y_valid = y_train[idx][-5000:].astype('int32')
    X_train = X_train[idx][:-5000].astype(floatX)
    y_train = y_train[idx][:-5000].astype('int32')

    n_train_batches = X_train.shape[0] // batch_size
    n_valid_batches = X_valid.shape[0] // batch_size

    X_train /= 9.0
    X_valid /= 9.0

    X_train = theano.shared(X_train, name='X_train')
    y_train = theano.shared(y_train, name='y_train')
    X_valid = theano.shared(X_valid, name='X_valid')
    y_valid = theano.shared(y_valid, name='y_valid')

    # used in evaluation with multiple samples
    y_valid = np.array(y_valid.eval())

    return X_train, y_train, X_valid, y_valid, n_train_batches, n_valid_batches


def prepare_mnist_test_data(batch_size):

    # load data
    dataset = 'mnist.pkl.gz'
    datasets = load_data(dataset)

    X_test, y_test = datasets[2]
    y_test = np.array(y_test.eval())

    n_test_batches = X_test.get_value(borrow=True).shape[0] // batch_size

    X_test /= 255.0

    return X_test, y_test, n_test_batches


def prepare_mnist_data(batch_size):

    # load data
    dataset = 'mnist.pkl.gz'
    datasets = load_data(dataset)

    # TODO: change for the final test error estimate
    X_train, y_train = datasets[0]
    X_valid, y_valid = datasets[1]

    # used in evaluation with multiple samples
    y_valid = np.array(y_valid.eval())

    n_train_batches = X_train.get_value(borrow=True).shape[0] // batch_size
    n_valid_batches = X_valid.get_value(borrow=True).shape[0] // batch_size

    X_train /= 255.0
    X_valid /= 255.0

    return X_train, y_train, X_valid, y_valid, n_train_batches, n_valid_batches

    # X_train, y_train = datasets[0]
    # valid_set_x, valid_set_y = datasets[1]
    # X_test, test_set_y = datasets[2]
    #
    # # convert to 60,000 train set, 10,000 test set
    # X_train = (
    #     theano.shared(np.vstack((X_train.get_value(),
    #                              valid_set_x.get_value())),
    #                   name='X_train')
    # )
    # y_train = (
    #     theano.shared(np.concatenate((y_train.eval(),
    #                                   valid_set_y.eval())),
    #                   name='y_train')
    # )
    #
    # # free up memory
    # del valid_set_x
    # del valid_set_y
    #
    # # train_y = y_train.eval()
    # y_test = test_set_y.eval()
    # n_train_batches = X_train.get_value(borrow=True).shape[0] // batch_size
    # n_test_batches = X_test.get_value(borrow=True).shape[0] // batch_size
    #
    # # scaling down the input values
    # X_train /= 255.0
    # X_test /= 255.0
    #
    # return (
    #     X_train, y_train, X_test, y_test, n_train_batches, n_test_batches
    # )


def error_classification_batch(idx, test_predict, y, n_samples):
    # misclassification error (1-0 loss)
    probs = np.mean([test_predict(idx) for _ in xrange(n_samples)], axis=0)
    pred = np.argmax(probs, axis=1)
    return np.sum(pred != y)


def error_regression_batch(idx, test_predict, y, n_samples):
    # residual mean squared error
    pred_mean = np.mean([test_predict(idx) for _ in xrange(n_samples)], axis=0)
    return np.mean((pred_mean - y) ** 2)


def train_model(train, test_predict, y_test, batch_size, test_at_iter,
                n_iter, n_test_batches, n_test_samples, n_train_batches,
                classification=True):

    print('... training')

    error_batch = error_classification_batch \
        if classification else error_regression_batch

    if type(y_test) == np.ndarray:
        n_test_points = float(y_test.shape[0])
    else:
        n_test_points = float(y_test.shape[0].eval())

    cost_stats = np.zeros(n_iter, dtype=floatX)
    ll_stats = np.zeros(n_iter, dtype=floatX)
    kl_W_stats = np.zeros(n_iter, dtype=floatX)
    kl_b_stats = np.zeros(n_iter, dtype=floatX)
    test_err = np.zeros(n_iter // test_at_iter)

    for i in range(n_iter):
        cost_iter = np.zeros(n_train_batches, dtype=floatX)
        ll_iter = np.zeros(n_train_batches, dtype=floatX)
        kl_M_iter = np.zeros(n_train_batches, dtype=floatX)
        kl_m_iter = np.zeros(n_train_batches, dtype=floatX)

        for j in range(n_train_batches):
            cost_iter[j], ll_iter[j], kl_M_iter[j], kl_m_iter[j] = train(j)

        cost_stats[i] = np.mean(cost_iter)
        ll_stats[i] = np.mean(ll_iter)
        kl_W_stats[i] = np.mean(kl_M_iter)
        kl_b_stats[i] = np.mean(kl_m_iter)

        if (i + 1) % test_at_iter == 0:
            print('finished iteration: %d' % (i + 1))

            err = np.sum(
                [error_batch(
                    idx=k, test_predict=test_predict,
                    y=y_test[k * batch_size:(k + 1) * batch_size],
                    n_samples=n_test_samples) for k in xrange(n_test_batches)]
            ) / n_test_points

            test_err[((i + 1) // test_at_iter) - 1] = err
            print('average test error: %f' % err)

    print('min cost: %f' % np.min(cost_stats))

    return cost_stats, ll_stats, kl_W_stats, kl_b_stats, test_err
