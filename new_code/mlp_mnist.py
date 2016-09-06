from __future__ import print_function

__docformat__ = 'restructedtext en'

import numpy as np

import theano
import theano.tensor as T

from utils import load_data, identity_map
from layers import HiddenLayer, DropoutLayer, StochLayer, SoftmaxLayer

# from lasagne.updates import sgd, apply_nesterov_momentum
from lasagne.updates import adam

# import os
# os.chdir('/home/ucabjh1/gitHub/machineLearning/bayesian_dp/mog_approx/code')


class MLP(object):

    def __init__(self, input, n_in=28**2, n_hidden_1=1024, n_hidden_2=1024,
                 n_hidden_3=1024, n_hidden_4=1024,
                 n_out=10, W_hidden_1=None, W_hidden_2=None,
                 W_hidden_3=None, W_hidden_4=None,
                 W_out=None, dropout=0.0, seed=None):

        relu_activation = lambda x: T.nnet.relu(x, 0.1)
        # relu_activation = T.nnet.relu

        seed = np.random.randint(int(1e5)) if seed is None else seed

        self.dropout_layer_1 = DropoutLayer(
                input=input,
                seed=seed,
                dropout=dropout
        )

        self.hidden_1 = HiddenLayer(
                seed=seed + 1,
                # input=input,
                input=self.dropout_layer_1.output,
                # input=self.dropout_layer.output,
                n_in=n_in,
                n_out=n_hidden_1,
                activation=relu_activation,
                W=W_hidden_1,
        )

        self.dropout_layer_2 = DropoutLayer(
                input=self.hidden_1.output,
                seed=seed + 2,
                dropout=dropout
        )

        self.hidden_2 = HiddenLayer(
                seed=seed + 3,
                # input=self.hidden_1.output,
                input=self.dropout_layer_2.output,
                n_in=n_hidden_1, n_out=n_hidden_2,
                activation=relu_activation,
                W=W_hidden_2
        )

        self.dropout_layer_3 = DropoutLayer(
                input=self.hidden_2.output,
                seed=seed + 4, dropout=dropout
        )

        self.hidden_3 = HiddenLayer(
                seed=seed + 5, input=self.dropout_layer_3.output,
                n_in=n_hidden_2, n_out=n_hidden_3,
                activation=relu_activation, W=W_hidden_3
        )

        self.dropout_layer_4 = DropoutLayer(
                input=self.hidden_3.output,
                seed=seed + 6, dropout=dropout
        )

        self.hidden_4 = HiddenLayer(
                seed=seed + 7, input=self.dropout_layer_4.output,
                n_in=n_hidden_3, n_out=n_hidden_4,
                activation=relu_activation, W=W_hidden_4
        )

        self.dropout_layer_5 = DropoutLayer(
                input=self.hidden_4.output, seed=seed + 8, dropout=dropout
        )

        self.linear_layer = HiddenLayer(
                seed=seed + 9,
                # input=self.hidden_1.output,
                # input=self.hidden_2.output,
                input=self.dropout_layer_5.output,
                n_in=n_hidden_4, n_out=n_out,
                activation=identity_map,
                W=W_out
        )

        self.softmax_layer = SoftmaxLayer(
                input=self.linear_layer.output
        )

        # keep track of model input
        self.input = input
        self.p_y_given_x = self.softmax_layer.p_y_given_x
        self.y_pred = self.softmax_layer.y_pred

        self.L1 = (
            abs(self.hidden_1.W).sum() + abs(self.hidden_2.W).sum()
            + abs(self.hidden_3.W).sum() + abs(self.hidden_4.W).sum()
            + abs(self.linear_layer.W).sum()
        )

        self.L2_sqr = (
            T.sum(self.hidden_1.W ** 2) + T.sum(self.hidden_2.W ** 2)
            + T.sum(self.hidden_3.W ** 2) + T.sum(self.hidden_4.W ** 2)
            + T.sum(self.linear_layer.W ** 2)
        )

        self.mean_log_likelihood = (
            self.softmax_layer.mean_log_likelihood
        )
        self.errors = self.softmax_layer.errors

        self.params = (
            self.hidden_1.params + self.hidden_2.params
            + self.hidden_3.params + self.hidden_4.params
            + self.linear_layer.params
        )


def test_mlp(learning_rate=0.01, n_epochs=10,  batch_size=128):

    dataset='mnist.pkl.gz'
    datasets = load_data(dataset)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    train_y = train_set_y.eval()
    valid_y = valid_set_y.eval()
    test_y = test_set_y.eval()

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] // batch_size

    # scaling down the input values
    train_set_x /= 255.0
    valid_set_x /= 255.0
    test_set_x /= 255.0

    print('... building the model')

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')  # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of

    n_in = 28 ** 2
    n_out = 10

    dropout = 0.5
    dd = 64

    n_hidden_1 = dd
    n_hidden_2 = dd
    n_hidden_3 = dd
    n_hidden_4 = dd

    n_cat_hidden_1 = n_in
    n_cat_hidden_2 = dd
    n_cat_hidden_3 = dd
    n_cat_hidden_4 = dd
    n_cat_out = dd

    # TODO: modify based on the updates made to StochLayer

    np.random.seed(1234)
    _, M_hidden_1, _ = (
        StochLayer.init_params(
                A=None, M=None, m=None, sigmoid_act=False,
                k=n_in, d=n_hidden_1, t=n_cat_hidden_1, n_out=n_hidden_1,
                A_scale=1.0, tie_weights=True
        )
    )
    _, M_hidden_2, _ = (
        StochLayer.init_params(
                A=None, M=None, m=None, sigmoid_act=False,
                k=n_hidden_1, d=n_hidden_2, t=n_cat_hidden_2, n_out=n_hidden_2,
                A_scale=1.0, tie_weights=True
        )
    )
    _, M_hidden_3, _ = (
        StochLayer.init_params(
                A=None, M=None, m=None, sigmoid_act=False,
                k=n_hidden_2, d=n_hidden_3, t=n_cat_hidden_3, n_out=n_hidden_3,
                A_scale=1.0, tie_weights=True
        )
    )
    _, M_hidden_4, _ = (
        StochLayer.init_params(
                A=None, M=None, m=None, sigmoid_act=False,
                k=n_hidden_3, d=n_hidden_4, t=n_cat_hidden_4, n_out=n_hidden_4,
                A_scale=1.0, tie_weights=True)
    )
    _, M_out, _ = (
        StochLayer.init_params(
                A=None, M=None, m=None, sigmoid_act=False,
                k=n_hidden_3, d=n_out, t=n_cat_out, n_out=n_hidden_4,
                A_scale=1.0, tie_weights=True
        )
    )

    # construct the MLP class
    classifier = MLP(
            input=x, n_in=n_in, n_hidden_1=n_hidden_1, n_hidden_2=n_hidden_2,
            n_out=n_out, n_hidden_3=n_hidden_3, n_hidden_4=n_hidden_4,
            # W_hidden_1=M_hidden_1, W_hidden_2=M_hidden_2,
            # W_hidden_3=M_hidden_3, W_hidden_4=M_hidden_4,
            # W_out=M_out,
            dropout=dropout, seed=1234
    )

    # TODO: change after debug
    cost = (
        -classifier.mean_log_likelihood(y)
        # + L1_reg * classifier.L1
        # + L2_reg * classifier.L2_sqr
    )

    predict_test = theano.function(
            inputs=[index],
            outputs=classifier.p_y_given_x,
            givens={x: test_set_x[index * batch_size:(index + 1) * batch_size]}
    )

    # validate_model = theano.function(
    #         inputs=[index],
    #         outputs=classifier.errors(y),
    #         givens={
    #             x: valid_set_x[index * batch_size:(index + 1) * batch_size],
    #             y: valid_set_y[index * batch_size:(index + 1) * batch_size]
    #         }
    # )

    updates = adam(cost, classifier.params, learning_rate=learning_rate)

    train_model = theano.function(
            inputs=[index],
            outputs=cost,
            updates=updates,
            givens={
                x: train_set_x[index * batch_size: (index + 1) * batch_size],
                y: train_set_y[index * batch_size: (index + 1) * batch_size]
            }
    )

    print('... training')
    n_pred_samples = 10

    cost_stats = np.zeros(n_epochs, dtype=theano.config.floatX)
    # ll_stats = np.zeros(n_epochs, dtype=theano.config.floatX)
    # kl_m_stats = np.zeros(n_epochs, dtype=theano.config.floatX)
    # kl_M_stats = np.zeros(n_epochs, dtype=theano.config.floatX)

    for i in range(n_epochs):
        cost_iter = np.zeros(n_train_batches, dtype=theano.config.floatX)
        # ll_iter = np.zeros(n_train_batches, dtype=theano.config.floatX)
        # kl_m_iter = np.zeros(n_train_batches, dtype=theano.config.floatX)
        # kl_M_iter = np.zeros(n_train_batches, dtype=theano.config.floatX)

        for batch_idx in range(n_train_batches):
            cost_iter[batch_idx] = train_model(batch_idx)
            # iter_out = train_model(batch_idx)
            # cost_iter[batch_idx] = iter_out[0]
            # ll_iter[batch_idx] = iter_out[1]
            # kl_m_iter[batch_idx] = iter_out[2]
            # kl_M_iter[batch_idx] = iter_out[3]

        cost_stats[i] = np.mean(cost_iter)
        # ll_stats[i] = np.mean(ll_iter)
        # kl_m_stats[i] = np.mean(kl_m_iter)
        # kl_M_stats[i] = np.mean(kl_M_iter)

        # TODO: add validation and testing (some early stopping rule)

        if (i+1) % 5 == 0:
            print('finished iteration: %d' % (i+1))
            error_rate = np.mean(
                    [np.argmax(np.mean(
                            [predict_test(i) for _ in range(n_pred_samples)],
                            axis=0), axis=1)
                     != test_y[i*batch_size:(i+1)*batch_size]
                     for i in range(n_test_batches)]
            )
            print('average test error: %f' % error_rate)

    print('min cost: %f' % np.min(cost_stats))


if __name__ == '__main__':
    test_mlp()
