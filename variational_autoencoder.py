from __future__ import division
import sys
import os
import numpy as np
import theano
import theano.tensor as T
import lasagne as nn
import time
from PIL import Image
from scipy.stats import norm
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import h5py
from scipy.io import wavfile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import wave




# ############################################################################
# Tencia Lee
# Some code borrowed from:
# https://github.com/Lasagne/Lasagne/blob/master/examples/mnist.py
#
# Implementation of variational autoencoder (AEVB) algorithm as in:
# [1] arXiv:1312.6114 [stat.ML] (Diederik P Kingma, Max Welling 2013)

# ################## Download and prepare the MNIST dataset ##################
# For the linked MNIST data, the autoencoder learns well only in binary mode.
# This is most likely due to the distribution of the values. Most pixels are
# either very close to 0, or very close to 1.
#
# Running this code with default settings should produce a manifold similar
# to the example in this directory. An animation of the manifold's evolution
# can be found here: https://youtu.be/pgmnCU_DxzM


# This code has been modified by Ryan Lowe for audio prediction


def create_dataset(test_pct, num_inputs):
    """ Takes the raw HDF5 file and converts it to a numpy array.
        No 'seq_len' since it is assumed to be 1 (not needed for VAE). 
    """
    path = './XqaJ2Ol5cC4.hdf5'
    with h5py.File(path, 'r') as f:
        dataset = f['features'][0].flatten()

    def segment_data(data, example_size):
        """ Data enters as a 1-D array, and split up into chunks of size
        example_size, which are separated into sequences of length seq_len.
        """
        data = (data - sum(data) / len(data)) /  (max(data) - min(data))
        num_seq = int(len(data) / (example_size))
        data = data[: num_seq * example_size]
        return np.array(data).reshape(num_seq, example_size).astype('float32')
    train_data = dataset[: len(dataset)*(1 - 2*test_pct)]
    data_avg = sum(train_data) / len(train_data)
    data_range = max(train_data) - min(train_data)
    data = {} 
    data['train'] = segment_data(dataset[: len(dataset)*(1 - 2*test_pct)],
            num_inputs)
    data['val'] = segment_data(dataset[len(dataset)*(1 - 2*test_pct):
        len(dataset)*(1 - test_pct)], num_inputs)
    data['test'] = segment_data(dataset[len(dataset)*(1 - test_pct):],
            num_inputs)
    return data, data_avg, data_range

# ############################# Batch iterator ###############################

def iterate_minibatches(inputs, batchsize, shuffle=False):
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt]

# ##################### Custom layer for middle of VCAE ######################
# This layer takes the mu and sigma (both DenseLayers) and combines them with
# a random vector epsilon to sample values for a multivariate Gaussian

class GaussianSampleLayer(nn.layers.MergeLayer):
    def __init__(self, mu, logsigma, rng=None, **kwargs):
        self.rng = rng if rng else RandomStreams(nn.random.get_rng().randint(1,2147462579))
        super(GaussianSampleLayer, self).__init__([mu, logsigma], **kwargs)

    def get_output_shape_for(self, input_shapes):
        return input_shapes[0]

    def get_output_for(self, inputs, deterministic=False, **kwargs):
        mu, logsigma = inputs
        shape=(self.input_shapes[0][0] or inputs[0].shape[0],
                self.input_shapes[0][1] or inputs[0].shape[1])
        if deterministic:
            return mu
        return mu + T.exp(logsigma) * self.rng.normal(shape)

# ############################## Build Model #################################
# encoder has 1 hidden layer, where we get mu and sigma for Z given an inp X
# continuous decoder has 1 hidden layer, where we get mu and sigma for X given code Z
# binary decoder has 1 hidden layer, where we calculate p(X=1)
# once we have (mu, sigma) for Z, we sample L times
# Then L separate outputs are constructed and the final layer averages them

def build_vae(inputvar, L=2, binary=True, z_dim=2, n_hid=1024, num_inputs=32000):
    x_dim = num_inputs
    l_input = nn.layers.InputLayer(shape=(None,x_dim),
            input_var=inputvar, name='input')
    l_enc_hid1 = nn.layers.DenseLayer(l_input, num_units=n_hid,
            nonlinearity=nn.nonlinearities.tanh if binary else T.nnet.softplus,
            name='enc_hid1')
    l_enc_hid = nn.layers.DenseLayer(l_input, num_units=n_hid,
            nonlinearity=nn.nonlinearities.tanh if binary else T.nnet.softplus,
            name='enc_hid2')
    
    l_enc_mu = nn.layers.DenseLayer(l_enc_hid, num_units=z_dim,
            nonlinearity = None, name='enc_mu')
    l_enc_logsigma = nn.layers.DenseLayer(l_enc_hid, num_units=z_dim,
            nonlinearity = None, name='enc_logsigma')
    l_dec_mu_list = []
    l_dec_logsigma_list = []
    l_output_list = []
    # tie the weights of all L versions so they are the "same" layer
    W_dec_hid = None
    b_dec_hid = None
    W_dec_hid1 = None
    b_dec_hid1 = None
    W_dec_mu = None
    b_dec_mu = None
    W_dec_ls = None
    b_dec_ls = None
    for i in xrange(L):
        l_Z = GaussianSampleLayer(l_enc_mu, l_enc_logsigma, name='Z')
        #l_dec_hid1 = nn.layers.DenseLayer(l_Z, num_units=n_hid,
        #        nonlinearity = nn.nonlinearities.tanh if binary else T.nnet.softplus,
        #        W=nn.init.GlorotUniform() if W_dec_hid is None else W_dec_hid,
        #        b=nn.init.Constant(0.) if b_dec_hid is None else b_dec_hid,
        #        name='dec_hid1')
        l_dec_hid = nn.layers.DenseLayer(l_Z, num_units=n_hid,
                nonlinearity = nn.nonlinearities.tanh if binary else T.nnet.softplus,
                W=nn.init.GlorotUniform() if W_dec_hid is None else W_dec_hid,
                b=nn.init.Constant(0.) if b_dec_hid is None else b_dec_hid,
                name='dec_hid')
        if binary:
            l_output = nn.layers.DenseLayer(l_dec_hid, num_units = x_dim,
                    nonlinearity = nn.nonlinearities.sigmoid,
                    W = nn.init.GlorotUniform() if W_dec_mu is None else W_dec_mu,
                    b = nn.init.Constant(0.) if b_dec_mu is None else b_dec_mu,
                    name = 'dec_output')
            l_output_list.append(l_output)
            if W_dec_hid is None:
                #W_dec_hid1 = l_dec_hid1.W
                #b_dec_hid1 = l_dec_hid1.b
                W_dec_hid = l_dec_hid.W
                b_dec_hid = l_dec_hid.b
                W_dec_mu = l_output.W
                b_dec_mu = l_output.b
        else:
            l_dec_mu = nn.layers.DenseLayer(l_dec_hid, num_units=x_dim,
                    nonlinearity = None,
                    W = nn.init.GlorotUniform() if W_dec_mu is None else W_dec_mu,
                    b = nn.init.Constant(0) if b_dec_mu is None else b_dec_mu,
                    name = 'dec_mu')
            # relu_shift is for numerical stability - if training data has any
            # dimensions where stdev=0, allowing logsigma to approach -inf
            # will cause the loss function to become NAN. So we set the limit
            # stdev >= exp(-1 * relu_shift)
            relu_shift = 10
            l_dec_logsigma = nn.layers.DenseLayer(l_dec_hid, num_units=x_dim,
                    W = nn.init.GlorotUniform() if W_dec_ls is None else W_dec_ls,
                    b = nn.init.Constant(0) if b_dec_ls is None else b_dec_ls,
                    nonlinearity = lambda a: T.nnet.relu(a+relu_shift)-relu_shift,
                    name='dec_logsigma')
            l_output = GaussianSampleLayer(l_dec_mu, l_dec_logsigma,
                    name='dec_output')
            l_dec_mu_list.append(l_dec_mu)
            l_dec_logsigma_list.append(l_dec_logsigma)
            l_output_list.append(l_output)
            if W_dec_hid is None:
                #W_dec_hid1 = l_dec_hid1.W
                #b_dec_hid1 = l_dec_hid1.b
                W_dec_hid = l_dec_hid.W
                b_dec_hid = l_dec_hid.b
                W_dec_mu = l_dec_mu.W
                b_dec_mu = l_dec_mu.b
                W_dec_ls = l_dec_logsigma.W
                b_dec_ls = l_dec_logsigma.b
    l_output = nn.layers.ElemwiseSumLayer(l_output_list, coeffs=1./L, name='output')
    return l_enc_mu, l_enc_logsigma, l_dec_mu_list, l_dec_logsigma_list, l_output_list, l_output

# ############################## Main program ################################

def log_likelihood(tgt, mu, ls):
    return T.sum(-(np.float32(0.5 * np.log(2 * np.pi)) + ls)
            - 0.5 * T.sqr(tgt - mu) / T.exp(2 * ls))

def main(L=2, z_dim=5, n_hid=2000, num_epochs=2, binary=True, test_pct=0.04, num_inputs=32000,
        lr=1e-5, kl_term=1, folder="z5h2k"):
    print("Loading data...")
    data, data_avg, data_range = create_dataset(test_pct, num_inputs)
    X_train = data['train']
    X_val = data['val']
    X_test = data['test']
    print X_train.shape
    print X_val.shape
    print X_test.shape
    print X_train[0]
    print data_avg
    print data_range

    #width, height = X_train.shape[2], X_train.shape[3]
    input_var = T.matrix('inputs')

    # Create VAE model
    print("Building model and compiling functions...")
    print("L = {}, z_dim = {}, n_hid = {}, binary={}, kl_term={}".format(L, z_dim, n_hid, binary, kl_term))
    x_dim = num_inputs
    l_z_mu, l_z_ls, l_x_mu_list, l_x_ls_list, l_x_list, l_x = \
           build_vae(input_var, L=L, binary=binary, z_dim=z_dim, n_hid=n_hid, num_inputs=num_inputs)

    def build_loss(deterministic):
        layer_outputs = nn.layers.get_output([l_z_mu, l_z_ls] + l_x_mu_list + l_x_ls_list
                + l_x_list + [l_x], deterministic=deterministic)
        z_mu =  layer_outputs[0]
        z_ls =  layer_outputs[1]
        x_mu =  [] if binary else layer_outputs[2:2+L]
        x_ls =  [] if binary else layer_outputs[2+L:2+2*L]
        x_list =  layer_outputs[2:2+L] if binary else layer_outputs[2+2*L:2+3*L]
        x = layer_outputs[-1]
        # Loss expression has two parts as specified in [1]
        # kl_div = KL divergence between p_theta(z) and p(z|x)
        # - divergence between prior distr and approx posterior of z given x
        # - or how likely we are to see this z when accounting for Gaussian prior
        # logpxz = log p(x|z)
        # - log-likelihood of x given z
        # - in binary case logpxz = cross-entropy
        # - in continuous case, is log-likelihood of seeing the target x under the
        #   Gaussian distribution parameterized by dec_mu, sigma = exp(dec_logsigma)
        kl_div = 0.5 * T.sum(1 + 2*z_ls - T.sqr(z_mu) - T.exp(2 * z_ls))
        if binary:
            logpxz = sum(nn.objectives.binary_crossentropy(x,
                input_var.flatten(2)).sum() for x in x_list) * (-1./L)
            prediction = x_list[0] if deterministic else x
        else:
            logpxz = sum(log_likelihood(input_var.flatten(2), mu, ls)
                for mu, ls in zip(x_mu, x_ls))/L
            prediction = x_mu[0] if deterministic else T.sum(x_mu, axis=0)/L
        loss = -1 * (logpxz + kl_term * kl_div)
        return loss, prediction

    # If there are dropout layers etc these functions return masked or non-masked expressions
    # depending on if they will be used for training or validation/test err calcs
    loss, _ = build_loss(deterministic=False)
    test_loss, test_prediction = build_loss(deterministic=True)

    # ADAM updates
    params = nn.layers.get_all_params(l_x, trainable=True)
    updates = nn.updates.adam(loss, params, learning_rate=lr)
    train_fn = theano.function([input_var], loss, updates=updates)
    val_fn = theano.function([input_var], test_loss)
    
    best_val_err = np.inf
    break_count = 0
    print("Starting training...")
    batch_size = 100
    train_err_list = []
    val_err_list = []
    for epoch in range(num_epochs):
        train_err = 0
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(X_train, batch_size, shuffle=True):
            this_err = train_fn(batch)
            train_err += this_err
            train_batches += 1
        val_err = 0
        val_batches = 0
        for batch in iterate_minibatches(X_val, batch_size, shuffle=False):
            err = val_fn(batch)
            val_err += err
            val_batches += 1
        train_err_list.append(train_err / train_batches)
        val_err_list.append(val_err / val_batches)
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
        if val_err > best_val_err:
            break_count += 1
        else:
            break_count = 0
            best_val_err = val_err
        if break_count > 10:
            break
    test_err = 0
    test_batches = 0
    for batch in iterate_minibatches(X_test, batch_size, shuffle=False):
        err = val_fn(batch)
        test_err += err
        test_batches += 1
    test_err /= test_batches
    print("Final results:")
    print("  test loss:\t\t\t{:.6f}".format(test_err))
   
    def plot_seq(generated_seq, filename):
        plt.plot(generated_seq)
        plt.savefig(filename)
        plt.clf()
    
    def plot_err(train_err, val_err, filename):
        xax = np.arange(0, len(train_err))
        plt.plot(xax, train_err, 'b', label='Train error')
        plt.plot(xax, val_err, 'r', label='Val error')
        plt.legend(loc='upper left')
        plt.title("Train and validation error")
        plt.xlabel("Number of epochs")
        plt.ylabel("Loss")
        plt.savefig(filename)
        plt.clf()



    def write_to_wav(filename, sampling_rate, generated_seq):
        generated_seq = np.array(generated_seq).flatten() * data_range
        generated_seq = generated_seq.astype('int16')
        wavfile.write(filename, sampling_rate, generated_seq)
        np.savetxt(filename[:-3]+'txt', generated_seq)
    
    if not os.path.exists('./samples/'+ str(folder)):
        os.makedirs('./samples/'+str(folder))

    # save some example audio
    num_samples = 5
    sampling_rate = 16000
    X_comp = X_test[:num_samples]
    pred_fn = theano.function([input_var], test_prediction)
    print X_comp.shape
    X_pred = pred_fn(X_comp) #.reshape(-1, 1, width, height)
    for i in range(num_samples):
        orig_file = './samples/'+ str(folder)+'/orig_sample_' + str(i) + '.wav'
        write_to_wav(orig_file, sampling_rate, X_comp[i])
        output_file = './samples/'+str(folder)+'/vae_generated_sample_' + str(i) + '.wav'
        write_to_wav(output_file, sampling_rate, X_pred[i])
        plot_seq(X_pred[i], output_file[:-3]+'png')
    plot_err(train_err_list, val_err_list, './samples/'+str(folder)+'/error.png')
        # get_image_pair(X_comp, X_pred, idx=i, channels=1).save('output_{}.jpg'.format(i))
    

    #This iteratively "reconstructs" a given input, and chains the "reconstructions" together
    #to form a longer audio. Like unrolling the VAE into a Markov chain
    num_iters = 20
    full_gen = X_pred[-1]
    print full_gen.shape
    print X_comp[-1].shape
    last_pred = full_gen
    for i in range(num_iters):        
        next_pred = pred_fn(last_pred.reshape(1, num_inputs))
        full_gen = np.concatenate([full_gen, next_pred.reshape(num_inputs,)])
        last_pred = next_pred
    output_file = './samples/'+str(folder)+'/vae_generated_seq.wav'
    write_to_wav(output_file, sampling_rate, full_gen)
    plot_seq(full_gen, output_file[:-3]+'png')
    
    #sample from the latent space
    if z_dim == 2:
        z_var = T.vector()
        generated_x = nn.layers.get_output(l_x_mu_list[0], {l_z_mu:z_var}, 
                    deterministic=True)
        gen_fn = theano.function([z_var], generated_x)
        interp_gen = np.array([])
        for (x,y),val in np.ndenumerate(np.zeros((19,19))):
            z = np.asarray([norm.ppf(0.05*(x+1)), norm.ppf(0.05*(y+1))],
                    dtype=theano.config.floatX)
            x_gen = gen_fn(z)
            print x_gen.shape
            print interp_gen.shape
            interp_gen = np.concatenate([interp_gen, x_gen.reshape(num_inputs,)])
            
    output_file = './samples/'+str(folder)+'/vae_interp_seq.wav'
    write_to_wav(output_file, sampling_rate, interp_gen)
    plot_seq(interp_gen, output_file[:-3]+'png')


    # save the parameters so they can be loaded for next time
    #print("Saving")
    #fn = 'params_{:.6f}'.format(test_err)
    #np.savez(fn + '.npz', *nn.layers.get_all_param_values(l_x))
    
    """
    # sample from latent space if it's 2d
    if z_dim == 2:
        # functions for generating images given a code (used for visualization)
        # for an given code z, we deterministically take x_mu as the generated data
        # (no Gaussian noise is used to either encode or decode).
        z_var = T.vector()
        if binary:
            generated_x = nn.layers.get_output(l_x, {l_z_mu:z_var}, deterministic=True)
        else:
            generated_x = nn.layers.get_output(l_x_mu_list[0], {l_z_mu:z_var}, 
                    deterministic=True)
        gen_fn = theano.function([z_var], generated_x)
        im = Image.new('L', (width*19,height*19))
        for (x,y),val in np.ndenumerate(np.zeros((19,19))):
            z = np.asarray([norm.ppf(0.05*(x+1)), norm.ppf(0.05*(y+1))],
                    dtype=theano.config.floatX)
            x_gen = gen_fn(z).reshape(-1, 1, width, height)
            im.paste(Image.fromarray(get_image_array(x_gen,0)), (x*width,y*height))
            im.save('gen.jpg')
    """

if __name__ == '__main__':
    # Arguments - integers, except for binary/continous. Default uses binary.
    # Run with option --continuous for continuous output.
    import argparse
    parser = argparse.ArgumentParser(description='Command line options')
    parser.add_argument('--num_epochs', type=int, dest='num_epochs')
    parser.add_argument('--L', type=int, dest='L')
    parser.add_argument('--z_dim', type=int, dest='z_dim')
    parser.add_argument('--n_hid', type=int, dest='n_hid')
    parser.add_argument('--binary', dest='binary', action='store_true')
    parser.add_argument('--continuous', dest='binary', action='store_false')
    parser.set_defaults(binary=False)
    parser.add_argument('--kl_term', type=float, dest='kl_term')
    parser.add_argument('--lr', type=int, dest='lr')
    parser.add_argument('--folder', type=str, dest='folder')
    args = parser.parse_args(sys.argv[1:])
    main(**{k:v for (k,v) in vars(args).items() if v is not None})
