import sys
import theano
import theano.tensor as T
import numpy as np
import os
import lasagne as nn
import time
from PIL import Image
from scipy.stats import norm
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import cPickle
import gzip
import pickle 

def load_dataset():
    def load_mnist_images(f1):
        with gzip.open(f1, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        data = data.reshape(-1, 1, 28, 28).transpose(0,1,3,2)
        return data / np.float32(255)
    X_train = load_mnist_images('train-images-idx3-ubyte.gz')
    X_test = load_mnist_images('t10k-images-idx3-ubyte.gz')
    X_train, X_val = X_train[:-10000], X_train[-10000:]
    return X_train, X_val, X_test



def get_image_array(X, index, shp=(28,28), channels=1):
    ret = (X[index] * 255.).reshape(channels,shp[0],shp[1]) \
            .transpose(2,1,0).clip(0,255).astype(np.uint8)
    if channels == 1:
        ret = ret.reshape(shp[1], shp[0])
    return ret

def get_image_pair(X, Xpr, channels=1, idx=-1):
    mode = 'RGB' if channels == 3 else 'L'
    shp=X[0][0].shape
    i = np.random.randint(X.shape[0]) if idx == -1 else idx
    orig = Image.fromarray(get_image_array(X, i, shp, channels), mode=mode)
    ret = Image.new(mode, (orig.size[0], orig.size[1]*2))
    ret.paste(orig, (0,0))
    new = Image.fromarray(get_image_array(Xpr, i, shp, channels), mode=mode)
    ret.paste(new, (0, orig.size[1]))
    return ret


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

############################## Algorithm #################################

class GaussianSampleLayer(nn.layers.MergeLayer):
    def __init__(self, mu, logsigma, rng=None, **kwargs):
        self.rng = rng if rng else RandomStreams(nn.random.get_rng().randint(1,1000))
        super(GaussianSampleLayer, self).__init__([mu, logsigma], **kwargs)
        self.mu = mu
        self.logsigma = logsigma
    
    def get_output_shape_for(self, input_shapes):
        return input_shapes[0]

    def get_output_for(self, inputs, deterministic=False, **kwargs):
        mu, logsigma = inputs
        shape = shape=(self.input_shapes[0][0] or inputs[0].shape[0],
                    self.input_shapes[0][1] or inputs[0].shape[1])
        if deterministic:
            return self.mu
        return self.mu + T.exp(self.logsigma) * self.rng.normal(shape)
        
def build_vae(inputvar, hid_size, z_size, num_passes=2, input_size=(28,28), output_size=784):
    l_input = nn.layers.InputLayer(shape=(None,1,input_size[0],input_size[1]), input_var=inputvar, name='input')    
    l_enc_hid = nn.layers.DenseLayer(l_input, num_units=hid_size, 
        nonlinearity=nn.nonlinearities.tanh, name='enc_hid')
    l_enc_mu = nn.layers.DenseLayer(l_enc_hid, num_units=z_size, nonlinearity=None,
        name='enc_mu')
    l_enc_logsigma = nn.layers.DenseLayer(l_enc_hid, num_units=z_size, nonlinearity=None,
        name='enc_logsigma')
    l_output_list = []
    W_dec = None
    b_dec = None
    W_out = None
    b_out = None

    for i in range(num_passes):
        l_Z = GaussianSampleLayer(l_enc_mu, l_enc_logsigma, name='Z')
        l_dec_hid = nn.layers.DenseLayer(l_Z, num_units=hid_size, 
            nonlinearity=nn.nonlinearities.tanh, 
            W=nn.init.GlorotUniform() if W_dec == None else W_dec, 
            b=nn.init.Constant(0.) if b_dec == None else b_dec,
            name='dec_hid')
        l_output = nn.layers.DenseLayer(l_dec_hid, num_units=output_size,
            nonlinearity=nn.nonlinearities.sigmoid,        
            W=nn.init.GlorotUniform() if W_out == None else W_out,
            b=nn.init.Constant(0.) if b_out == None else b_out,
            name='dec_out')
        l_output_list.append(l_output)
    
    l_output = nn.layers.ElemwiseSumLayer(l_output_list, coeffs=1./num_passes, name='output')
    return l_enc_mu, l_enc_logsigma, l_output_list, l_output


def log_likelihood(tgt, mu, ls):
    return T.sum(-(np.float32(0.5 * np.log(2 * np.pi)) + ls)
        - 0.5 * T.sqr(tgt - mu) / T.exp(2 * ls))

def main(num_passes=2, z_size=2, hid_size=1024, num_epochs=300):
    print '...loading data'
    X_train, X_val, X_test = load_dataset()
    width = X_train[2]
    height = X_train[3]
    size = width*height

    print '...data loaded, building model'
    inputvar = T.tensor4('inputs')
    enc_mu, enc_logsigma, l_output_list, l_output = build_vae(inputvar, hid_size=hid_size, z_size=z_size, 
        num_passes=num_passes)
    
    def build_loss(deterministic):
        layer_outputs = nn.layers.get_output([enc_mu, enc_logsigma] + l_output_list + [l_output])
        z_mu = layer_outputs[0]
        z_ls = layer_outputs[1]
        x_list = layer_outputs[2:2+L]
        x_out = layer_otuputs[-1]

        # Calculating loss via KL and reconstruction term
        kl_div = 0.5 * T.sum(1 + 2*z_ls - T.sqr(z_mu) + T.exp(2*z_ls))
        log_pxz = -1./L * sum(nn.objectives.crossentropy(x, 
            inputvar_flatten(2)).sum() for x in x_list)
        loss = -1. * (kl_div + log_pxz)
        prediction = x_mu[0] if deterministic else 0
        return loss, prediction

    train_loss, _ = build_loss(deterministic=False)
    test_loss, test_pred = build_loss(deterministic=True)
    
    # ADAM updates
    params = nn.layers.get_all_params(l_output, trainable=True)
    updates = nn.updates.adam(train_loss, params, learning_rate=1e-4)
    train_fn = theano.function([input_var], train_loss, updates=updates)
    test_fn = theano.function([input_var], test_loss)

    print '...starting training'
    batch_size = 100
    while epoch <= num_epochs:
        train_batch = 0 
        err = 0
        start_time = time.time()
        for batch in iterate_minibatches(X_train, batch_size, shuffle=True):
            this_err = train_fn(batch)
            err += this_err
            train_batch += 1
        
        val_err = 0
        val_batch = 0
        for batch in iterate_minibatches(X_val, batch_size, shuffle=False):
            this_val_err = test_fn(batch)
            val_err += this_val_err
            val_batch += 1

        epoch += 1
        time_elapsed = start_time - time.time()
        print "Finished %d epochs, %f seconds elapsed"%(epoch, time_elapsed)
        print "Training error = %f"%(err / train_batch)
        print "Validation error = %f"%(val_err / val_batch)

    test_err = 0
    test_batch = 0
    for batch in iterate_minibatches(X_test, batch_size, shuffle=False):
        this_test_err = test_fn(batch)
        test_err += this_test_err
        test_batch += 1
    print "Final test error: %f"%(test_err / test_batch)

            
if __name__ == '__main__':
    main()


































