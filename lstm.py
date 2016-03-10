from __future__ import division
import theano
import theano.tensor as T
import lasagne as nn
import h5py
import numpy as np
from lasagne.layers import *
from lasagne.init import *
from lasagne.nonlinearities import *
from lasagne.objectives import *
from lasagne.updates import *
from scipy.io import wavfile
from time import time

def segment_data(data, example_size, seq_len):
    data = (data - sum(data) / len(data)) /  (max(data) - min(data))
    #data = [(d - sum(data)/len(data)) /  (max(data) - min(data)) for d in data]
    num_seq = int(len(data) / (example_size * seq_len))
    data = data[: num_seq * seq_len * example_size]
    return np.array(data).reshape(num_seq, seq_len, example_size).astype('float32')

def create_dataset(test_pct, num_inputs, seq_len):
    path = './XqaJ2Ol5cC4.hdf5'
    with h5py.File(path, 'r') as f:
        dataset = f['features'][0].flatten()
    data = {}
    data['train'] = segment_data(dataset[: len(dataset)*(1 - 2*test_pct)], 
            num_inputs, seq_len)
    data['val'] = segment_data(dataset[len(dataset)*(1 - 2*test_pct): 
        len(dataset)*(1 - test_pct)], num_inputs, seq_len)
    data['test'] = segment_data(dataset[len(dataset)*(1 - test_pct):],
            num_inputs, seq_len)
    return data

def generate(X_test, model, l_out, val_fn, length):
    set_all_param_values(l_out, model)
    seed = X_test[0:1]
    generated_seq = []
    prev_input = seed
    for x in range(0, length):
        next_input = val_fn(prev_input)
        generated_seq.append(next_input.flatten()[0])
        prev_input = next_input
    return generated_seq

def write_seq(generated_seq, output_file):
    pass

def main(n_hid=500, batch_size=16, num_epochs=200, lr=0.01, seq_len=10,
        num_inputs=8000, testpct=0.01, length=10000, 
        output_file='lstm_gen.wav'):
    print "...loading data"
    data = create_dataset(testpct, num_inputs, seq_len)
    print "...building model"
    X_train = data['train']
    X_val = data['val']
    X_test = data['test']
    X = T.tensor3('X')

    l_in = InputLayer((None, seq_len, num_inputs))
    l_hid1 = LSTMLayer(l_in, n_hid, nonlinearity=nn.nonlinearities.tanh)
    l_hid2 = LSTMLayer(l_hid1, n_hid, nonlinearity=nn.nonlinearities.tanh)
    l_hid3 = LSTMLayer(l_hid2, n_hid, nonlinearity=nn.nonlinearities.tanh)
    l_shp = ReshapeLayer(l_hid3, (-1, num_units))
    l_dense = DenseLayer(l_shp, num_units=num_inputs, nonlinearity=tanh)
    l_out = ReshapeLayer(l_dense, (-1, seq_len, num_inputs))

    pred_values = get_output(l_out, X).flatten()
    target_values = X[:,1::,:]
    loss = T.mean((pred_values - target_values)**2)
    loss_fn = theano.function([X], loss)

    params = get_all_params(l_out, trainable=True)
    grads = T.grad(loss, params)
    updates = adagrad(grads, params, lr)
    train_fn = theano.function([X], loss, updates=updates)
    val_fn = theano.function([X], loss)
    
    print "...training model"
    epoch = 0
    t0 = time()
    best_val_loss = float('inf')
    best_model = None
    num_batches = int(len(X_train) / batch_size) - 1
    while epoch < num_epochs:
        epoch += 1
        train_losses = []
        for i in range(num_batches):
            X_train_batch = X_train[i*batch_size: (i + 1)*batch_size]
            train_losses = train_fn(X_train_batch)
        train_loss = train_losses.mean()

        this_val_loss = val_fn(X_val)
        if this_val_loss < best_val_loss:
            best_val_loss = this_val_loss
            best_model = get_all_param_values(l_out)
        t = time() - t0
        print "Epoch " + str(epoch) \
            + "Train loss " + str(train_loss) \
            + "Val loss " + str(val_loss) \
            + "Total time: " + str(t)
    
    generated_seq = generate(X_test, best_model, l_out, val_fn, length)
    write_seq(generated_seq, output_file)



if __name__ == '__main__':
    main()





