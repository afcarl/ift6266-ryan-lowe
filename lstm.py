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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import wave

def create_dataset(test_pct, num_inputs, seq_len):
    """ Takes the raw HDF5 file and converts it to a numpy array.
    """
    path = './XqaJ2Ol5cC4.hdf5'
    with h5py.File(path, 'r') as f:
        dataset = f['features'][0].flatten()
    
    def segment_data(data, example_size, seq_len):
        """ Data enters as a 1-D array, and split up into chunks of size
        example_size, which are separated into sequences of length seq_len.
        """
        data = (data - sum(data) / len(data)) /  (max(data) - min(data))
        num_seq = int(len(data) / (example_size * seq_len))
        data = data[: num_seq * seq_len * example_size]
        return np.array(data).reshape(num_seq, seq_len, example_size).astype('float32')
    train_data = dataset[: len(dataset)*(1 - 2*test_pct)]
    data_avg = sum(train_data) / len(train_data)
    data_range = max(train_data) - min(train_data)
    data = {}
    data['train'] = segment_data(dataset[: len(dataset)*(1 - 2*test_pct)], 
            num_inputs, seq_len)
    data['val'] = segment_data(dataset[len(dataset)*(1 - 2*test_pct): 
        len(dataset)*(1 - test_pct)], num_inputs, seq_len)
    data['test'] = segment_data(dataset[len(dataset)*(1 - test_pct):],
            num_inputs, seq_len)
    return data, data_avg, data_range

def generate(X_test, model, l_out, out_fn, length, data_avg, data_range):
    """ Given a trained model, generates output audio using the start
    of the test set as a seed.
    """
    set_all_param_values(l_out, model)
    seed = X_test[0:1]
    generated_seq = []
    prev_input = seed
    for x in range(0, length):
        next_input = out_fn(prev_input)
        generated_seq.append(next_input.flatten()[0:8000])
        prev_input = next_input
    #generated_seq = generated_seq.flatten()
    generated_seq = np.array(generated_seq).flatten() * data_range
    generated_seq = generated_seq.astype('int16')
    return generated_seq

def write_seq(output_file, generated_seq, sampling_rate=16000):
    print np.max(generated_seq)
    print np.min(generated_seq)
    print np.mean(generated_seq)
    #with open(output_file, 'w') as f1:
    #    f1.setnchannel(1)
    #    f1.setsamplewidth(4)
    #    f1.setframerate(sampling_rate)
    #    f1.writeframes(generated_seq)
    wavfile.write(output_file, sampling_rate, generated_seq)

def plot_seq(generated_seq):
    print generated_seq.shape
    plt.plot(generated_seq)
    plt.savefig('lstm.png')


def main(n_hid=500, batch_size=16, num_epochs=2, lr=0.01, seq_len=10,
        num_inputs=8000, testpct=0.01, length=50, 
        output_file='lstm_gen.wav'):
    """ Builds the LSTM that is trained on the data.
    """
    print "...loading data"
    data, data_avg, data_range = create_dataset(testpct, num_inputs, seq_len)
    print "...building model"
    X_train = data['train']
    X_val = data['val']
    X_test = data['test']
    X = T.tensor3('X')

    l_in = InputLayer((None, seq_len, num_inputs))
    l_hid1 = LSTMLayer(l_in, n_hid, nonlinearity=nn.nonlinearities.tanh)
    l_hid2 = LSTMLayer(l_hid1, n_hid, nonlinearity=nn.nonlinearities.tanh)
    l_hid3 = LSTMLayer(l_hid2, n_hid, nonlinearity=nn.nonlinearities.tanh)
    l_shp = ReshapeLayer(l_hid3, (-1, n_hid))
    l_dense = DenseLayer(l_shp, num_units=num_inputs, nonlinearity=tanh)
    l_out = ReshapeLayer(l_dense, (-1, seq_len, num_inputs))
    net_out = get_output(l_out, X)

    pred_values = net_out[:, 0: seq_len - 1, :]
    target_values = X[:, 1:, :]
    loss = T.mean((pred_values - target_values)**2)
    loss_fn = theano.function([X], loss)
    out_fn = theano.function([X], net_out)

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
            + "  Train loss " + str(train_loss) \
            + "  Val loss " + str(this_val_loss) \
            + "  Total time: " + str(t)
    
    print "...generating sequence."
    generated_seq = generate(X_test, best_model, l_out, out_fn, length, data_avg, data_range)
    plot_seq(generated_seq)
    np.savetxt('lstm_seq.txt', generated_seq)
    write_seq(output_file, generated_seq)


if __name__ == '__main__':
    main()


''' 
def keras_model():
    import keras.models as krm
    import keras.layers.core as klc
    import keras.layers.recurrent as krr
    m = krm.Sequential()
    m.add(krr.LSTM(n_hid, input_shape=(num_inputs,), activation='tanh'))
    m.add(krr.LSTM(n_hid, activation='tanh'))
    m.add(krr.LSTM(n_hid, activation='tanh'))
    m.add(klc.Flatten())
    m.add(klc.Dense(num_inputs, activation='tanh'))

    m.compile(optimizer='adam', loss='mse')
    return m
'''
    


