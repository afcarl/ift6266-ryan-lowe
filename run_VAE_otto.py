import numpy as np
import time
import os
from VRAE import VRAE
import cPickle
import gzip
import h5py
from scipy.io import wavfile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import wave

np.random.seed(1)

hu_encoder = 400
hu_decoder = 400
n_latent = 20
continuous = True
n_epochs = 40

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

data, data_avg, data_range = create_dataset(0.04, 32000)
x_train = data['train']
x_valid = data['val']

"""
if continuous:
    print "Loading Freyface data"
    # Retrieved from: http://deeplearning.net/data/mnist/mnist.pkl.gz
    f = open('freyfaces.pkl', 'rb')
    x = cPickle.load(f)
    f.close()
    x_train = x[:1500]
    x_valid = x[1500:]
else:
    print "Loading MNIST data"
    # Retrieved from: http://deeplearning.net/data/mnist/mnist.pkl.gz
    f = gzip.open('mnist.pkl.gz', 'rb')
    (x_train, t_train), (x_valid, t_valid), (x_test, t_test) = cPickle.load(f)
    f.close()
"""

path = "./"

print "instantiating model"
b1 = 0.05
b2 = 0.001
lr = 0.001
batch_size = 100
sigma_init = 0.01
num_inputs = 32000
model = VRAE(hu_encoder, hu_decoder, x_train, n_latent, b1, b2, lr, sigma_init, batch_size)


batch_order = np.arange(int(model.N / model.batch_size))
epoch = 0
LB_list = []

if os.path.isfile(path + "params.pkl"):
    print "Restarting from earlier saved parameters!"
    model.load_parameters(path)
    LB_list = np.load(path + "LB_list.npy")
    epoch = len(LB_list)

if __name__ == "__main__":
    print "iterating"
    while epoch < n_epochs:
        epoch += 1
        start = time.time()
        np.random.shuffle(batch_order)
        LB = 0.

        for batch in batch_order:
            batch_LB = model.update(batch, epoch)
            LB += batch_LB

        LB /= len(batch_order)

        LB_list = np.append(LB_list, LB)
        print "Epoch {0} finished. LB: {1}, time: {2}".format(epoch, LB, time.time() - start)
        np.save(path + "LB_list.npy", LB_list)
        #model.save_parameters(path)

    valid_LB = model.likelihood(x_valid)
    print "LB on validation set: {0}".format(valid_LB)



