"""
Variational Autoencoder

Written by Ryan Lowe in conjunction with Jean Harb
IFT 6266
"""

import numpy as np
import pickle as pkl
import theano, pdb, gzip
import theano.tensor as T
import lasagne, time, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from matplotlib.backends.backend_pdf import PdfPages
import wave

from lstm import create_dataset, write_seq, plot_seq

def main():
    testpct = 0.04
    num_inputs = 32000 # 2 second time intervals
    seq_len = 1 # since we are modeling it with a non-sequential VAE
    print "...loading data"
    data, data_avg, data_range = create_dataset(testpct, num_inputs, seq_len)
    print "...building model"
    train_set = data['train']
    val_set = data['val']
    test_set = data['test']

    print train_set.shape
    print val_set.shape
    print test_set.shape
    
    

    NUM_SAMPLES=3
    NUM_HIDDEN=500
    NUM_EPOCHS=300
    BATCH_SIZE=100
    Z_SIZE=2
    x = T.matrix()
    index = T.iscalar()
    shared_x = theano.shared(train_set.astype("float32"), borrow=True)
    rng = RandomStreams(np.random.randint(1,2147462579))

    enc_hidden_layer = lasagne.layers.DenseLayer((None, num_inputs), num_units=NUM_HIDDEN, nonlinearity=T.tanh, name="hid_enc")
    z_out = lasagne.layers.DenseLayer(enc_hidden_layer, num_units=Z_SIZE*2, nonlinearity=None, name="out_enc")
    params = [i for i in enc_hidden_layer.params] + [i for i in z_out.params]


    z_mu_sig = z_out.get_output_for(enc_hidden_layer.get_output_for(x))
    mu = z_mu_sig[:, :Z_SIZE]
    sig = z_mu_sig[:, Z_SIZE:]

    dec_hidden_layer = lasagne.layers.DenseLayer((None, Z_SIZE), num_units=NUM_HIDDEN, nonlinearity=T.tanh, name="hid_dec")
    x_out = lasagne.layers.DenseLayer(dec_hidden_layer, num_units=num_inputs, nonlinearity=T.nnet.sigmoid, name="out_dec")
    params += [i for i in dec_hidden_layer.params] + [i for i in x_out.params]

    def step():
        z = mu + T.exp(sig)*rng.normal((mu.shape[0], mu.shape[1]))
        output = x_out.get_output_for(dec_hidden_layer.get_output_for(z))
        cost = T.sum((1-x)*T.log(1-output)+x*T.log(output))
        # could do output > 0.5 to make binary
        return output, cost

    all_samples_and_cost, _ = theano.scan(step, n_steps=NUM_SAMPLES)

    samples = all_samples_and_cost[0]
    costs = all_samples_and_cost[1]

    kl_div = 0.5 * T.sum(1 + 2*sig - T.sqr(mu) - T.exp(2 * sig))
    loss = -costs.mean() - kl_div
    updates = lasagne.updates.adam(loss, params, learning_rate=0.0001)
    train_function = theano.function([index], loss, updates=updates, givens={x: shared_x[index*BATCH_SIZE:BATCH_SIZE*(1+index)]})
    samples = theano.function([index], [samples, mu], givens={x: shared_x[index:index+1]})

    man_sample = x_out.get_output_for(dec_hidden_layer.get_output_for(mu))

    get_manifold_samples = theano.function([z_mu_sig], man_sample)

    for i in range(NUM_EPOCHS):
        t = time.time()
        for j in range(train_set.shape[0]/BATCH_SIZE):
            l = train_function(j)
            if j % 100 == 0:
                print "\b"*100, i, j, l,
                sys.stdout.flush()
        all_examples = []
        #sample generation code hasn't actually been converted yet: still have to fix bug with
        #model compilation
        for k in range(10):
            random_index = np.random.randint(0, train_set.shape[0])
            random_samples, muu = samples(random_index)
            random_samples = random_samples.reshape((NUM_SAMPLES, 28, 28))
            print muu
            examples = np.concatenate([train_set[0][random_index].reshape(1, 28, 28), random_samples], axis=0)
            all_examples += [examples]
        print "%.2f secs" % (time.time() - t)

        all_examples = np.array(all_examples)
        photo = np.concatenate(all_examples[0], axis=0)#.reshape((28, -1))
        for j in range(1, 10):
            photo = np.concatenate([photo, np.concatenate(all_examples[j], axis=0)], axis=1)
        #photo = (photo > 0.5).astype(int)
        fig, ax1 = plt.subplots()
        ax1.imshow(photo, cmap=cm.Greys)
        pp = PdfPages("VAE_samples/samples_%d" % i)
        pp.savefig(fig)
        pp.close()
        plt.close()
        #plt.show()
        if Z_SIZE == 2:
            step_ = 0.3
            mu_range = np.arange(-3, 3+step_, step_, dtype="float32")
            all_examples = []
            for k in mu_range:
                manifold_sample = []
                for l in mu_range:
                    manifold_sample += [get_manifold_samples([[k, l]])[0].reshape(28, 28)]
                    #print manifold_sample[0].shape
                all_examples += [np.concatenate(manifold_sample, axis=0)]
            all_examples = np.array(all_examples)
            photo = np.concatenate(all_examples, axis=1)
            fig, ax1 = plt.subplots()
            ax1.imshow(photo, cmap=cm.Greys)
            pp = PdfPages("VAE_images/manifold_%d" % i)
            pp.savefig(fig)
            pp.close()
            plt.close()


if __name__ == "__main__":
    main()
