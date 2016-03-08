import theano
import theano.tensor as T
import lasagne as nn
import h5py

def segment_data(data, example_size=8000, sequence_length=10):
    print data.shape #t
    return [data[i*example_size, (i + 1)*example_size] for i in range(len(data)/example_size - 1)]

    

def create_dataset(test_pct = 0.02):
    path = './XqaJ2Ol5cC4.hdf5'
    with h5py.File(path, 'r') as f:
        dataset = f['features'][0]
    data = {}
    data['train'] = segment_data(dataset[: len(dataset)*(1 - 2*test_pct)])
    data['val'] = dataset[len(dataset)*(1 - 2*test_pct): len(dataset)*(1 - test_pct)]
    data['test'] = dataset[len(dataset)*(1 - test_pct):]
    
    print data.shape

    return data


create_dataset()



