import json
import numpy as np
import os
import torch
import torch.nn as nn
from torch.autograd import Variable




def suffer_data(data):
    data_x = data['x']
    data_y = data['y']
        # randomly shuffle data
    np.random.seed(100)
    rng_state = np.random.get_state()
    np.random.shuffle(data_x)
    np.random.set_state(rng_state)
    np.random.shuffle(data_y)
    return (data_x, data_y)

def batch_data(data, batch_size):

    data_x = data['x']
    data_y = data['y']

    # randomly shuffle data
    np.random.seed(100)
    rng_state = np.random.get_state()
    np.random.shuffle(data_x)
    np.random.set_state(rng_state)
    np.random.shuffle(data_y)

    # loop through mini-batches
    for i in range(0, len(data_x), batch_size):
        batched_x = data_x[i:i+batch_size]
        batched_y = data_y[i:i+batch_size]
        yield (batched_x, batched_y)


def get_random_batch_sample(data_x, data_y, batch_size):
    num_parts = len(data_x)//batch_size + 1
    if(len(data_x) > batch_size):
        batch_idx = np.random.choice(list(range(num_parts +1)))
        sample_index = batch_idx*batch_size
        if(sample_index + batch_size > len(data_x)):
            return (data_x[sample_index:], data_y[sample_index:])
        else:
            return (data_x[sample_index: sample_index+batch_size], data_y[sample_index: sample_index+batch_size])
    else:
        return (data_x,data_y)


def get_batch_sample(data, batch_size):
    data_x = data['x']
    data_y = data['y']

    np.random.seed(100)
    rng_state = np.random.get_state()
    np.random.shuffle(data_x)
    np.random.set_state(rng_state)
    np.random.shuffle(data_y)

    batched_x = data_x[0:batch_size]
    batched_y = data_y[0:batch_size]
    return (batched_x, batched_y)



def read_data(dataset):


    train_data_dir = os.path.join('data', dataset, 'data', 'train')
    test_data_dir = os.path.join('data', dataset, 'data', 'test')


    clients = []
    groups = []
    train_data = {}
    test_data = {}

    train_files = os.listdir(train_data_dir)
    train_files = [f for f in train_files if f.endswith('.json')]
    for f in train_files:
        file_path = os.path.join(train_data_dir, f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        clients.extend(cdata['users'])
        if 'hierarchies' in cdata:
            groups.extend(cdata['hierarchies'])
        train_data.update(cdata['user_data'])

    test_files = os.listdir(test_data_dir)
    test_files = [f for f in test_files if f.endswith('.json')]
    for f in test_files:
        file_path = os.path.join(test_data_dir, f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        test_data.update(cdata['user_data'])
    clients = list(sorted(train_data.keys()))

    return clients, groups, train_data, test_data

def read_user_data(index,data,dataset):

    id = data[0][index]
    train_data = data[2][id]
    test_data = data[3][id]

    # X_train, y_train, X_test, y_test = train_data['x'], train_data['y'], test_data['x'], test_data['y']
    if(dataset == "Epic"):
        X_train, y_train, X_test, y_test = train_data['x'], train_data['y'], test_data['x'], test_data['y']
        X_train = torch.Tensor(X_train).view(len(X_train), 1024).type(torch.float32)  # use for image
        # X_train = torch.Tensor(X_train).view(len(X_train), 400).type(torch.float32)  # use for amazon
        # X_train = torch.Tensor(X_train).view(len(X_train), 3, 32, 32).type(torch.float32) # use for Digit-five
        y_train = np.array(y_train, dtype=int)
        # X_test = torch.Tensor(X_test).reshape(len(X_test), 3, 32, 32).type(torch.float32)  # use for Digit-five
        X_test = torch.Tensor(X_test).reshape(len(X_test), 1024).type(torch.float32)  # use for image
        # X_test = torch.Tensor(X_test).reshape(len(X_test), 400).type(torch.float32)  # use for amazon
        y_test = np.array(y_test, dtype=int)
        X_train = torch.Tensor(X_train).type(torch.float32)
        y_train = torch.Tensor(y_train).type(torch.int64)
        y_train = y_train.reshape(-1)   # use for office               #  remember change
        X_test = torch.Tensor(X_test).type(torch.float32)
        y_test = torch.Tensor(y_test).type(torch.int64)
        y_test = y_test.reshape(-1)  # use for office

    else:
        X_train = torch.Tensor(X_train).type(torch.float32)
        y_train = torch.Tensor(y_train).type(torch.int64)
        X_test = torch.Tensor(X_test).type(torch.float32)
        y_test = torch.Tensor(y_test).type(torch.int64)

    train_data = [(x, y) for x, y in zip(X_train, y_train)]
    test_data = [(x, y) for x, y in zip(X_test, y_test)]
    return id, train_data, test_data


