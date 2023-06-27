from functools import partial
from itertools import product

import numpy as np
from funcy import identity

from data_utils import gather, DataGenerator, Data
from data_utils.sampling import DataSampler
from models_utils import init_image as def_init_image, INPUTS, TARGET

import raven_utils.group as group

from data_utils import ops as D

init_image = partial(def_init_image, shape=(16, 8, 80, 80, 1))


def get_val_index(no=group.NO, base=3,add_end=False):
    indexes = np.arange(no) * 2000 + base
    if add_end:
        indexes = np.concatenate([indexes, no*2000])
    return indexes


def get_matrix(inputs, index):
    return np.concatenate([inputs[:, :8], gather(inputs, index[:, 0])[:, None]], axis=1)


def get_matrix_from_data(x):
    inputs = x["inputs"]
    index = x["index"]
    return get_matrix(inputs, index)


def get_data_class(data, batch_size=128):
    fn = identity
    shape = data[0].shape
    train_generator = DataGenerator(
        {
            INPUTS: Data(data[0], fn),
            TARGET: Data(data[2], fn),
        },
        sampler=DataSampler(np.array(list(product(np.arange(shape[0]), np.arange(shape[1]))))),
        batch=batch_size
    )
    shape = data[1].shape
    val_generator = DataGenerator(
        {
            INPUTS: Data(data[1], fn),
            TARGET: Data(data[3], fn),
        },
        sampler=DataSampler(np.array(list(product(np.arange(shape[0]), np.arange(shape[1])))), shuffle=False),
        batch=batch_size
    )
    return train_generator, val_generator


def compare_from_result(result, data):
    data = data.data.data
    answer = D.gather(data['target'].data, data['index'].data[:, 0])
    import raven_utils as rv
    predict = result['predict']
    predict_mask = result['predict_mask']
    return np.all(rv.decode.compare(answer[:len(predict)], predict, predict_mask), axis=-1)
