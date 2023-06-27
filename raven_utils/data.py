import os
from functools import partial

import numpy as np
import tensorflow as tf
from config_utils.globals import STORAGE

from data_utils import ops as D, COLOR, DataSetFromFolder

from ml_utils import lw, lu, pj
from models_utils import INPUTS, TARGET

from raven_utils.config.constant import RAVEN, LABELS, INDEX, FEATURES, RAV_METRICS, IMP_RAV_METRICS, ACC_NO_GROUP, \
    PROPERTY, RAVEN_BIG, IRAVEN, FRAVEN
import raven_utils as rv

from typing import Any

from data_utils import pre, Data, gather, vec, resize, DATASET_SPLIT, nload, DataSetFromFolder
from data_utils.data_generator import DataGenerator
from funcy import identity

SOURCE_PATH = {
    RAVEN_BIG: "RAVEN",
    RAVEN: "RAVEN-10000-release/RAVEN-10000",
    IRAVEN: "data",
    FRAVEN: "data",
}


def get_data_folders(dataset_name):
    return DataSetFromFolder(
        pj(
            STORAGE.dataset,
            dataset_name,
            SOURCE_PATH[dataset_name]
        ),
        file_type="dir"
    )

def get_data(data, batch_size, test=False, steps=None, val_steps=None):
    if val_steps is None:
        val_steps = steps
    fn = identity
    train_target_index = data[4] + 8
    train_generator = DataGenerator({
        INPUTS: Data(data[0], identity),
        TARGET: Data(data[2], identity),
        INDEX: train_target_index[:, None],
        # FEATURES: data[6]
    },
        batch=batch_size,
        steps=steps
    )
    val_target_index = data[5] + 8
    val_data = {
        INPUTS: Data(data[1], identity),
        TARGET: Data(data[3], identity),
        INDEX: val_target_index[:, None],
        # FEATURES: data[7]
    }
    val_generator = DataGenerator(
        val_data,
        batch=batch_size,
        sampler="val",
        steps=val_steps
    )
    return train_generator, val_generator


def get_data_2(
        split=2,
        batch=32,
        path="",
        dataset_name=RAVEN,
        add_properties=False,
):
    path = pj(path, dataset_name)
    properties_path = pj(
        path,
        SOURCE_PATH[dataset_name]
    )
    properties_folder = [os.path.join(properties_path, n) for n in rv.group.NAMES]
    if isinstance(split, int):
        split = DATASET_SPLIT[:split]
    elif isinstance(split, slice):
        split = DATASET_SPLIT[split]

    gen = []
    for i, dname in enumerate(lw(split)):
        target_index = nload(pj(path, f"{dname}_index.npy")) + 8
        data = {
            INPUTS: nload(pj(path, f"{dname}.npy")),
            TARGET: nload(pj(path, f"{dname}_target.npy")),
            INDEX: target_index[:, None],
            # FEATURES: data[7]
        }
        if add_properties:
            properties = DataSetFromFolder(properties_folder[:], file_type="xml", extension=dname)
            data[PROPERTY] = properties
        generator = DataGenerator(
            data,
            batch=batch,
            sampler=dname,
        )
        gen.append(generator)
    return lu(tuple(gen))


def get_fake_target(
        split=2,
        batch=32,
        path="",
        dataset_name=RAVEN,
        add_properties=False,
        mask_answers=True
):
    path = pj(path, dataset_name)
    properties_path = pj(
        path,
        SOURCE_PATH[dataset_name]
    )
    properties_folder = [os.path.join(properties_path, n) for n in rv.group.NAMES]
    if isinstance(split, int):
        split = DATASET_SPLIT[:split]
    elif isinstance(split, slice):
        split = DATASET_SPLIT[split]

    def fake_target(data):
        group = D.init_image(data.shape[0:2] + (1,), mode="int", min=0, max=rv.group.NO)
        slot = D.init_image(data.shape[0:2] + (rv.entity.SUM,), mode="int", min=0, max=2)
        color = D.init_image(data.shape[0:2] + (rv.entity.SUM,), mode="int", min=0, max=rv.properties.SIZE["Color"])
        size = D.init_image(data.shape[0:2] + (rv.entity.SUM,), mode="int", min=0, max=rv.properties.SIZE["Size"])
        type_ = D.init_image(data.shape[0:2] + (rv.entity.SUM,), mode="int", min=0, max=rv.properties.SIZE["Type"])
        rules = D.init_image(data.shape[0:2] + (rv.target.RULES_ATTRIBUTES_ALL_LEN,), mode="int", min=0,
                             max=rv.rules.TYPES_LEN)
        uniformity = D.init_image(data.shape[0:2] + (rv.target.UNIFORMITY_NO,), mode="int", min=0, max=4)

        import numpy as np

        properties = D.interleave([color, size, type_])

        return np.concatenate((group, slot, properties, rules, uniformity), axis=-1)

    inputs_mask = np.array([1] * 8 + [0] * 8, dtype=np.uint8)[None, :, None, None]

    def mask_answers_fn(data):
        return data * inputs_mask

    gen = []
    for i, dname in enumerate(lw(split)):
        target_index = nload(pj(path, f"{dname}_index.npy")) + 8
        inputs_data = nload(pj(path, f"{dname}.npy"))
        data = {
            INPUTS: Data(inputs_data, fn=mask_answers_fn) if mask_answers else inputs_data,
            # TARGET: Data(nload(pj(path, f"{dname}_target.npy")), fn=partial(init_like, mode="int", max=2)),
            TARGET: Data(nload(pj(path, f"{dname}_target.npy")), fn=fake_target),
            INDEX: Data(target_index[:, None], fn=partial(D.init_like, mode="int", min=8, max=16)),
            # FEATURES: data[7]
        }
        if add_properties:
            properties = DataSetFromFolder(properties_folder[:], file_type="xml", extension=dname)
            data[PROPERTY] = properties
        generator = DataGenerator(
            data,
            batch=batch,
            sampler=dname,
        )
        gen.append(generator)
    return lu(tuple(gen))


def get_wrong_target(dataset_name=2, batch=32, path="", add_properties=False):
    path = pj(path, "arr")
    properties_path = pj(path, "RAVEN-10000-release", "RAVEN-10000")
    properties_folder = [os.path.join(properties_path, n) for n in rv.group.NAMES]
    if isinstance(dataset_name, int):
        dataset_name = DATASET_SPLIT[:dataset_name]
    elif isinstance(dataset_name, slice):
        dataset_name = DATASET_SPLIT[dataset_name]

    def wrong_data(data, base=0):
        return np.where(data - 1 >= base, data - 1, data + 1)

    gen = []
    for i, dname in enumerate(lw(dataset_name)):
        target_index = nload(pj(path, f"{dname}_index.npy")) + 8
        data = {
            INPUTS: nload(pj(path, f"{dname}.npy")),
            # TARGET: Data(nload(pj(path, f"{dname}_target.npy")), fn=partial(init_like, mode="int", max=2)),
            TARGET: Data(nload(pj(path, f"{dname}_target.npy")), fn=wrong_data),
            INDEX: Data(target_index[:, None], fn=partial(wrong_data, base=8)),
            # FEATURES: data[7]
        }
        if add_properties:
            properties = DataSetFromFolder(properties_folder[:], file_type="xml", extension=dname)
            data[PROPERTY] = properties
        generator = DataGenerator(
            data,
            batch=batch,
            sampler=dname,
        )
        gen.append(generator)
    return lu(tuple(gen))


