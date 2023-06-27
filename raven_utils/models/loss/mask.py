import tensorflow.experimental.numpy as tnp

import tensorflow as tf

import raven_utils as rv
from raven_utils.constant import NUM_POS_ARTH, NUM_POS, NUM, UNIFORM, CHANGE, ALL


def get_no_constant_mask(target, properties=-3, layout="both"):
    first_layout = slice(rv.target.END_INDEX, rv.target.END_INDEX + rv.rules.ATTRIBUTES_LEN)
    second_layout = slice(rv.target.END_INDEX + rv.rules.ATTRIBUTES_LEN,
                          rv.target.END_INDEX + (rv.rules.ATTRIBUTES_LEN * 2))
    prop_slise = slice(properties) if properties > 0 else slice(properties, None)
    left = target[:, first_layout][:, prop_slise]
    right = target[:, second_layout][:, prop_slise]
    if layout == "left":
        attrs = left
    elif layout == "right":
        attrs = right
    else:
        attrs = tf.concat([left, right], axis=-1)
    return attrs > 0


def create_change_mask(target):
    properties_mask = get_no_constant_mask(target)
    return [create_mask(properties_mask, i) for i, _ in enumerate(rv.rules.ATTRIBUTES)]


def create_uniform_mask(target):
    u_mask = lambda i: tf.tile(target[:, rv.target.UNIFORMITY_INDEX + i, None] == 3, [1, rv.rules.ATTRIBUTES_LEN])
    properties_mask = tf.concat([u_mask(0), u_mask(1)], axis=-1) | get_no_constant_mask(target, 0)
    return [create_mask(properties_mask, i) for i, _ in enumerate(rv.rules.ATTRIBUTES)]


# def create_uniform_mask_2(target):
#     u_mask = lambda i: tf.tile(target[:, rv.target.UNIFORMITY_INDEX + i, None] == 3, [1, rv.rules.ATTRIBUTES_LEN])
#     properties_mask = tf.concat([u_mask(0), u_mask(1)], axis=-1) | get_no_constant_mask(target,0) | tf.reduce_all(
#         get_no_constant_mask(target, 2))
#     return [create_mask(properties_mask, i) for i, _ in enumerate(rv.rules.ATTRIBUTES)]

# rules for randomness/non-uniformity (and)
# 1. uniformity < 3
# 2. attribute of entity is constant
# 3. number or position attribute of layout is non constant

# rules for uniformity (or)
# 1. uniformity = 3
# 2. attribute of entity is non-constant
# 3. number or position attribute of layout is constant

def create_uniform_num_pos_mask(target):
    attr = rv.rules.ATTRIBUTES[2:]
    u_mask = lambda i: tf.tile(target[:, rv.target.UNIFORMITY_INDEX + i, None] == 3, [1, len(attr)])
    first_layout_num_pos = tf.tile(
        ~tf.reduce_any(get_no_constant_mask(target, properties=2, layout="left"), axis=-1, keepdims=True), [1, 3])
    second_layout_num_pos = tf.tile(
        ~tf.reduce_any(get_no_constant_mask(target, properties=2, layout="right"), axis=-1, keepdims=True), [1, 3])
    num_pos_mask = tf.concat([first_layout_num_pos, second_layout_num_pos], axis=-1)
    properties_mask = tf.concat([u_mask(0), u_mask(1)], axis=-1) | get_no_constant_mask(target) | num_pos_mask
    return [tf.ones((tf.shape(target)[0], rv.entity.SUM), dtype=tf.bool)] * 2 + [create_mask(properties_mask, i) for
                                                                                 i, _ in enumerate(attr)]


# rules for randomness/non-uniformity (and)
# 1. uniformity < 3
# 2. attribute of entity is constant
# 3. number attribute of layout is non constant

# rules for uniformity (or)
# 1. uniformity = 3
# 2. attribute of entity is non-constant
# 3. number attribute of layout is constant

def create_uniform_num_mask(target):
    attr = rv.rules.ATTRIBUTES[2:]
    u_mask = lambda i: tf.tile(target[:, rv.target.UNIFORMITY_INDEX + i, None] == 3, [1, len(attr)])
    first_layout_num = tf.tile(~get_no_constant_mask(target, properties=1, layout="left"), [1, 3])
    second_layout_num = tf.tile(~get_no_constant_mask(target, properties=1, layout="right"), [1, 3])
    num_mask = tf.concat([first_layout_num, second_layout_num], axis=-1)
    properties_mask = tf.concat([u_mask(0), u_mask(1)], axis=-1) | get_no_constant_mask(target) | num_mask
    return [tf.ones((tf.shape(target)[0], rv.entity.SUM), dtype=tf.bool)] * 2 + [create_mask(properties_mask, i) for
                                                                                 i, _ in enumerate(attr)]


# rules for randomness/non-uniformity (resample) (and)
# 1. uniformity < 3
# 2. attribute of entity is constant
# 3. number attribute of layout is non constant
# 4. position attribute of layout is arithmetic

# rules for uniformity (nonresample) (or)
# 1. uniformity = 3
# 2. attribute of entity is non-constant
# 3. (number attribute of layout is constant) and (position attribute of layout is not arithmetic)


def create_uniform_num_pos_arth_mask(target):
    attr = rv.rules.ATTRIBUTES[2:]
    u_mask = lambda i: tf.tile(target[:, rv.target.UNIFORMITY_INDEX + i, None] == 3, [1, len(attr)])
    first_layout_num = tf.tile(get_no_constant_mask(target, properties=1, layout="left"), [1, 3])
    second_layout_num = tf.tile(get_no_constant_mask(target, properties=1, layout="right"), [1, 3])
    num_non_constant = tf.concat([first_layout_num, second_layout_num], axis=-1)

    first_layout_pos_arith = tf.tile(get_arithmetic_mask(target, properties=slice(1, 2), layout="left"),
                                     [1, 3])
    second_layout_pos_arith = tf.tile(
        get_arithmetic_mask(target, properties=slice(1, 2), layout="right"), [1, 3])

    pos_arith = tf.concat([first_layout_pos_arith, second_layout_pos_arith], axis=-1)

    properties_mask = tf.concat([u_mask(0), u_mask(1)], axis=-1) | get_no_constant_mask(
        target) | ~(num_non_constant | pos_arith)
    return [tf.ones((tf.shape(target)[0], rv.entity.SUM), dtype=tf.bool)] * 2 + [create_mask(properties_mask, i) for
                                                                                 i, _ in enumerate(attr)]


def get_arithmetic_mask(target, properties=-3, layout="both"):
    first_layout = slice(rv.target.END_INDEX, rv.target.END_INDEX + rv.rules.ATTRIBUTES_LEN)
    second_layout = slice(rv.target.END_INDEX + rv.rules.ATTRIBUTES_LEN,
                          rv.target.END_INDEX + (rv.rules.ATTRIBUTES_LEN * 2))
    if isinstance(properties, int):
        prop_slise = slice(properties) if properties > 0 else slice(properties, None)
    else:
        prop_slise = properties
    left = target[:, first_layout][:, prop_slise]
    right = target[:, second_layout][:, prop_slise]
    if layout == "left":
        attrs = left
    elif layout == "right":
        attrs = right
    else:
        attrs = tf.concat([left, right], axis=-1)
    return attrs == 1


def create_all_mask(target):
    return [
        tf.cast(tf.ones(tf.stack([tf.shape(target)[0], rv.entity.SUM])), dtype=tf.bool) for i, _ in
        enumerate(rv.rules.ATTRIBUTES)]


def create_mask(rules, i):
    shape = tf.shape(rules)
    mask_1 = tf.tile(rules[:, i][None], [len(rv.target.FIRST_LAYOUT), 1])
    mask_2 = tf.tile(rules[:, i + int(rules.shape[-1] / 2)][None], [len(rv.target.SECOND_LAYOUT), 1])
    full_mask_1 = tf.scatter_nd(tnp.array(rv.target.FIRST_LAYOUT)[:, None], mask_1, shape=(rv.entity.SUM, shape[0]))
    full_mask_2 = tf.tensor_scatter_nd_update(full_mask_1, tnp.array(rv.target.SECOND_LAYOUT)[:, None], mask_2)
    return tf.transpose(full_mask_2)


LOSS_MODE = {
    ALL: create_all_mask,
    CHANGE: create_change_mask,
    UNIFORM: create_uniform_mask,
    NUM: create_uniform_num_mask,
    NUM_POS: create_uniform_num_pos_mask,
    NUM_POS_ARTH: create_uniform_num_pos_arth_mask,

}
