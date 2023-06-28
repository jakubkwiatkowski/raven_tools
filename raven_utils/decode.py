import numpy as np
from core_tools.core import lw
import tensorflow as tf

def ibin(x):
    return tf.cast(bin(x), dtype=tf.int32)

import raven_utils as rv

def np_split(ary, indices_or_sections, axis=-1):
    return np.split(ary, np.cumsum(indices_or_sections), axis)[:-1]


def output(x, split_fn=np_split, predict_fn_1=np.argmax, predict_fn_2=ibin):
    res = output_divide(x, split_fn=split_fn)
    res = output_predict(res, predict_fn_1=predict_fn_1, predict_fn_2=predict_fn_2)
    return (res[0], res[1]) + tuple(output_properties(res[2], predict_fn=predict_fn_1))


def output_divide(output, split_fn=np_split):
    group_output = output[..., rv.output.GROUP_SLICE_END]
    slot_output = output[..., rv.output.SLOT_SLICE_END]
    properties_output = output[..., rv.output.PROPERTIES_SLICE_END]
    # split into list with tensors for each property
    properties_output_splited = split_fn(properties_output, list(rv.properties.INDEX.values()), axis=-1)
    return group_output, slot_output, properties_output_splited


def output_predict(output, predict_fn_1=np.argmax, predict_fn_2=ibin):
    return predict_fn_1(output[0]), predict_fn_2(output[1]), output[2]


def output_properties(x, predict_fn=np.argmax):
    out_reshaped = []
    for i, out in enumerate(x):
        shape = (-1, rv.entity.SUM, rv.properties.RAW_SIZE[i])
        out_reshaped.append(predict_fn(out.reshape(shape)))
    # list of predictions for each property
    return out_reshaped


def output_result(output, split_fn=np_split, arg_max=np.argmax):
    result = output_properties(output, predict_fn=split_fn)
    res = []
    for i, r in enumerate(result):
        if i == 1:
            res.append(r)
        else:
            res.append(arg_max(r, axis=-1))
    return tuple(res)


def decode_inference(inference, reshape=np.reshape):
    return reshape(inference[rv.inference.SLOT_SLICE],
                   [-1, rv.group.NO, rv.inference.PROPERTY_TRANSFORMATION_NO]), reshape(
        inference[rv.inference.PROPERTIES_SLICE],
        [-1, rv.properties.NO, rv.entity.SUM, rv.inference.PROPERTY_TRANSFORMATION_NO])


def decode_target(target):
    target_group = target[..., 0]
    target_slot = target[..., 1:rv.target.INDEX[0]]
    target_properties = target[..., rv.target.INDEX[0]:rv.target.END_INDEX]
    # Divide into list of tenor for each property.
    target_properties_splited = [
        target_properties[..., ::rv.properties.NO],
        target_properties[..., 1::rv.properties.NO],
        target_properties[..., 2::rv.properties.NO]
    ]
    return target_group, target_slot, target_properties_splited




def decode_target_flat(target):
    t = decode_target(target)
    return t[0], t[1], t[2][0], t[2][1], t[2][2]


def demask(target, mask=None, group=None, zeroes=None):
    if mask is None:
        if group is None:
            group = target[0]
        # todo Use numpy range Mask
        from models.uitls_ import RangeMask
        mask = RangeMask()(group).numpy()
    if zeroes is None:
        return np.concatenate([t[mask] for t in lw(target[1:])])
    return np.concatenate([target[0][None]] + [t * mask for t in lw(target[1:])],axis=-1)


def target_mask(mask,right=1):
    shape = mask.shape
    return np.concatenate([np.ones([shape[0], 1]) ,mask, np.repeat(mask,3,axis=1), np.ones([shape[0], right])],axis=1)


def get_full_range_mask(mask):
    return np.concatenate([mask, np.repeat(mask, 3, axis=-1)], axis=-1)

def compare(target, predict, mask):
    target_comp = target[:, 1:rv.target.END_INDEX]
    predict_comp = predict[:, 1:rv.target.END_INDEX]

    mask = get_full_range_mask(mask)

    target_masked = target_comp * mask
    predict_masked = predict_comp * mask
    return target_masked == predict_masked
