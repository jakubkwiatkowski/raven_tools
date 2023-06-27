import tensorflow as tf

from models_utils import ops as K


def get_matches(diff, target_index):
    diff_sum = K.sum(diff)
    db_argsort = tf.argsort(diff_sum, axis=-1)
    db_sorted = tf.sort(diff_sum)
    same_as_first = db_sorted[:, 0, None] == db_sorted
    more_closest = K.sum(same_as_first) > 1
    once_closest = tf.math.logical_not(more_closest)
    same_as_first_index = tf.where(same_as_first, db_argsort, -1 * tf.ones_like(db_argsort))
    matched_index = same_as_first_index == target_index
    # setting shape needed for TensorFlow graph
    matched_index.set_shape(same_as_first_index.shape)
    # matches
    matches = K.any(matched_index)
    once_matches = matches & once_closest
    more_matches = matches & more_closest
    return matches, once_matches, more_matches

# diff_sum.numpy()
# matches_index.numpy()
# matched_index.numpy()
# db_sorted.numpy()
# db_argsort.numpy()
# diff[3].numpy()
#
# print(list(diff_sum[9].numpy()))
