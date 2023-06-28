import tensorflow as tf
from tensorflow.keras import Model
import tensorflow.experimental.numpy as tnp

import raven_utils as rv
from core_tools import ops as K
from core_tools.core import  PREDICT, AUTO
from raven_utils.config.constant import ACC_SAME, ACC_CHOICE_UPPER, ACC_CHOICE_LOWER, ACC_CHOiCE_UPPER_2
from raven_utils.constant import HAMF, HAM, HAMS, T_ACC, AVG_PROP, TARGET, ANSWER
from raven_utils.models.loss.uitls import get_matches
from raven_utils.models.loss.mask import create_uniform_mask, create_uniform_num_pos_arth_mask
from raven_utils.models.uitls_ import RangeMask

def interleave(a):
    return tf.reshape(
        tf.concat([i[..., tf.newaxis] for i in a], axis=-1),
        [tf.shape(a[0])[0], -1])

SIM_MASK = 'sim_mask'

SIM_MASKED = 'sim_masked'

DIFF_ANSWERS = 'diff_answers'

DIFF = 'diff'

SIM_FULL = 'sim_full'

SIM = "sim"
SIM_RATIO = "sim_ratio"

SIM_ENTITY_SUM = "sim_entity_sum"
SIM_ENTITY_SUM_RATIO = "sim_entity_sum_ratio"
SIM_ENTITY_AVG = "sim_entity_avg"

SIM_ENTITY_RATIO = "sim_entity_ratio"


class SimilarityRaven(Model):
    def __init__(
            self,
            mode=create_uniform_num_pos_arth_mask,
            batch_metrics=False,
            sim_metrics=False,
            mask=ANSWER,
            metric_prefix=AUTO,
            mask_slot=AUTO
    ):
        super().__init__()
        self.range_mask = RangeMask()
        self.mode = mode
        self.batch_metrics = batch_metrics
        self.sim_metrics = sim_metrics
        self.mask_ = mask
        if metric_prefix == "auto":
            self.metric_prefix = f"{self.mask_}_" if self.mask_ != ANSWER else ""
        else:
            self.metric_prefix = metric_prefix
        if mask_slot == AUTO:
            mask_slot = self.mask_ == PREDICT
        self.mask_slot = mask_slot

        # self.predict_fn = partial(tf.argmax, axis=-1)

    # INDEX, PREDICT, LABELS
    def call(self, inputs):
        metrics = {}
        target_index = inputs[0] - 8
        predict = inputs[1]
        answers = inputs[2][:, 8:]
        shape = tf.shape(predict)

        target = K.gather(answers, target_index[:, 0])
        target_group = target[:, 0]

        # target_slot = target[..., 1:rv.target.INDEX[0]]
        # predict_slot = predict[..., 1:rv.target.INDEX[0]]
        if self.mask_ == PREDICT:
            slot_mask = predict[..., 1:rv.target.INDEX[0]]
            group = predict[:, 0]
        else:
            # for answer is uses target for acc_same, ham, and other metrics
            # the differnce between target and anwser is in the same ansers that is later used only for upper and lower metrics
            # for uniformity mask the target and answer do not differ
            slot_mask = target[..., 1:rv.target.INDEX[0]]
            group = target_group

        # comp_slice = np.
        target_comp = target[:, 1:rv.target.END_INDEX]
        predict_comp = predict[:, 1:rv.target.END_INDEX]
        answers_comp = answers[:, :, 1:rv.target.END_INDEX]

        full_properties_musks = self.mode(target)
        fpm = tf.concat([full_properties_musks[0], interleave(full_properties_musks[2:])], axis=-1)

        range_mask = self.range_mask(group)
        if self.mask_slot:
            # for predicted slot we need to masked it by group, because model predict all slots
            # for target slot we do no need as target slot is masked at stared
            slot_mask = slot_mask * range_mask
        full_range_mask = K.cat([range_mask, tf.repeat(slot_mask, 3, axis=-1)], axis=-1)

        final_mask = fpm & full_range_mask

        target_masked = target_comp * final_mask
        predict_masked = predict_comp * final_mask
        same = target_masked == predict_masked
        if self.mask_ == ANSWER:
            answer_range_mask = tf.tile(range_mask[:, None], [1, 8, 1])
            answer_slot_mask = answers[..., 1:rv.target.INDEX[0]]
            answer_properties_mask = tf.repeat(answer_slot_mask, 3, axis=-1)

            if self.mask_slot:
                answer_properties_mask = answer_properties_mask * tf.repeat(answer_range_mask, 3, axis=-1)

            full_range_mask = K.cat([
                answer_range_mask,
                answer_properties_mask
            ],
                axis=-1)
            answer_mask = fpm[:, None] & full_range_mask

            predict_answer_masked = predict_comp[:, None] * answer_mask
            answers_masked = answers_comp * answer_mask

            same_answers = predict_answer_masked == answers_masked
        else:
            answers_masked = answers_comp * tf.tile(final_mask[:, None], [1, 8, 1])
            same_answers = predict_masked[:, None] == answers_masked

        # same = K.gather(same_answers, target_index[:, 0])

        diff_answers = ~same_answers
        diff = ~same
        if self.batch_metrics:
            metrics[f"{self.metric_prefix}{DIFF}"] = diff
            metrics[f"{self.metric_prefix}{DIFF_ANSWERS}"] = diff_answers

        acc_same_batch = K.all(same)
        acc_same = K.mean(acc_same_batch)
        self.add_metric(acc_same, f"{self.metric_prefix}{ACC_SAME}")
        metrics[f"{self.metric_prefix}{ACC_SAME}"] = acc_same_batch if self.batch_metrics else acc_same

        hamf_batch = K.sum(predict[:, :101] != target[:, :101])
        hamf = K.mean(hamf_batch)
        self.add_metric(hamf, f"{self.metric_prefix}{HAMF}")
        metrics[f"{self.metric_prefix}{HAMF}"] = hamf_batch if self.batch_metrics else hamf

        ham_sum = K.sum(diff) + (target_group != predict[:, 0])
        total_predictions = K.sum(K.sum(final_mask) + 1)
        accuracy = 1 - (K.sum(ham_sum) / total_predictions)
        self.add_metric(accuracy, f"{self.metric_prefix}{T_ACC}")
        metrics[f"{self.metric_prefix}{T_ACC}"] = ham_sum if self.batch_metrics else accuracy

        ham_batch = ham_sum
        ham = K.mean(ham_batch)
        self.add_metric(ham, f"{self.metric_prefix}{HAM}")
        metrics[f"{self.metric_prefix}{HAM}"] = ham_batch if self.batch_metrics else ham

        # hams_batch = ham_sum / K.array(list(rv.properties.COUNT_ALL.values()))[target_group]
        hams_batch = ham_sum / K.sum(final_mask)
        hams = K.mean(hams_batch)
        self.add_metric(hams, f"{self.metric_prefix}{HAMS}")
        metrics[f"{self.metric_prefix}{HAMS}"] = hams_batch if self.batch_metrics else hams
        self.add_metric(1 - hams, f"{self.metric_prefix}{AVG_PROP}")
        metrics[f"{self.metric_prefix}{AVG_PROP}"] = (1 - hams_batch) if self.batch_metrics else (1 - hams)

        matches, once_matches, more_matches = get_matches(tf.cast(diff_answers, dtype=tf.int32), target_index)

        acc_choice_upper = K.mean(matches)
        self.add_metric(acc_choice_upper, f"{self.metric_prefix}{ACC_CHOICE_UPPER}")
        metrics[f"{self.metric_prefix}{ACC_CHOICE_UPPER}"] = matches if self.batch_metrics else acc_choice_upper

        # sanity check
        acc_choice_upper_2 = (K.sum(once_matches) + K.sum(more_matches)) / shape[0]
        self.add_metric(acc_choice_upper_2, f"{self.metric_prefix}{ACC_CHOiCE_UPPER_2}")

        acc_choice_lower = K.mean(once_matches)
        self.add_metric(acc_choice_lower, f"{self.metric_prefix}{ACC_CHOICE_LOWER}")
        metrics[f"{self.metric_prefix}{ACC_CHOICE_LOWER}"] = once_matches if self.batch_metrics else acc_choice_lower

        if self.sim_metrics:
            sim_full = tf.abs(tf.concat(
                [
                    K.int(target_group != predict[:, 0])[..., None],
                    K.int(target_masked - predict_masked)
                ],
                axis=-1))

            sim_mask = tf.concat(
                [
                    tf.ones((shape[0], 1), dtype=tf.bool),
                    final_mask,
                ],
                axis=-1)
            if self.batch_metrics:
                metrics[f"{self.metric_prefix}{SIM_FULL}"] = sim_full
                metrics[f"{self.metric_prefix}{SIM_MASK}"] = sim_mask

            sim_div = tnp.array(rv.properties.MAX_DIFF)

            sim_avg = K.mean(sim_full[sim_mask])
            self.add_metric(sim_avg, f"{self.metric_prefix}{SIM}")  # average distance per property
            metrics[f"{self.metric_prefix}{SIM}"] = sim_full if self.batch_metrics else sim_avg

            sim_entity_sum_batch = K.sum(sim_full)
            sim_entity_sum = K.mean(sim_entity_sum_batch)
            self.add_metric(sim_entity_sum, f"{self.metric_prefix}{SIM_ENTITY_SUM}")  # average distance per sample
            metrics[
                f"{self.metric_prefix}{SIM_ENTITY_SUM}"] = sim_entity_sum_batch if self.batch_metrics else sim_entity_sum

            sim_entity_sum_ratio_batch = sim_entity_sum_batch / K.sum(sim_mask * sim_div)
            sim_entity_sum_ratio = 1 - K.mean(sim_entity_sum_ratio_batch)
            self.add_metric(sim_entity_sum_ratio,
                            SIM_ENTITY_SUM_RATIO)  # average distance per sample scaled by max diff
            metrics[
                f"{self.metric_prefix}{SIM_ENTITY_SUM_RATIO}"] = sim_entity_sum_ratio_batch if self.batch_metrics else sim_entity_sum_ratio

            sim_entity_avg_batch = sim_entity_sum_batch / K.sum(sim_mask)
            sim_entity_avg = K.mean(sim_entity_avg_batch)
            self.add_metric(sim_entity_avg,
                            f"{self.metric_prefix}{SIM_ENTITY_AVG}")  # average average distance per property in sample
            metrics[
                f"{self.metric_prefix}{SIM_ENTITY_AVG}"] = sim_entity_avg_batch if self.batch_metrics else sim_entity_avg

            sim_scaled = sim_full / sim_div
            sim_ratio = 1 - K.mean(sim_scaled[sim_mask])
            self.add_metric(sim_ratio, f"{self.metric_prefix}{SIM_RATIO}")  # average scaled property distance
            metrics[f"{self.metric_prefix}{SIM_RATIO}"] = sim_scaled if self.batch_metrics else sim_ratio

            sim_entity_ratio_batch = K.sum(sim_scaled) / K.sum(sim_mask)
            sim_entity_ratio = 1 - K.mean(sim_entity_ratio_batch)
            self.add_metric(sim_entity_ratio,
                            f"{self.metric_prefix}{SIM_ENTITY_RATIO}")  # average scaled property distance per sample
            metrics[
                f"{self.metric_prefix}{SIM_ENTITY_RATIO}"] = sim_entity_ratio_batch if self.batch_metrics else sim_entity_ratio

        return metrics

# slot_mask.numpy()

# sim_scaled.numpy()
# import numpy as np
# np.stack(
#     [metrics[f"{self.metric_prefix}{r] for r in metrics if r in ['acc_same', 'acc_choice_upper', 'acc_choice_lower']]
# )
# pr({r:metrics[f"{self.metric_prefix}{r][11] for r in metrics if r in ['acc_same', 'acc_choice_upper', 'acc_choice_lower']})
# # ham_batch.numpy()
# # diff.numpy()
# # same[-3].numpy()
# import numpy as np
# np.stack([target_masked[-3],  predict_masked[-3]])
# # np.stack([target_masked[-3],  target[-3][1:101]])
# np.stack([target_masked[0],  predict_masked[0]])
# # #
# # #
# # # fpm[-3].numpy()
# # # target[-3, 111:].numpy()
# # #
# # np.stack([f[-3] for f in full_properties_musks])
# ham_batch.numpy()

# (target_masked - predict_masked).numpy()
