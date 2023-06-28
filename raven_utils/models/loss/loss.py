from core_tools.core import METRIC
from core_tools.core import OUTPUT, TARGET, PREDICT, add_loss, LOSS, FilterList
from tensorflow.keras import Model
# from tensorflow.keras import backend as K

from raven_utils.config.constant import LABELS, INDEX, MASK
from raven_utils.constant import ANSWER
from raven_utils.models.loss.class_ import ClassRavenModel
from raven_utils.models.loss.mask import create_uniform_mask, create_all_mask, create_uniform_num_pos_mask, \
    create_uniform_num_pos_arth_mask
from raven_utils.models.loss.sim import SimilarityRaven


class RavenLoss(Model):
    def __init__(
            self,
            mode=create_uniform_num_pos_arth_mask,
            number_loss=False,
            slot_loss=True,
            group_loss=True,
            lw=(2.0, 1.0),
            plw=None,
            batch_metrics=False,
            sim_metrics=False,
            sim_mask=ANSWER,
            return_prop_mask=False,
            loss_fn=ClassRavenModel,
    ):
        super().__init__()
        self.loss_fn = add_loss(
            loss_fn(
                mode=mode,
                number_loss=number_loss,
                slot_loss=slot_loss,
                group_loss=group_loss,
                plw=plw,
                return_prop_mask=return_prop_mask
            ),
            lw=lw[0],
            name="add_loss",
            filter_=(lambda x: FilterList(x, index=0)) if return_prop_mask else None
        )
        self.metric_fn = SimilarityRaven(
            mode=mode,
            batch_metrics=batch_metrics,
            sim_metrics=sim_metrics,
            mask=sim_mask,
        )
        self.loss_fn_2 = add_loss(
            loss_fn(mode=create_all_mask, number_loss=number_loss, slot_loss=slot_loss,
                    group_loss=group_loss, enable_metrics="c", plw=plw), lw=lw[1], name="class_loss")
        self.mask_prefix = "" if ANSWER else sim_mask

    def call(self, inputs):
        losses = []
        output = inputs[OUTPUT]
        target = inputs[TARGET]
        labels = inputs[LABELS]
        mask = inputs[MASK]
        index = inputs[INDEX]
        predict = inputs[PREDICT]

        target_masked = target[mask]
        output_masked = output[mask]
        losses.append(self.loss_fn([target_masked, output_masked]))

        target_unmasked = target[~mask]
        output_unmasked = output[~mask]
        losses.append(self.loss_fn_2([target_unmasked, output_unmasked]))

        metric = self.metric_fn([index, predict, labels])

        return {**inputs, LOSS: losses, METRIC: metric}
