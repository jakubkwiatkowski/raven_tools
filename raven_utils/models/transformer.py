from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from loguru import logger
from tensorflow.keras.layers import Lambda, Activation, Dense

from grid_transformer import GridTransformer
from grid_transformer.grid_transformer import get_model_output_names
from ml_utils import filter_keys, il, lu, lw

from raven_utils.constant import NUM_POS_ARTH, TARGET, ANSWER, DIRECT, FULL_JOIN, SEPARATE, LATENT
from raven_utils.models.loss import RavenLoss, create_uniform_mask, PredictModel, NonMaskedMetricRaven, create_all_mask, \
    ProbMetric, create_uniform_num_pos_arth_mask
from data_utils import OUTPUT, PREDICT, ims, INDEX
from models_utils import ops as K

from data_utils import DataGenerator, LOSS
from models_utils import DictModel, build_functional, Last, bm, \
    Predictor, AUGMENTATION, Trainer, SubClassingModel, CROSS_ENTROPY, BatchModel, ACC

from raven_utils.models.loss.contrastive import ContrastiveMetric
from raven_utils.models.loss.mask import LOSS_MODE
from raven_utils.models.loss.prob_dist import ProbDistMetric, reverse_index_loss, contrastive_loss
from raven_utils.models.select_ import ChoiceMaker2, ChoiceMaker1, ChoiceMaker, DCPMetric, CLASS_IMAGE
from experiment_utils.keras_model import load_weights as model_load_weights
from models_utils.regularization import Regularization

from raven_utils.params import PROB

PREDICT_MASK = "predict_mask"


def get_rav_trans(
        data,
        loss_mode=NUM_POS_ARTH,
        loss_weight=2.0,
        number_loss=False,
        dry_run="auto",
        non_masked_metrics=False,
        prob_metric=False,
        plw=None,
        additional_output=None,
        return_prop_mask=False,
        batch_metrics=False,
        sim_metrics=False,
        sim_mask=ANSWER,
        **kwargs):
    if isinstance(loss_weight, float):
        loss_weight = (loss_weight, 1.0)

    trans_raven = build_functional(
        model=GridTransformer,
        inputs_=data[0] if isinstance(data, DataGenerator) else data,
        batch_=None,
        dry_run=dry_run,
        **kwargs
    )

    if isinstance(loss_mode, str):
        loss_mode = LOSS_MODE[loss_mode]

    if il(trans_raven):
        if 'image' in trans_raven[1]:
            for im in trans_raven[1]['image'][:3]:
                if im.ndim == 3:
                    im = im[None]
                ims(im.transpose((0, 3, 1, 2)))
                # ims(im.transpose((0, 3, 1, 2)), path=path)
        trans_raven = trans_raven[0]
    if loss_weight is not None:
        loss = [
            #  SimilarityRaven in RavenLoss always make loss mask with target
            RavenLoss(
                mode=loss_mode,
                number_loss=number_loss,
                lw=loss_weight,
                plw=plw,
                batch_metrics=batch_metrics,
                sim_metrics=sim_metrics,
                sim_mask=sm,
                return_prop_mask=return_prop_mask
            ) for sm in lw(sim_mask)
        ]
    else:
        loss = []
    if non_masked_metrics:
        loss = [NonMaskedMetricRaven(mode=create_all_mask)] + loss
    if prob_metric:
        loss = [
                   # 1. ProbMetrics is for comparing grand truth with probability prediction. It already used target,
                   # so the loss mask  could be made also with target and use the uniformity from target.
                   # On the other hanad the ProbDistMetric is for comparing probability prediction with probability prediction,
                   # so there is no target, and moddel do not have the unfiormity from target. So there is no uniformity loss mask
                   # 2. Also in this case ther loss is used that always compares the only the prediction and target and runs it 8 times,
                   # so the target is automatically answer.

                   ProbMetric(
                       batch_metrics=batch_metrics,
                       enable_metrics="predict_prob",
                       mask=PREDICT
                   ),
                   ProbMetric(
                       batch_metrics=batch_metrics
                   ),
                   ProbMetric(
                       batch_metrics=batch_metrics,
                       loss_type="kl_2",
                       enable_metrics="kl_2"
                   ),
                   ProbMetric(
                       batch_metrics=batch_metrics,
                       loss_type="kl",
                       enable_metrics="kl"
                   ),
               ] + loss
    return Trainer(
        model=trans_raven,
        loss=lu(loss),
        model_wrap=False,
        predict=DictModel(SubClassingModel([lambda x: x[:, -1], PredictModel()]), in_=OUTPUT,
                          out=[PREDICT, "predict_mask"], name="pred"),
        loss_wrap=False,
        add_loss=False,
        output=get_model_output_names(
            None,
            additional_names=additional_output,
            **filter_keys(
                kwargs,
                (
                    "return_attention_scores",
                    "return_extractor_input"
                )
            ),
        ),
        name="full"
    )


def rav_trans(loss_mode=create_uniform_num_pos_arth_mask,
              loss_weight=2.0,
              number_loss=False,
              dry_run="auto",
              plw=None,
              **kwargs):
    if isinstance(loss_weight, float):
        loss_weight = (loss_weight, 1.0)

    model = GridTransformer(**kwargs)
    predict_model = PredictModel()

    def apply(x):
        x = model(x)
        predict = predict_model(x[OUTPUT][:, -1])
        if loss_weight != 0:
            loss = RavenLoss(mode=loss_mode, classification=True, number_loss=number_loss, lw=loss_weight, plw=plw)(
                {**x, "predict": predict[0]})

        return {**x, "predict": predict[0], "predict_mask": predict[1]}

    return apply


def rav_select_model(
        data,
        load_weights=None,
        loss_weight=(0.01, 0.0),
        plw=5.0,
        load_metric="acc",
        select_type=2,
        select_out=0,
        additional_out=0,
        additional_reg="auto",
        additional_copy=True,
        loss_fn=CROSS_ENTROPY,
        metric_fn=ACC,
        predictor=(1000, 1000),
        predictor_pre=None,
        batch_metrics=False,
        predictor_post=None,
        additional_output=None,
        metrics_all=False,
        change_last_act=False,
        contrastive_type="prob",
        contrastive_loss_fn=None,
        type_="direct",
        **kwargs
):
    out_layers = Last()
    if additional_out > 0:
        model3 = get_rav_trans(
            # TakeDict(data[0])[:, :8],
            data,
            plw=plw,
            loss_weight=loss_weight,
            **kwargs
        )

        if load_weights:
            model_load_weights(
                model3,
                load_weights,
                # sample_data,
                None,
                key=load_metric,
            )

        if AUGMENTATION in kwargs and kwargs[AUGMENTATION] is not None:
            index = -1
        else:
            index = -2

        if "out_pre" in kwargs and len(lw(kwargs['out_pre'])) > 0:
            additional_out *= (len(lw(kwargs['out_pre'])) + 1)

        # do not know it this one work propertly
        if "out_post" in kwargs and len(lw(kwargs['out_post'])) > 0:
            additional_out *= (len(lw(kwargs['out_post'])) + 1)

        out = model3[0, index, :additional_out]
        logger.info(f"Additional out from: {model3[0, index]}.")

        if additional_reg == "auto":
            if additional_out > 2:
                out += [Activation("gelu")]
        elif isinstance(additional_reg, int):
            out += [Dense(additional_reg)]
        elif additional_reg:
            out += [Regularization(additional_reg)]
        # high to have access to 3D Normalization layer the out need to be a end.

        if change_last_act is not False:
            for o in out[::-1]:
                if hasattr(o, "activation"):
                    o.activation = change_last_act
                    break

        out_layers = bm(out + [out_layers], add_flatten=False)
    model = get_rav_trans(
        # TakeDict(data[0])[:, :8],
        data,
        plw=plw,
        loss_weight=loss_weight,
        **{
            **kwargs,
            "out_layers": out_layers,
            "return_extractor_input": True,
            "mask": "last",
            # "pre": NO_ANSWER
            # "pre": INDEX

        },
        # **{**as_dict(p.mp), "show_shape": True, "save_shape": f"output/shapes/type_{p.mp.type_}.json"},
    )
    # from data_utils.ops import Equal
    # o = []
    # for i in range(1, 3):
    #     for j in range(2):
    #         o.append(
    #             Equal(
    #                 # model[0,:,-2, i].variables[j],
    #                 model2[0, :, -2, i].variables[j],
    #                 # out_layers[i].variables[j]
    #                 second_pooling[i].variables[j]
    #             ).equal
    #         )
    # assert all(o)
    # model = get_rav_trans(
    #     # TakeDict(val_generator[0])[:, 8:],
    #     # TakeDict(val_generator[0])[:, 8:],
    #     val_generator[0],
    #     plw=p.plw,
    #     loss_weight=p.loss_weight,
    #     **{**as_dict(p.mp),
    #        # "out_layers": out_layers,
    #        }
    #     # **{**as_dict(p.mp), "show_shape": True, "save_shape": f"output/shapes/type_{p.mp.type_}.json"},
    # )
    if load_weights:
        model_load_weights(model,
                           load_weights,
                           # sample_data,
                           None,
                           key=load_metric,
                           )
    # model.compile()
    # model.evaluate(val_generator.data[:1000])
    # model(TakeDict(val_generator[0])[:, 8:])
    trans_raven = model[0]
    # s = trans_raven(TakeDict(val_generator[0])[:, 8:])
    if select_type == 2:
        second_pooling = Lambda(lambda x: x[:, :-1])
    else:
        second_pooling = Last()
    if additional_out > 0:
        if additional_copy:
            model4 = get_rav_trans(
                # TakeDict(data[0])[:, :8],
                data,
                plw=plw,
                loss_weight=loss_weight,
                **kwargs
            )
            if load_weights:
                model_load_weights(model4,
                                   load_weights,
                                   # sample_data,
                                   None,
                                   key=load_metric,
                                   )

            if AUGMENTATION in kwargs and kwargs[AUGMENTATION] is not None:
                index = -1
            else:
                index = -2

            # if "out_pre" in kwargs and len(lw(kwargs['out_pre'])) > 0:
            #     additional_out *= (len(lw(kwargs['out_pre'])) + 1)
            #
            # # do not know it this one work propertly
            # if "out_post" in kwargs and len(lw(kwargs['out_post'])) > 0:
            #     additional_out *= (len(lw(kwargs['out_post'])) + 1)

            out2 = model4[0, index, :additional_out]
            logger.info(f"Additional out from: {model4[0, index]}.")

            if change_last_act is not False:
                for o in out2[::-1]:
                    if hasattr(o, "activation"):
                        o.activation = change_last_act
                        break

            if additional_reg == "auto":
                if additional_out > len(model4[0, index].layers):
                    out2 += [Activation("gelu")]
            elif additional_reg:
                out2 += [Regularization(additional_reg)]
        else:
            out2 = out

        # high to have access to 3D Normalization layer the out need to be a end.
        second_pooling = bm(out2 + [second_pooling], add_flatten=False)

    model2 = get_rav_trans(
        # TakeDict(data[0])[:, 8:] if select_type != 9 else TakeDict(filter_keys(data[0], 'inputs'))[:, 8:],
        # TakeDict(data[0])[:, :8],
        data,
        plw=plw,
        loss_weight=loss_weight,
        **{
            **kwargs,
            "out_layers": second_pooling,
            "mask": "no",
        }
        # **{**as_dict(p.mp), "show_shape": True, "save_shape": f"output/shapes/type_{p.mp.type_}.json"},
    )
    if load_weights:
        model_load_weights(
            model2,
            load_weights,
            # sample_data,
            None,
            key=load_metric,
        )
    if select_type == 0:
        # not working
        trans_raven2 = model2[0]
    else:
        trans_raven2 = model2[0]
    if select_out in [SEPARATE, FULL_JOIN]:
        predictor = Predictor(
            model=predictor,
            pre=predictor_pre,
            post=predictor_post,
            output_size=8 if select_out == FULL_JOIN else 1
        )
        logger.info(f"Predictor: {predictor}")
    # trans_raven2.mask_fn = ImageMask(last=take_by_index)
    if select_type == 2:
        select_model_class = ChoiceMaker2
    elif select_type == 1:
        select_model_class = ChoiceMaker1
    else:
        select_model_class = ChoiceMaker
    select_model = select_model_class(trans_raven, model2=trans_raven2, predictor=predictor, select_out=select_out)

    if contrastive_type == LATENT:
        contrastive_type = ContrastiveMetric
    elif contrastive_type == PROB:
        contrastive_type = ProbDistMetric

    if select_out == DIRECT:
        if metrics_all:
            loss_fn = [
                          # DCPMetric(),
                          DCPMetric(mask=PREDICT, batch_metrics=batch_metrics),

                          # ProbDistMetric(
                          #     loss_type="kl",
                          #     mode=create_all_mask,
                          #     batch_metrics=batch_metrics,
                          #     enable_metrics="dist_prob_kl"
                          # ),
                          # ProbDistMetric(
                          #     loss_type="kl_2",
                          #     mode=create_all_mask,
                          #     batch_metrics=batch_metrics,
                          #     enable_metrics="dist_prob_kl_2"
                          # ),
                          # ProbDistMetric(
                          #     mode=create_all_mask,
                          #     batch_metrics=batch_metrics,
                          # ),
                          #
                          # ProbDistMetric(
                          #     loss_type="kl",
                          #     mode=create_all_mask,
                          #     batch_metrics=batch_metrics,
                          #     enable_metrics="target_dist_prob_kl",
                          #     mask=TARGET
                          # ),
                          # ProbDistMetric(
                          #     loss_type="kl_2",
                          #     mode=create_all_mask,
                          #     batch_metrics=batch_metrics,
                          #     enable_metrics="target_dist_prob_kl_2",
                          #     mask=TARGET
                          # ),
                      ] + [
                          ProbDistMetric(
                              loss_type=lt,
                              mode=create_all_mask,
                              batch_metrics=batch_metrics,
                              enable_metrics=f"{m}_dist_prob_{lt}{'_sym' if ls else ''}",
                              mask=m,
                              plw=plw,
                              loss_sym=ls,
                          )
                          for lt in ["kl", "kl_2", "ce"] for m in [PREDICT, TARGET] for ls in [False, True]
                      ] + [
                          ProbDistMetric(
                              loss_type=lt,
                              mode=create_all_mask,
                              batch_metrics=batch_metrics,
                              enable_metrics=f"{m}_dist_prob_{lt}",
                              mask=m,
                              plw=plw,
                              loss_sym=False,
                          )
                          for lt in ["js", "js_2"] for m in [PREDICT, TARGET]

                      ]

        else:
            loss_fn = [
                contrastive_type(
                    # ProbDistMetric(
                    mode=create_all_mask,
                    enable_metrics="target_dist_prob",
                    plw=plw,
                    batch_metrics=batch_metrics,
                    mask=TARGET,
                    # loss_fn=reverse_index_loss,
                    loss_fn=contrastive_loss_fn,
                ),
            ]
        if additional_out > 0 and additional_out >= len(model3[0, index].layers) and not isinstance(additional_reg,
                                                                                                    int):
            predict_fn = DictModel(BatchModel(PredictModel()), OUTPUT, [PREDICT, PREDICT_MASK])
            loss_fn += [
                # DCPMetric(),
                DCPMetric(batch_metrics=batch_metrics),

            ]
        else:
            predict_fn = PREDICT
        trainer_kwargs = {
            "loss_wrap": False,
            "add_loss": False,
            "output": [
                PREDICT,
                PREDICT_MASK,
                'target_dist_prob_output',
                'target_dist_prob_metric'
            ]
        }

    else:

        loss_fn = (
            DictModel(
                (
                    Lambda(lambda x: (K.int(x[0] - 8), x[1])),
                    SparseCategoricalCrossentropy(from_logits=True),
                ),
                (INDEX, OUTPUT),
                LOSS
            ),

            DictModel(
                (
                    Lambda(lambda x: (K.int(x[0] - 8), x[1])),
                    SparseCategoricalAccuracy(),
                ),
                (INDEX, OUTPUT),
                LOSS
            ),
        )
        predict_fn = PREDICT
        trainer_kwargs = {
            OUTPUT: [
                PREDICT,
                CLASS_IMAGE,
            ]
        }
        # trainer_kwargs = {}

    # if loss_fn == CROSS_ENTROPY:
    # else:
    #     loss_fn = loss_fn
    # if metric_fn == ACC:
    # else:
    #     metric_fn = metric_fn
    if additional_output:
        if "output" in trainer_kwargs:
            trainer_kwargs["output"] = lw(trainer_kwargs['output']) + lw(additional_output)
        else:
            trainer_kwargs["output"] = lw(additional_output)
    select_model = Trainer(select_model, loss=loss_fn, predict=predict_fn, model_wrap=False, **trainer_kwargs)
    return select_model
