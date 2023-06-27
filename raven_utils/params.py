from dataclasses import dataclass, field
from typing import Tuple, Union, Callable

from ml_utils import get_str_name
from grid_transformer import GridTransformerParameters
from models_utils import LAST
from raven_utils import output

from grid_transformer.constants import RANDOM, TILE
from raven_utils.constant import NUM_POS_ARTH, DIRECT, SEPARATE, TASK, PANEL, ROW, IMAGES, RAV, DENSEFORMER, INPUT

LATENT = "latent"

PROB = "prob"




@dataclass
class RowTokenizerParameters(GridTransformerParameters):
    tokenizer_name: str = ROW
    last_index: int = 8
    col: int = 1
    row: int = 1
    output_size: int = output.SIZE
    number_loss: int = 2
    pre: str = IMAGES
    num_heads: int = 8
    mask: str = RANDOM
    pooling: str = 9
    out_pre:Tuple = ("LN",)


MP = RowTokenizerParameters


@dataclass
class PanelTokenizerParameters(RowTokenizerParameters):
    tokenizer_name: str = PANEL
    channel: int = TILE


@dataclass
class TaskTokenizerParameters(RowTokenizerParameters):
    tokenizer_name: str = TASK
    col: int = 3
    row: int = 3
    extractor_input: int = 252
    channel: int = TILE


MP = RowTokenizerParameters


@dataclass
class DenseFormerParameters(RowTokenizerParameters):
    pos_emd: str = DENSEFORMER
    dense_size: int = 336
    dense_no: int = 5
    dense_norm: float = 0.0
    interleave_before: str = None
    interleave_after: str = ("LN", "LD")
    _experiment: str = f"{RAV}/{DENSEFORMER}"


@dataclass
class DenseFormerParameters2(RowTokenizerParameters):
    pos_emd: str = DENSEFORMER
    dense_size: int = 512
    dense_no: int = 4
    dense_norm: float = 0.0
    interleave_before: str = None
    interleave_after: str = ("LN", "LD")
    _experiment: str = f"{RAV}/{DENSEFORMER}"


@dataclass
class DoubleRowTokenizerParameters(RowTokenizerParameters):
    number_loss: int = 0
    mask: str = LAST


DMP = DoubleRowTokenizerParameters


# @dataclass
# class RowTokenizerParameters(RowTokenizerParameters):
#     select_type: int = 9
#     select_out: int = 0
#     additional_out: int = 0
#     additional_copy: bool = True
#     predictor: Tuple = (1000, 1000)
#     # pre: str = "index"
#     additional_reg: str = None
#     train_only_predictor: bool = True
#     predictor_size: int = 1024
#     predictor_no: int = 5
#     predictor_norm: Tuple = (tuple(), ("LN", "LD"))


@dataclass
class DirectChoiceMakerParameters:
    select_type: int = 9
    predictor: int = None
    select_out: str = DIRECT
    additional_out: int = 3
    return_extractor_input: bool = True
    change_last_act: bool = False
    contrastive_type: str = PROB
    contrastive_loss: bool = False
    train_only_predictor: bool = False
    additional_copy: bool = False
    additional_reg: str = None


@dataclass
class LearnableChoiceMakerParameters(DirectChoiceMakerParameters):
    select_out: str = SEPARATE
    predictor_no: int = 5
    predictor_size: int = 1024
    predictor_norm: Tuple = (("LN",), ("LD",),)


@dataclass
class ContrastiveLearnableChoiceMakerParameters(DirectChoiceMakerParameters):
    contrastive_loss_fn: bool = "reverse_index"


@dataclass
class Contrastive2LearnableChoiceMakerParameters(DirectChoiceMakerParameters):
    contrastive_loss_fn: bool = "contrastive"


@dataclass
class LatentContrastiveLearnableChoiceMakerParameters(DirectChoiceMakerParameters):
    additional_out: int = 0
    contrastive_type: str = LATENT


@dataclass
class Latent2ContrastiveLearnableChoiceMakerParameters(LatentContrastiveLearnableChoiceMakerParameters):
    additional_out: int = 2
    contrastive_type: str = LATENT
    change_last_act: bool = None


@dataclass
class Latent3ContrastiveLearnableChoiceMakerParameters(DirectChoiceMakerParameters):
    contrastive_type: str = LATENT

@dataclass
class LatentProjectionContrastiveLearnableChoiceMakerParameters(DirectChoiceMakerParameters):
    contrastive_type: str = LATENT
    additional_reg: int = 16

SMP = RowTokenizerParameters

from experiment_utils.parameters.nn_clean import Parameters as BaseParameters
from raven_utils.config.constant import RAVEN, IMP_RAV_METRICS, ACC_SAME

MODEL_NO = -1


@dataclass
class RavenParameters(BaseParameters):
    dataset_name: str = RAVEN
    data: str = None
    dataset_split: str = ("train", "val")

    # core_metrics: tuple = tuple(RAV_METRICS)
    filter_metrics: tuple = tuple(IMP_RAV_METRICS)
    # result_metric: str = ACC_NO_GROUP
    result_metric: str = ACC_SAME
    loss_mode: Union[Callable, str] = NUM_POS_ARTH

    lw: float = 0.0001  # Autoencoder
    loss_weight: Union[Tuple, float] = 2.0
    plw: int = 5.0
    mp: RowTokenizerParameters = field(default_factory=lambda:RowTokenizerParameters())

    _experiment: str = "rav/best_test3"

    # @property
    # def experiment(self):
    #     # return "rav/trans"
    #     return "rav/best_test3"
    #     # return "rav/trans_weight"

    # @property
    # def name(self):
    #     # return f"i{self.extractor}_{len(self.tail)}{self.tail[0]}_{self.type_}_{self.epsilon}_{self.last}_{self.epsilon_step}"
    #     return f"{get_str_name(self.mp.pre)[0]}_{str(self.plw)[0]}_{str(self.mp.number_loss)[0]}_{self.mp.extractor}_{self.mp.noise if self.mp.noise else ''}_{self.mp.augmentation if self.mp.augmentation else ''}_{self.mp.extractor_shape}_{self.mp.no}_{self.mp.num_heads}_{self.mp.size}_{self.mp.pos_emd}_{self.mp.ff_mul}_{self.tp.batch}"


@dataclass
class DoubleRavenParameters(RavenParameters):
    loss_weight: Union[Tuple, float] = (0.01, 0.0),
    mp: RowTokenizerParameters = field(default_factory=lambda:DoubleRowTokenizerParameters())


@dataclass
class BaselineRavenParameters(RavenParameters):

    @property
    def experiment(self):
        # return "rav/best_test3"
        return "rav/baseline"
        # return "rav/trans_weight"

    @property
    def name(self):
        # return f"i{self.extractor}_{len(self.tail)}{self.tail[0]}_{self.type_}_{self.epsilon}_{self.last}_{self.epsilon_step}"
        return f"{get_str_name(self.mp.pre)[0]}_{str(self.plw)[0]}_{str(self.mp.number_loss)[0]}_{self.mp.extractor}_{self.mp.noise if self.mp.noise else ''}_{self.mp.augmentation if self.mp.augmentation else ''}_{self.mp.extractor_input}_{self.mp.no}_{self.mp.num_heads}_{self.mp.size}_{self.mp.pos_emd}_{self.mp.ff_mul}_{self.tp.batch}"


@dataclass
class RavenParametersEval(RavenParameters):
    dataset_split: int = ("train", "val")

    @property
    def experiment(self):
        return "rav/trans_eval"

    @property
    def name(self):
        return f"{self.dataset_split}|" + super(RavenParametersEval, self).name


if __name__ == '__main__':
    params = DoubleRowTokenizerParameters()
