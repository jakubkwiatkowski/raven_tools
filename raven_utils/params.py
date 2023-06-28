from dataclasses import dataclass, field
from typing import Tuple, Union, Callable

from grid_transformer import GridTransformerParameters
from core_tools.core import LAST
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
