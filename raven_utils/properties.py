import raven_utils as rv
import numpy as np
from core_tools.core import dict_from_list2, CalcDict
import raven_utils.entity as entity

NAMES = [
    'Color',
    'Size',
    'Type',
]
RAW_SIZE = [10, 6, 5]
SIZE = dict_from_list2(NAMES, RAW_SIZE)
ANGLE_SIZE = 7
NO = len(NAMES)

INDEX = (CalcDict(SIZE) * entity.SUM).to_dict()
SUM = sum(list(SIZE.values()))

COUNT = CalcDict(entity.NO) * NO
COUNT_WITH_SLOT = CalcDict(entity.NO) * (NO + 1)
COUNT_ALL = COUNT_WITH_SLOT + 1

MAX_DIFF = np.array([2] * 26 + RAW_SIZE * 25) - 1
