import numpy as np
import raven_utils.group as group
import raven_utils.entity as entity
import raven_utils.rules as rules
import raven_utils.properties as properties


ENTITY_INDEX = entity.INDEX + 1
ENTITY_DICT = dict(zip(group.NAMES, ENTITY_INDEX[:-1]))
NAMES_ORDER = dict(zip(group.NAMES, np.arange(len(group.NAMES))))
PROPERTIES_INDEXES = np.cumsum(np.array(list(entity.NO.values())) * properties.NO)
INDEX = np.concatenate([[0], PROPERTIES_INDEXES]) + entity.SUM + 1  # +2 type and uniformity

SECOND_LAYOUT = [i - 1 for i in [
    ENTITY_DICT["in_center_single_out_center_single"] + 1,
    # due to different order for in_4_out_1
    ENTITY_DICT["in_distribute_four_out_center_single"],
    ENTITY_DICT["in_distribute_four_out_center_single"] + 1,
    ENTITY_DICT["in_distribute_four_out_center_single"] + 2,
    ENTITY_DICT["in_distribute_four_out_center_single"] + 3,
    ENTITY_DICT["left_center_single_right_center_single"] + 1,
    ENTITY_DICT["up_center_single_down_center_single"] + 1
]]

FIRST_LAYOUT = list(set(range(entity.SUM)) - set(SECOND_LAYOUT))
LAYOUT_NO = 2

START_INDEX = dict(zip(group.NAMES, INDEX[:-1]))
END_INDEX = INDEX[-1]

RULES_ATTRIBUTES_ALL_LEN = rules.ATTRIBUTES_LEN * LAYOUT_NO
UNIFORMITY_NO = 2
UNIFORMITY_INDEX = END_INDEX + RULES_ATTRIBUTES_ALL_LEN

SIZE = UNIFORMITY_INDEX + UNIFORMITY_NO

def take(target):
    return target[1], target[2]


def create(images, index, pattern_index=(2, 5), full_index=False, arrange=np.arange, shape=lambda x: x.shape):
    return [images[:, pattern_index[0]], images[:, pattern_index[1]],
            images[arrange(shape(index)[0]), (0 if full_index else 8) + index[:, 0]]]



def take_simple(target):
    return target[1], target[0]


def create_simple(images, target, index=slice(None), pattern_index=(2, 5)):
    return [images[:, pattern_index[0]], images[:, pattern_index[1]], target][index]