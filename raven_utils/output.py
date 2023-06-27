import numpy as np
import raven_utils.entity as entity
import raven_utils.properties as properties
import raven_utils.group as group

SIZE = entity.SUM * properties.SUM + group.NO + entity.SUM

SLOT_AND_GROUP = group.NO + entity.SUM

PROPERTIES_SLICE = np.s_[:, :-SLOT_AND_GROUP]
SLOT_SLICE = np.s_[:, -SLOT_AND_GROUP:-group.NO]
GROUP_SLICE = np.s_[:, -group.NO:]

GROUP_SLICE_END = np.s_[-group.NO:]
SLOT_SLICE_END = np.s_[-SLOT_AND_GROUP:-group.NO]
PROPERTIES_SLICE_END = np.s_[:-SLOT_AND_GROUP]
