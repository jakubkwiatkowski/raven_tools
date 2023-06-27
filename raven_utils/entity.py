import raven_utils.group as group
import numpy as np
NO = dict(zip(group.NAMES, [1, 4, 9, 2, 5, 2, 2]))
SUM = sum(list(NO.values()))

INDEX = np.concatenate([[0], np.cumsum(list(NO.values()))])
