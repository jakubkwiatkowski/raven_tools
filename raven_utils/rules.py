from ml_utils import dict_from_list
COMBINE = "Number/Position"

ATTRIBUTES = [
    "Number",
    "Position",
    "Color",
    "Size",
    "Type"
]
ATTRIBUTES_LEN = len(ATTRIBUTES)
ATTRIBUTES_INDEX = dict_from_list(ATTRIBUTES)

TYPES = [
    "Constant",
    "Arithmetic",
    "Progression",
    "Distribute_Three"
]
TYPES_INDEX = dict_from_list(TYPES)
TYPES_LEN = len(TYPES)
