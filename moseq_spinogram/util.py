

import json
from typing import Dict, List
from typing_extensions import TypedDict
from moseq2_viz.model.util import parse_model_results, relabel_by_usage


import numpy as np


LabelMap = TypedDict('LabelMap', {
    'raw': int,
    'usage': int,
    'frames': int,
})
def get_syllable_id_mapping(model_file: str) -> List[LabelMap]:
    ''' Gets a mapping of syllable IDs

    Parameters:
        model_file (str): path to a model to interrogate

    Returns:
        list of dicts, each dict contains raw, usage, and frame ID assignments
    '''
    mdl = parse_model_results(model_file, sort_labels_by_usage=False)
    labels_usage = relabel_by_usage(mdl['labels'], count='usage')[0]
    labels_frames = relabel_by_usage(mdl['labels'], count='frames')[0]

    label_map: Dict[int, LabelMap] = {}
    for si, sl in enumerate(mdl['labels']):
        for i, l in enumerate(sl):
            if l not in label_map:
                label_map[l] = {
                    'raw': l,
                    'usage': labels_usage[si][i],
                    'frames': labels_frames[si][i]
                }
    return list(label_map.values())


class NumpyEncoder(json.JSONEncoder):
    ''' Special json encoder for numpy types '''
    def default(self, obj): # pylint: disable=method-hidden
        np_int_types = (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32,
                    np.int64, np.uint8, np.uint16, np.uint32, np.uint64)
        np_flt_types = (np.float_, np.float16, np.float32, np.float64)
        if isinstance(obj, np_int_types):
            return int(obj)
        elif isinstance(obj, np_flt_types):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)): #### This is the fix
            return obj.tolist()
        else:
            return json.JSONEncoder.default(self, obj)
#end class NumpyEncoder