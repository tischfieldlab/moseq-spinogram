

import json
from typing import Dict, List
from typing_extensions import Literal, TypedDict
from moseq2_viz.model.util import parse_model_results, relabel_by_usage


import numpy as np


class LabelMapping(TypedDict):
    raw: int
    usage: int
    frames: int

LabelMap = Dict[int, LabelMapping]
def get_syllable_id_mapping(model_file: str) -> LabelMap:
    ''' Gets a mapping of syllable IDs

    Parameters:
        model_file (str): path to a model to interrogate

    Returns:
        dict[int, LabelMapping]: A mapping of syllable IDs to their raw ID, usage ID, and frames ID, indexed by the raw ID.
    '''
    mdl = parse_model_results(model_file, sort_labels_by_usage=False)
    labels_usage = relabel_by_usage(mdl['labels'], count='usage')[1]
    labels_frames = relabel_by_usage(mdl['labels'], count='frames')[1]

    available_ids = list(set(labels_usage + labels_frames))
    label_map: LabelMap = {i: {'raw': i, 'usage': -1, 'frames': -1} for i in available_ids}
    label_map[-5] = {'raw': -5, 'usage': -5, 'frames': -5}  # -5 is the "unknown" label

    for usage_id, raw_id in enumerate(labels_usage):
        label_map[raw_id]['usage'] = usage_id

    for frames_id, raw_id in enumerate(labels_frames):
        label_map[raw_id]['frames'] = frames_id

    return label_map


def reindex_label_map(label_map: LabelMap, by: Literal['usage', 'frames', 'raw']) -> LabelMap:
    ''' Reindex a label map by usage, frames, or raw ID

    Parameters:
        label_map (LabelMap): The label map to reindex
        by (str): The key to reindex by, one of {'usage', 'frames', 'raw'}

    Returns:
        LabelMap: A new label map indexed by the specified key
    '''
    if by not in ['usage', 'frames', 'raw']:
        raise ValueError(f"Invalid index type '{by}'. Must be one of ['usage', 'frames', 'raw']")

    return {itm[by]: itm for itm in label_map.values()}


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