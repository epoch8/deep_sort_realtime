import json
import pickle

from deep_sort_realtime.deepsort_tracker import DeepSort


def load_tracker(path, serialization_type):
    if serialization_type == 'json':
        return read_json(path)
    elif serialization_type == 'pickle':
        return read_pickle(path)
    else:
        raise NotImplementedError()


def save_tracker(path, tracker, serialization_type, round_big_arrays_to=32):
    if serialization_type == 'json':
        write_json(path, tracker, round_big_arrays_to)
    elif serialization_type == 'pickle':
        write_pickle(path, tracker)
    else:
        raise NotImplementedError()


def read_json(path):
    with open(path, 'r') as r:
        tracker_data = json.load(r)
    return DeepSort.from_json(tracker_data)


def write_json(path, tracker, round_big_arrays_to=32):
    with open(path, 'w+') as w:
        w.write(json.dumps(tracker.to_json(round_big_arrays_to=round_big_arrays_to)))


def read_pickle(path):
    with open(path, 'rb') as r:
        return pickle.load(r)


def write_pickle(path, tracker):
    with open(path, 'wb') as w:
        pickle.dump(tracker, w)
