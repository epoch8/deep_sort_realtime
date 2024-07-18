import io
import json
import pickle

from deep_sort_realtime.deepsort_tracker import DeepSort


def load_tracker(path, serialization_type, s3_client=None, s3_bucket=None):
    if serialization_type == 'json':
        return read_json(path)
    elif serialization_type == 'pickle':
        return read_pickle(path)
    elif serialization_type == 'pickle_s3':
        return read_pickle_s3(s3_bucket, path, s3_client)
    else:
        raise NotImplementedError()


def save_tracker(path, tracker, serialization_type, round_big_arrays_to=32, s3_client=None, s3_bucket=None):
    if serialization_type == 'json':
        write_json(path, tracker, round_big_arrays_to)
    elif serialization_type == 'pickle':
        write_pickle(path, tracker)
    elif serialization_type == 'pickle_s3':
        write_pickle_s3(tracker, s3_bucket, path, s3_client)
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


def write_pickle_s3(tracker, bucket, path, s3_client):
    buf = io.BytesIO()
    pickle.dump(tracker, buf)
    buf.seek(0)
    s3_client.upload_fileobj(buf, bucket, path)


def read_pickle_s3(bucket, path, s3_client):
    buf = io.BytesIO()
    s3_client.download_fileobj(bucket, path, buf)
    buf.seek(0)
    return pickle.load(buf)
