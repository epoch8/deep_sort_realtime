import json
import pickle

from deep_sort_realtime.deepsort_tracker import DeepSort


class DeepSortController:
    def __init__(self, deepsort_kwargs, round_big_arrays_to=32, serialize='json'):
        self.deepsort_kwargs = deepsort_kwargs
        self.round_big_arrays_to = round_big_arrays_to
        self.serialize = serialize

    def update(
        self,
        detections,
        embeddings,
        load_state_path=None,
        save_state_path=None,
        anchor=False
    ):
        tracker = self._create_tracker(load_state_path, create_new=anchor)
        tracks = tracker.update_tracks(detections, embeddings, anchor=anchor)
        if save_state_path is not None:
            self._save_tracker_data(save_state_path, tracker)
        return tracks

    def _create_tracker(self, state_path, create_new=False):
        if create_new:
            return DeepSort(**self.deepsort_kwargs)
        if self.serialize == 'json':
            with open(state_path, 'r') as r:
                tracker_data = json.load(r)
            return DeepSort.from_json(tracker_data)
        elif self.serialize == 'pickle':
            with open(state_path, 'rb') as r:
                return pickle.load(r)
        else:
            raise NotImplementedError()

    def _save_tracker_data(self, state_path, tracker):
        if self.serialize == 'json':
            tracker_data = tracker.to_json(round_big_arrays_to=self.round_big_arrays_to)
            with open(state_path, 'w+') as w:
                w.write(json.dumps(tracker_data))
        elif self.serialize == 'pickle':
            with open(state_path, 'wb') as w:
                pickle.dump(tracker, w)
        else:
            raise NotImplementedError()
