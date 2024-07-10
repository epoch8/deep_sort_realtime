import json

from deep_sort_realtime.deepsort_tracker import DeepSort


class DeepSortController:
    def __init__(self, deepsort_kwargs, round_big_arrays_to=32):
        self.deepsort_kwargs = deepsort_kwargs
        self.round_big_arrays_to = round_big_arrays_to

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
        with open(state_path, 'r') as r:
            tracker_data = json.load(r)
        return DeepSort.from_json(tracker_data)

    def _save_tracker_data(self, state_path, tracker):
        tracker_data = tracker.to_json(round_big_arrays_to=self.round_big_arrays_to)
        with open(state_path, 'w+') as w:
            w.write(json.dumps(tracker_data))
