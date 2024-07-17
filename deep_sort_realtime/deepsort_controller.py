from deep_sort_realtime.deepsort_tracker import DeepSort
from deep_sort_realtime.utils.serialize import load_tracker, save_tracker


class DeepSortController:
    def __init__(self, deepsort_kwargs, round_big_arrays_to=32, serialize='json'):
        self.deepsort_kwargs = deepsort_kwargs
        self.round_big_arrays_to = round_big_arrays_to
        self.serialize = serialize

    def update(self, crops, load_state_path=None, save_state_path=None, create_new=False):
        tracker = self._create_tracker(load_state_path, create_new=create_new)
        detections, embeddings = self._format_input(crops)
        tracks = tracker.update_tracks(detections, embeddings, anchor=create_new)
        if save_state_path is not None:
            self._save_tracker_data(save_state_path, tracker)
        return self._format_output(tracks)

    def _create_tracker(self, state_path, create_new=False):
        if create_new:
            return DeepSort(**self.deepsort_kwargs)
        return load_tracker(state_path, self.serialize)

    def _save_tracker_data(self, state_path, tracker):
        save_tracker(state_path, tracker, self.serialize, self.round_big_arrays_to)

    def _format_input(self, crops):
        detections = []
        embeddings = []
        for crop in crops:
            detections.append([
                crop['coords'], crop.get('confidence', 1.), crop['product_id'], crop['bbox_id']
            ])
            embeddings.append(crop['embedding'])
        return detections, embeddings

    def _format_output(self, tracks):
        return [
            {'bbox_id': track.bbox_id, 'product_id': track.det_class, 'new': track.status == 'new'}
            for track in tracks
        ]
