import redis

from deep_sort_realtime.deepsort_tracker import DeepSort


class DeepSortController:
    def __init__(self, deepsort_kwargs, redis_kwargs, round_big_arrays_to=32):
        self.r = redis.Redis(**redis_kwargs)
        self.deepsort_kwargs = deepsort_kwargs
        self.round_big_arrays_to = round_big_arrays_to

    def update(
        self,
        camera_id,
        detections,
        embeddings,
        annotation=None,
        update_storage=True,
        force_create_new=False,
        anchor=False  # this flag is just for debugging
    ):
        # check if annotation is new
        is_new_anno = self._is_new_annotation(annotation)
        if is_new_anno:
            # if annotation is new - recreate tracker, pass annotation to it
            tracker = self._create_tracker_from_redis(camera_id, create_new=True)
            tracker = self._pass_anno_to_tracker(annotation, tracker)
        else:
            # if annotation is not new - just load it from redis or create new if force_create_new
            tracker = self._create_tracker_from_redis(camera_id, create_new=force_create_new)
        # pass detections and embeddings to tracker
        tracks = tracker.update_tracks(detections, embeddings, anchor=anchor)
        # save tracker data to redis
        if update_storage:
            self._save_tracker_data_to_redis(camera_id, tracker)
        return tracks

    def update_new_tracks(self, camera_id, tracks):
        for track in tracks:
            if track.status != 'new':
                continue
            set_path = f'.tracker.tracks[?(@.init_kwargs.track_id == "{track.track_id}")].init_kwargs.det_class'
            self.r.json().set(camera_id, set_path, track.det_class)

    def _create_tracker_from_redis(self, camera_id, create_new=False):
        if create_new:
            return DeepSort(**self.deepsort_kwargs)
        tracker_data = self.r.json().get(camera_id, '.')
        return DeepSort.from_json(tracker_data) if tracker_data else DeepSort(**self.deepsort_kwargs)

    def _save_tracker_data_to_redis(self, camera_id, tracker):
        tracker_data = tracker.to_json(round_big_arrays_to=self.round_big_arrays_to)
        self.r.json().set(camera_id, '.', tracker_data)

    def _is_new_annotation(self, annotation):
        return False

    def _pass_anno_to_tracker(self, annotation, tracker):
        return tracker
