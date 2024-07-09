import json

import redis

from deep_sort_realtime.deepsort_tracker import DeepSort


class DeepSortController:
    def __init__(self, deepsort_kwargs, redis_kwargs):
        self.r = redis.Redis(**redis_kwargs)
        self.deepsort_kwargs = deepsort_kwargs

    def update(self, camera_id, detections, embeddings, annotation=None):
        # check if annotation is new
        is_new_anno = self._is_new_annotation(annotation)
        if is_new_anno:
            # if annotation is new - recreate tracker, pass annotation to it
            tracker = self._create_tracker_from_redis(camera_id, create_new=True)
            tracker = self._pass_anno_to_tracker(annotation, tracker)
        else:
            # if annotation is not new - just load it from redis
            tracker = self._create_tracker_from_redis(camera_id, create_new=False)
        # pass detections and embeddings to tracker
        tracks = tracker.update_tracks(detections, embeddings, anchor=False)
        # save tracker data to redis
        self._save_tracker_data_to_redis(camera_id, tracker)
        return tracks

    def _create_tracker_from_redis(self, camera_id, create_new=False):
        if create_new:
            return DeepSort(**self.deepsort_kwargs)
        tracker_data = self.r.get(camera_id)
        return DeepSort.from_json(json.loads(tracker_data)) if tracker_data else DeepSort(**self.deepsort_kwargs)

    def _save_tracker_data_to_redis(self, camera_id, tracker):
        tracker_data = tracker.to_json()
        self.r.set(camera_id, json.dumps(tracker_data))

    def _is_new_annotation(self, annotation):
        return False

    def _pass_anno_to_tracker(self, annotation, tracker):
        return tracker
