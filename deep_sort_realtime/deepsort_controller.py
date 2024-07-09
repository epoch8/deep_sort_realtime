import json

import redis

from deep_sort_realtime.deepsort_tracker import DeepSort


class DeepSortController:
    def __init__(self, deepsort_kwargs, redis_kwargs):
        self.r = redis.Redis(**redis_kwargs)
        self.deepsort_kwargs = deepsort_kwargs

    def update(self, camera_id, detections, embeddings, annotation=None):
        # create tracker by camera id
        tracker = self._create_tracker(camera_id)
        # check if annotation is new
        pass
        # if annotation is new - recreate tracker, pass annotation to it
        pass
        # pass detections and embeddings to tracker
        tracks = tracker.update_tracks(detections, embeddings, anchor=False)
        # save tracker data to redis
        self._save_tracker_data(camera_id, tracker)
        # return tracks
        return tracks

    def _create_tracker(self, camera_id):
        tracker_data = self.r.get(camera_id)
        return DeepSort.from_json(json.loads(tracker_data)) if tracker_data else DeepSort(**self.deepsort_kwargs)

    def _save_tracker_data(self, camera_id, tracker):
        tracker_data = tracker.to_json()
        self.r.set(camera_id, json.dumps(tracker_data))
