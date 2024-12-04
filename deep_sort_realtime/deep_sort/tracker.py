# vim: expandtab:ts=4:sw=4
from __future__ import absolute_import

from copy import deepcopy
from datetime import datetime
import numpy as np
from . import kalman_filter
from . import linear_assignment
from . import iou_matching
from .track import Track
from .nn_matching import NearestNeighborDistanceMetric


class Tracker:
    """
    This is the multi-target tracker.

    Parameters
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        A distance metric for measurement-to-track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.
    today: Optional[datetime.date]
            Provide today's date, for naming of tracks

    Attributes
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        The distance metric used for measurement to track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of frames that a track remains in initialization phase.
    kf : kalman_filter.KalmanFilter
        A Kalman filter to filter target trajectories in image space.
    tracks : List[Track]
        The list of active tracks at the current time step.
    gating_only_position : Optional[bool]
        Used during gating, comparing KF predicted and measured states. If True, only the x, y position of the state distribution is considered during gating. Defaults to False, where x,y, aspect ratio and height will be considered.
    """

    def __init__(
        self,
        metric,
        max_iou_distance=0.7,
        max_age=30,
        n_init=3,
        override_track_class=None,
        today=None,
        gating_only_position=False,
        restore_removed_anchor_tracks=False,
    ):
        self.today = today
        self.metric = metric
        self.max_iou_distance = max_iou_distance
        self.max_age = max_age
        self.n_init = n_init
        self.gating_only_position = gating_only_position
        self.restore_removed_anchor_tracks = restore_removed_anchor_tracks

        self.kf = kalman_filter.KalmanFilter()
        self.tracks = []
        self.anchor_track_ids = set()
        self.removed_anchor_tracks = []
        self.removed_tracks = []
        self.del_tracks_ids = []
        self._next_id = 1
        if override_track_class:
            self.track_class = override_track_class
        else:
            self.track_class = Track

    def predict(self):
        """Propagate track state distributions one time step forward.

        This function should be called once every time step, before `update`.
        """
        for track in self.tracks:
            track.predict(self.kf)

    def update(self, detections, today=None, anchor=False):
        """Perform measurement update and track management.

        Parameters
        ----------
        detections : List[deep_sort.detection.Detection]
            A list of detections at the current time step.
        today: Optional[datetime.date]
            Provide today's date, for naming of tracks
        """
        if self.today:
            if today is None:
                today = datetime.now().date()
            # Check if its a new day, then refresh idx
            if today != self.today:
                self.today = today
                self._next_id = 1

        # when we update existing tracks - we don't need classes
        detections_maybe_without_classes = deepcopy(detections)
        for det in detections_maybe_without_classes:
            if not anchor:
                det.class_name = None

        # Run matching cascade.
        matches, unmatched_tracks, unmatched_detections = self._match(detections_maybe_without_classes)

        # Update track set.
        for track_idx, detection_idx, score in matches:
            if track_idx < len(self.tracks):  # match is in current track
                self.tracks[track_idx].update(self.kf, detections_maybe_without_classes[detection_idx], score)
            else:  # match is in removed anchor track
                removed_track_idx = track_idx - len(self.tracks)
                self.removed_anchor_tracks[removed_track_idx].update(
                    self.kf, detections_maybe_without_classes[detection_idx], score
                )
                self.removed_anchor_tracks[removed_track_idx].mark_confirmed()
                # print(f'Restore removed anchor {self.removed_anchor_tracks[removed_track_idx].track_id}')
        for track_idx in unmatched_tracks:
            if track_idx < len(self.tracks):  # current track is unmatched - remove it
                self.tracks[track_idx].mark_missed()
            else:  # if removed track is unmatched - we don't need to remove it again
                pass
        for detection_idx in unmatched_detections:
            # for new tracks - give them classes from search space predictions
            self._initiate_track(detections[detection_idx])
        new_tracks = []
        self.del_tracks_ids = []
        # because of pickling, we don't have this attribute yet
        if not hasattr(self, 'removed_tracks'):
            self.removed_tracks = []
        for t in self.tracks:
            if not t.is_deleted():
                new_tracks.append(t)
            else:
                self.del_tracks_ids.append(t.track_id)
                if t.track_id in self.anchor_track_ids:
                    self.removed_anchor_tracks.append(t)
                else:
                    self.removed_tracks.append(t)
        # add those removed anchor tracks which are confirmed to new tracks
        new_tracks.extend(filter(lambda track: track.is_confirmed(), self.removed_anchor_tracks))
        # remove confirmed tracks from removed_anchor_tracks
        self.removed_anchor_tracks = list(filter(lambda track: not track.is_confirmed(), self.removed_anchor_tracks))

        self.tracks = new_tracks
        if self.restore_removed_anchor_tracks and anchor:
            # if self.restore_removed_anchor_tracks is False - no anchor track ids will be saved ->
            # -> no anchor tracks will be restored
            self.anchor_track_ids.update(track.track_id for track in self.tracks)
            self.metric.set_anchor_track_ids(self.anchor_track_ids)

        # Update distance metric.
        features, targets = [], []
        for track in self.tracks:
            if not track.is_confirmed():
                continue
            features.append(track.features[-1])
            targets.append(track.track_id)
            track.features = [track.features[-1]]
        active_targets = [t.track_id for t in self.tracks + self.removed_anchor_tracks]
        self.metric.partial_fit(
            np.asarray(features), np.asarray(targets), active_targets
        )

    def _match(self, detections):
        def gated_metric(tracks, dets, track_indices, detection_indices):
            features = np.array([dets[i].feature for i in detection_indices])
            targets = np.array([tracks[i].track_id for i in track_indices])
            cost_matrix = self.metric.distance(features, targets)
            cost_matrix = linear_assignment.gate_cost_matrix(
                self.kf, cost_matrix, tracks, dets, track_indices, detection_indices, only_position=self.gating_only_position
            )

            return cost_matrix

        tracks_to_match = self.tracks + self.removed_anchor_tracks

        # first we match all tracks and all detections by IoU
        (
            matches_iou,
            unmatched_tracks_iou,  # those who didn't match by IoU
            unmatched_detections_iou,  # those who didn't match by IoU
        ) = linear_assignment.min_cost_matching(
            iou_matching.iou_cost,
            self.max_iou_distance,
            # match by iou not only from current tracks but also from removed anchor tracks
            tracks_to_match,
            detections
        )
        iou_match_to_iou = {(track_idx, det_idx): iou for track_idx, det_idx, iou in matches_iou}

        # then we leave only those track matches that are recent
        iou_emb_track_candidates = [
            k for k, _, _ in matches_iou if tracks_to_match[k].time_since_update == 1
        ]
        unmatched_tracks_iou_time = [
            k for k, _, _ in matches_iou if tracks_to_match[k].time_since_update != 1
        ]
        iou_emb_detection_candidates = [k for _, k, _ in matches_iou]
        # then we match by embeddings
        (
            matches_iou_emb,
            unmatched_tracks_emb,
            unmatched_detections_emb,
        ) = linear_assignment.matching_cascade(
            gated_metric,
            self.metric.matching_threshold,
            self.max_age,
            # match by embeddings not only from current tracks but also from removed anchor tracks
            tracks_to_match,
            detections,
            iou_emb_track_candidates,
            iou_emb_detection_candidates
        )

        def score_agg_function(score_iou, score_emb, eps=1e-8):
            # score_iou_first_n = int((1 - score_iou - eps) * 100)
            # score_emb_first_n = int((1 - score_emb - eps) * 100)
            # return float(f"0.{score_iou_first_n:02d}{score_emb_first_n:02d}")
            return (1 - score_iou) * (1 - score_emb)

        matches_iou_emb_with_agg_score = [
            (track_idx, det_idx, score_agg_function(iou_match_to_iou.get((track_idx, det_idx), 0.), emb_score))
            for track_idx, det_idx, emb_score in matches_iou_emb
        ]

        # we need to wrap in lists because they might be empty
        unmatched_tracks = list(set(list(unmatched_tracks_iou) + list(unmatched_tracks_iou_time) + list(unmatched_tracks_emb)))
        unmatched_detections = list(set(list(unmatched_detections_iou) + list(unmatched_detections_emb)))
        return matches_iou_emb_with_agg_score, unmatched_tracks, unmatched_detections

    def _initiate_track(self, detection):
        mean, covariance = self.kf.initiate(detection.to_xyah())

        if self.today:
            track_id = "{}_{}".format(self.today, self._next_id)
        else:
            track_id = "{}".format(self._next_id)
        self.tracks.append(
            self.track_class(
                mean,
                covariance,
                track_id,
                self.n_init,
                self.max_age,
                # mean, covariance, self._next_id, self.n_init, self.max_age,
                feature=detection.feature,
                original_ltwh=detection.get_ltwh(),
                det_class=detection.class_name,
                det_conf=detection.confidence,
                bbox_id=detection.bbox_id,
                instance_mask=detection.instance_mask,
                others=detection.others,
            )
        )
        self._next_id += 1

    def delete_all_tracks(self):
        self.tracks = []
        self._next_id = 1

    def to_json(self, round_big_arrays_to=32):
        tracks_list = [
            track.to_json(round_big_arrays_to=round_big_arrays_to)
            for track in self.tracks
        ]
        removed_anchor_tracks_list = [
            track.to_json(round_big_arrays_to=round_big_arrays_to)
            for track in self.removed_anchor_tracks
        ]
        anchor_track_ids_list = list(self.anchor_track_ids)
        metric_dict = self.metric.to_json(round_big_arrays_to=round_big_arrays_to)
        return {
            'tracks': tracks_list,
            'removed_anchor_tracks': removed_anchor_tracks_list,
            'anchor_track_ids': anchor_track_ids_list,
            '_next_id': self._next_id,
            'metric': metric_dict,
            'init_kwargs': {
                'max_iou_distance': self.max_iou_distance,
                'max_age': self.max_age,
                'n_init': self.n_init,
                'gating_only_position': self.gating_only_position,
                'restore_removed_anchor_tracks': self.restore_removed_anchor_tracks
            }
        }

    @staticmethod
    def from_json(data):
        metric = NearestNeighborDistanceMetric.from_json(data['metric'])
        tracks = [Track.from_json(track_data) for track_data in data['tracks']]
        rem_anch_tracks = [Track.from_json(track_data) for track_data in data['removed_anchor_tracks']]
        tracker = Tracker(metric, **data['init_kwargs'])

        tracker.tracks = tracks
        tracker.removed_anchor_tracks = rem_anch_tracks
        tracker.anchor_track_ids = set(data['anchor_track_ids'])
        tracker._next_id = data['_next_id']
        return tracker
