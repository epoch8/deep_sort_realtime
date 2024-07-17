import logging
from collections.abc import Iterable

import cv2
import numpy as np

from deep_sort_realtime.deep_sort import nn_matching
from deep_sort_realtime.deep_sort.detection import Detection
from deep_sort_realtime.deep_sort.tracker import Tracker
from deep_sort_realtime.utils.nms import non_max_suppression

logger = logging.getLogger(__name__)


class DeepSort(object):
    def __init__(
        self,
        max_iou_distance=0.7,
        max_age=30,
        n_init=3,
        nms_max_overlap=1.0,
        max_cosine_distance=0.2,
        nn_budget=None,
        gating_only_position=False,
        override_track_class=None,
        today=None,
        restore_removed_anchor_tracks=False,
        add_anchor_feature_threshold=0.05,
        min_num_anchor_features=10,
        round_big_arrays_to=32
    ):
        """

        Parameters
        ----------
        max_iou_distance : Optional[float] = 0.7
            Gating threshold on IoU. Associations with cost larger than this value are
            disregarded. Argument for deep_sort_realtime.deep_sort.tracker.Tracker.
        max_age : Optional[int] = 30
            Maximum number of missed misses before a track is deleted. Argument for deep_sort_realtime.deep_sort.tracker.Tracker.
        n_init : int
            Number of frames that a track remains in initialization phase. Defaults to 3. Argument for deep_sort_realtime.deep_sort.tracker.Tracker.
        nms_max_overlap : Optional[float] = 1.0
            Non-maxima suppression threshold: Maximum detection overlap, if is 1.0, nms will be disabled
        max_cosine_distance : Optional[float] = 0.2
            Gating threshold for cosine distance
        nn_budget :  Optional[int] = None
            Maximum size of the appearance descriptors, if None, no budget is enforced
        gating_only_position : Optional[bool]
            Used during gating, comparing KF predicted and measured states. If True, only the x, y position of the state distribution is considered during gating. Defaults to False, where x,y, aspect ratio and height will be considered.
        override_track_class : Optional[object] = None
            Giving this will override default Track class, this must inherit Track. Argument for deep_sort_realtime.deep_sort.tracker.Tracker.
            Whether detections are polygons (e.g. oriented bounding boxes)
        today: Optional[datetime.date]
            Provide today's date, for naming of tracks. Argument for deep_sort_realtime.deep_sort.tracker.Tracker.
        """
        self.nms_max_overlap = nms_max_overlap
        metric = nn_matching.NearestNeighborDistanceMetric(
            "cosine",
            max_cosine_distance,
            nn_budget,
            add_anchor_feature_threshold,
            min_num_anchor_features,
            round_big_arrays_to
        )
        self.tracker = Tracker(
            metric,
            max_iou_distance=max_iou_distance,
            max_age=max_age,
            n_init=n_init,
            override_track_class=override_track_class,
            today=today,
            gating_only_position=gating_only_position,
            restore_removed_anchor_tracks=restore_removed_anchor_tracks
        )
        logger.info("DeepSort Tracker initialised")
        logger.info(f"- max age: {max_age}")
        logger.info(f"- appearance threshold: {max_cosine_distance}")
        logger.info(
            f'- nms threshold: {"OFF" if self.nms_max_overlap == 1.0 else self.nms_max_overlap}'
        )
        logger.info(f"- max num of appearance features: {nn_budget}")
        logger.info(
            f'- overriding track class : {"No" if override_track_class is None else "Yes"}'
        )
        logger.info(f'- today given : {"No" if today is None else "Yes"}')

    def update_tracks(
            self, raw_detections, embeds=None, frame=None, today=None, others=None, instance_masks=None, anchor=False
    ):

        """Run multi-target tracker on a particular sequence.

        Parameters
        ----------
        raw_detections (horizontal bb) : List[ Tuple[ List[float or int], float, str ] ]
            List of detections, each in tuples of ( [left,top,w,h] , confidence, detection_class)
        raw_detections (polygon) : List[ List[float], List[int or str], List[float] ]
            List of Polygons, Classes, Confidences. All 3 sublists of the same length. A polygon defined as a ndarray-like [x1,y1,x2,y2,...].
        embeds : Optional[ List[] ] = None
            List of appearance features corresponding to detections
        frame : Optional [ np.ndarray ] = None
            if embeds not given, Image frame must be given here, in [H,W,C].
        today: Optional[datetime.date]
            Provide today's date, for naming of tracks
        others: Optional[ List ] = None
            Other things associated to detections to be stored in tracks, usually, could be corresponding segmentation mask, other associated values, etc. Currently others is ignored with polygon is True.
        instance_masks: Optional [ List ] = None
            Instance masks corresponding to detections. If given, they are used to filter out background and only use foreground for apperance embedding. Expects numpy boolean mask matrix.

        Returns
        -------
        list of track objects (Look into track.py for more info or see "main" section below in this script to see simple example)

        """
        assert isinstance(raw_detections, Iterable)

        assert len(raw_detections[0][0]) == 4
        raw_detections = [d for d in raw_detections if d[0][2] > 0 and d[0][3] > 0]

        # Proper deep sort detection objects that consist of bbox, confidence and embedding.
        detections = self.create_detections(raw_detections, embeds, instance_masks=instance_masks, others=others)

        # Run non-maxima suppression.
        boxes = np.array([d.ltwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        if self.nms_max_overlap < 1.0:
            # nms_tic = time.perf_counter()
            indices = non_max_suppression(boxes, self.nms_max_overlap, scores)
            # nms_toc = time.perf_counter()
            # logger.debug(f'nms time: {nms_toc-nms_tic}s')
            detections = [detections[i] for i in indices]

        # Update tracker.
        self.tracker.predict()
        self.tracker.update(detections, today=today, anchor=anchor)

        return self.tracker.tracks

    def create_detections(self, raw_dets, embeds, instance_masks=None, others=None):
        detection_list = []
        for i, (raw_det, embed) in enumerate(zip(raw_dets, embeds)):
            detection_list.append(
                Detection(
                    raw_det[0],
                    raw_det[1],
                    embed,
                    class_name=raw_det[2] if len(raw_det) == 3 else None,
                    instance_mask=instance_masks[i] if isinstance(instance_masks, Iterable) else instance_masks,
                    others=others[i] if isinstance(others, Iterable) else others,
                )
            )  # raw_det = [bbox, conf_score, class]
        return detection_list

    @staticmethod
    def crop_bb(frame, raw_dets, instance_masks=None):
        crops = []
        im_height, im_width = frame.shape[:2]
        if instance_masks is not None:
            masks = []
        else:
            masks = None
        for i, detection in enumerate(raw_dets):
            l, t, w, h = [int(x) for x in detection[0]]
            r = l + w
            b = t + h
            crop_l = max(0, l)
            crop_r = min(im_width, r)
            crop_t = max(0, t)
            crop_b = min(im_height, b)
            crops.append(frame[crop_t:crop_b, crop_l:crop_r])
            if instance_masks is not None:
                masks.append(instance_masks[i][crop_t:crop_b, crop_l:crop_r])

        return crops, masks

    def delete_all_tracks(self):
        self.tracker.delete_all_tracks()

    def to_json(self, round_big_arrays_to=32):
        return {'tracker': self.tracker.to_json(round_big_arrays_to=round_big_arrays_to)}

    @staticmethod
    def from_json(data):
        inner_tracker = Tracker.from_json(data['tracker'])
        tracker = DeepSort()  # no init kwargs - it's ok
        tracker.tracker = inner_tracker
        return tracker
