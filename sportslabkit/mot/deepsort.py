from sportslabkit.logger import logger
from sportslabkit.matching import MotionVisualMatchingFunction
from sportslabkit.metrics import CosineCMM, IoUCMM
from sportslabkit.mot.base import MultiObjectTracker
from sportslabkit.types.tracklet import Tracklet
from typing import List, Tuple, Dict, Any # For type hinting


class DeepSORTTracker(MultiObjectTracker):
    """DeepSORT tracker from https://arxiv.org/abs/1703.07402"""

    hparam_search_space = {
        "max_staleness": {"type": "int", "low": 1, "high": 1e3},
        "min_length": {"type": "int", "low": 1, "high": 1e3},
    }
    def __init__(
        self,
        detection_model=None,
        image_model=None,
        motion_model=None,
        # Accept separate matching functions
        matching_fn_ball: MotionVisualMatchingFunction = MotionVisualMatchingFunction(
            motion_metric=IoUCMM(),
            motion_metric_gate=0.2,
            visual_metric=CosineCMM(),
            visual_metric_gate=0.2,
            beta=0.5,
        ),
        matching_fn_players: MotionVisualMatchingFunction = MotionVisualMatchingFunction(
            motion_metric=IoUCMM(),
            motion_metric_gate=0.2,
            visual_metric=CosineCMM(),
            visual_metric_gate=0.2,
            beta=0.5,
        ),
        ball_class_id: int = 0, # Define the class ID for the ball
        window_size: int = 1,
        step_size: int | None = None,
        max_staleness: int = 5,
        min_length: int = 5,
        callbacks=None,
    ):
        super().__init__(
            window_size=window_size,
            step_size=step_size,
            max_staleness=max_staleness,
            min_length=min_length,
            callbacks=callbacks,
        )
        self.detection_model = detection_model
        self.image_model = image_model
        self.motion_model = motion_model
        # Store the separate matching functions
        self.matching_fn_ball = matching_fn_ball
        self.matching_fn_players = matching_fn_players
        self.ball_class_id = ball_class_id # Store the ball's class ID

    def update(self, current_frame, tracklets: List[Tracklet]) -> Tuple[List[Tracklet], List[Tracklet], List[Tracklet]]:
        """
        Updates the tracker state with a new frame, using separate matching
        for ball and players.
        """
        # 1. Detect objects using the detection model
        detections_raw = self.detection_model(current_frame)
        # Assuming detections_raw[0] holds the list of Detection objects
        # and converting to a standard list
        detections = detections_raw[0].to_list() if detections_raw and len(detections_raw) > 0 else []

        # 2. Update tracklets with motion model predictions
        for i, tracklet in enumerate(tracklets):
            predicted_box = self.motion_model(tracklet)
            tracklet.update_state("pred_box", predicted_box)

        # 3. Extract features for detections
        if len(detections) > 0:
            embeds = self.image_model.embed_detections(detections, current_frame)
            for i, det in enumerate(detections):
                det.feature = embeds[i]
        
        # 4. Split Detections and Tracklets by Class ID, preserving original indices
        
        # Detections
        detection_ball_with_orig_idx: List[Tuple[int, Any]] = []
        detection_players_with_orig_idx: List[Tuple[int, Any]] = []
        for i, det in enumerate(detections):
            if det.class_id == self.ball_class_id:
                detection_ball_with_orig_idx.append((i, det))
            else:
                detection_players_with_orig_idx.append((i, det))
        
        # Tracklets (using last known class_id)
        tracklets_ball_with_orig_idx: List[Tuple[int, Tracklet]] = []
        tracklets_players_with_orig_idx: List[Tuple[int, Tracklet]] = []
        for i, tracklet in enumerate(tracklets):
             # Get the class_id from the last observation or a reliable property
            # This assumes get_observation returns the last added observation dict
            last_class_id = tracklet.get_observation("class_id") 
            if last_class_id == self.ball_class_id:
                 tracklets_ball_with_orig_idx.append((i, tracklet))
            else:
                 tracklets_players_with_orig_idx.append((i, tracklet))

        # Prepare lists of objects for matching functions
        detections_ball = [item[1] for item in detection_ball_with_orig_idx]
        detections_players = [item[1] for item in detection_players_with_orig_idx]
        tracklets_ball = [item[1] for item in tracklets_ball_with_orig_idx]
        tracklets_players = [item[1] for item in tracklets_players_with_orig_idx]

        # 5. Perform Matching Separately
        matches_ball = []
        matches_players = []
        cost_matrix_ball = None
        cost_matrix_players = None

        if tracklets_ball and detections_ball:
            matches_ball, cost_matrix_ball = self.matching_fn_ball(
                tracklets_ball, detections_ball, return_cost_matrix=True
            )
            logger.debug(f"Ball matching: {len(matches_ball)} matches found.")

        if tracklets_players and detections_players:
             matches_players, cost_matrix_players = self.matching_fn_players(
                tracklets_players, detections_players, return_cost_matrix=True
            )
             logger.debug(f"Player matching: {len(matches_players)} matches found.")


        # 6. Process Results (Initialize lists and tracking sets)
        assigned_tracklets: List[Tracklet] = []
        new_tracklets: List[Tracklet] = []
        unassigned_tracklets: List[Tracklet] = []
        
        matched_orig_track_indices: set[int] = set()
        matched_orig_det_indices: set[int] = set()

        # 6a. Process Ball Matches
        for match in matches_ball:
            track_idx_ball, det_idx_ball = match[0], match[1]
            
            # Map back to original indices
            orig_track_idx = tracklets_ball_with_orig_idx[track_idx_ball][0]
            orig_det_idx = detection_ball_with_orig_idx[det_idx_ball][0]

            # Mark as matched
            matched_orig_track_indices.add(orig_track_idx)
            matched_orig_det_indices.add(orig_det_idx)

            # Get original objects
            tracklet = tracklets[orig_track_idx]
            detection = detections[orig_det_idx]

            # Log cost if available
            # cost = cost_matrix_ball[track_idx_ball, det_idx_ball] if cost_matrix_ball is not None else "N/A"
            # logger.debug(
            #     f"Ball Match: track_orig_idx={orig_track_idx}, det_orig_idx={orig_det_idx}, cost={cost}, staleness={tracklet.get_state('staleness')}"
            # )

            # Update assigned tracklet
            new_observation = {
                "box": detection.box,
                "score": detection.score,
                "frame": self.frame_count,
                "feature": detection.feature,
                "class_id": detection.class_id,
            }
            tracklet = self.update_tracklet(tracklet, new_observation)
            assigned_tracklets.append(tracklet)

        # 6b. Process Player Matches
        for match in matches_players:
            track_idx_players, det_idx_players = match[0], match[1]

            # Map back to original indices
            orig_track_idx = tracklets_players_with_orig_idx[track_idx_players][0]
            orig_det_idx = detection_players_with_orig_idx[det_idx_players][0]

            # Mark as matched
            matched_orig_track_indices.add(orig_track_idx)
            matched_orig_det_indices.add(orig_det_idx)

            # Get original objects
            tracklet = tracklets[orig_track_idx]
            detection = detections[orig_det_idx]

            # Log cost if available
            # cost = cost_matrix_players[track_idx_players, det_idx_players] if cost_matrix_players is not None else "N/A"
            # logger.debug(
            #      f"Player Match: track_orig_idx={orig_track_idx}, det_orig_idx={orig_det_idx}, cost={cost}, staleness={tracklet.get_state('staleness')}"
            # )

            # Update assigned tracklet
            new_observation = {
                "box": detection.box,
                "score": detection.score,
                "frame": self.frame_count,
                "feature": detection.feature,
                "class_id": detection.class_id,
            }
            tracklet = self.update_tracklet(tracklet, new_observation)
            assigned_tracklets.append(tracklet)


        # 7. Handle Unmatched Detections -> Create New Tracklets
        for i, det in enumerate(detections):
            if i not in matched_orig_det_indices:
                logger.debug(f"New tracklet created for detection index {i} (class {det.class_id})")
                new_observation = {
                    "box": det.box,
                    "score": det.score,
                    "frame": self.frame_count,
                    "feature": det.feature,
                    "class_id": det.class_id,
                }
                new_tracklet = self.create_tracklet(new_observation)
                new_tracklets.append(new_tracklet)

        # 8. Handle Unassigned Tracklets -> Update with Prediction
        for i, tracklet in enumerate(tracklets):
            if i not in matched_orig_track_indices:
                # Get required info from tracklet's history/state
                last_observation = tracklets[-1] # Need a method like this
                pred_box = tracklet.get_state("pred_box")

                if pred_box is not None and last_observation is not None:
                    logger.debug(f"Unassigned tracklet index {i} updated with prediction.")
                    new_observation = {
                        "box": pred_box, # Use predicted box
                        "score": last_observation.get_observation("score"), # Use previous score or default
                        "frame": self.frame_count,
                        "feature": last_observation.get_observation("feature"), # Use previous feature
                        "class_id": last_observation.get_observation("class_id"), # Use previous class_id
                    }
                    # Ensure all required keys are present even if using prediction
                    for key in self.required_observation_types:
                         if key not in new_observation:
                              new_observation[key] = None # Or appropriate default

                    tracklet = self.update_tracklet(tracklet, new_observation) # Pass is_match=False if your update_tracklet handles staleness differently
                    unassigned_tracklets.append(tracklet)
                else:
                     logger.warning(f"Could not update unassigned tracklet {i}: Missing predicted box or last observation.")
                     # Decide how to handle this - maybe add to unassigned without update or discard?
                     # Adding to unassigned without update to potentially prune later based on max_staleness
                     unassigned_tracklets.append(tracklet)


        # 9. Return results
        return assigned_tracklets, new_tracklets, unassigned_tracklets

    # --- (Properties remain the same) ---
    @property
    def required_observation_types(self):
        return ["box", "score", "feature", "frame", "class_id"]

    @property
    def required_state_types(self):
        motion_model_required_state_types = self.motion_model.required_state_types
        # Ensure 'pred_box' isn't duplicated if motion model also requires it
        required_state_types = list(set(motion_model_required_state_types + ["pred_box"]))
        return required_state_types