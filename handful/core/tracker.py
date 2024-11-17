from typing import List, Optional, Tuple
import cv2
import mediapipe as mp
import numpy as np

from handful.core.types import Color, HandLandmarks


class HandTracker:
    """Main class for hand tracking and gesture recognition."""

    def __init__(
        self,
        static_image_mode: bool = False,
        max_num_hands: int = 2,
        min_detection_confidence: float = 0.85,
        min_tracking_confidence: float = 0.5,
        draw_color: Color = Color.WHITE,
        draw_thickness: int = 2,
        draw_circle_radius: int = 2
    ):
        """Initialize the hand tracker with customizable parameters.
        :param static_image_mode: Whether to treat input as static images (vs video)
        :param max_num_hands: Maximum number of hands to detect
        :param min_detection_confidence: Minimum confidence for hand detection
        :param min_tracking_confidence: Minimum confidence for hand tracking
        :param draw_color: Color to use for landmark visualization
        :param draw_thickness: Thickness of drawn landmarks and connections
        :param draw_circle_radius: Radius of landmark circles when drawing
        """
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils

        self.hands = self.mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

        self.draw_specs = self.mp_draw.DrawingSpec(
            color=draw_color.value,
            thickness=draw_thickness,
            circle_radius=draw_circle_radius
        )

    def _process_landmarks(
        self,
        hand_landmarks,
        image_shape: Tuple[int, int, int]
    ) -> HandLandmarks:
        """Process detected landmarks to determine finger positions.
        :param hand_landmarks: MediaPipe hand landmarks
        :param image_shape: Shape of the input image (height, width, channels)
        :return HandLandmarks object containing processed data
        """
        height, width, _ = image_shape
        x_list = []
        y_list = []
        landmark_points = []

        # Convert normalized coordinates to pixel coordinates
        for landmark in hand_landmarks.landmark:
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            x_list.append(x)
            y_list.append(y)
            landmark_points.append((x, y))

        # Determine which fingers are up
        fingers_up = [
            x_list[4] <= x_list[5],  # thumb
            y_list[8] <= y_list[6],  # index
            y_list[12] <= y_list[10],  # middle
            y_list[16] <= y_list[14],  # ring
            y_list[20] < y_list[18]  # pinky
        ]

        return HandLandmarks(
            fingers_up=fingers_up,
            num_fingers_up=sum(fingers_up),
            landmark_points=landmark_points
        )

    def process_frame(
        self,
        frame: np.ndarray,
        draw_landmarks: bool = True,
        flip_horizontal: bool = True
    ) -> Tuple[np.ndarray, Optional[List[HandLandmarks]]]:
        """Process a single frame and detect hands.
        :param frame: Input frame (BGR format)
        :param draw_landmarks: Whether to draw landmarks on the output frame
        :param flip_horizontal: Whether to flip the frame horizontally
        :return tuple containing:
            - Processed frame with optional landmark visualization
            - List of HandLandmarks objects (None if no hands detected)
        """
        if flip_horizontal:
            frame = cv2.flip(frame, 1)

        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)

        output_frame = frame.copy()
        hand_data = []

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                if draw_landmarks:
                    self.mp_draw.draw_landmarks(
                        output_frame,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.draw_specs,
                        self.draw_specs
                    )

                hand_data.append(self._process_landmarks(hand_landmarks, frame.shape))

            return output_frame, hand_data

        return output_frame, None

    def create_debug_visualization(
        self,
        frame: np.ndarray,
        hand_data: List[HandLandmarks],
        show_finger_count: bool = True
    ) -> np.ndarray:
        """Create a debug visualization of the hand tracking results.
        :param frame: Input frame
        :param hand_data: List of HandLandmarks objects
        :param show_finger_count: Whether to show finger count on the frame
        :return Frame with debug visualization
        """
        debug_frame = frame.copy()

        if show_finger_count and hand_data:
            # Draw a blue rectangle with finger count
            cv2.rectangle(debug_frame, (25, 130), (100, 200), Color.BLUE.value, cv2.FILLED)
            cv2.putText(
                debug_frame,
                str(hand_data[0].num_fingers_up),
                (50, 180),
                cv2.FONT_HERSHEY_PLAIN,
                3,
                Color.WHITE.value,
                3
            )

        return debug_frame
