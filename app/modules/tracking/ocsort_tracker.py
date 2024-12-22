from typing import List, Any

from boxmot import OcSort
from app.interface import ServiceInterface


class Tracking(ServiceInterface):
    """
    A tracking service that utilizes the OcSort algorithm.
    """

    def __init__(self, name: str = "tracking_service") -> None:
        """
        Initializes the Tracking service.
        """
        super().__init__(name=name)
        self.tracker = OcSort()

    def inference(self, detections: List[Any], frame: Any) -> List[Any]:
        """
        Performs object tracking.

        Args:
            detections: A list of detected objects.
            frame: The current frame.

        Returns:
            A list of tracked objects.  => [[x1, y1, x2, y2, track_idx, conf, cls_idx, 0]]
        """
        result = self.tracker.update(detections, frame)
        return result