from typing import List, Any

from ultralytics import YOLO
from app.interface import ServiceInterface


class PersonDetect(ServiceInterface):
    """
    A service for detecting people using a YOLO model.
    """

    def __init__(
        self, model_path: str, name: str = "person_detect", threshold: float = 0.75
    ) -> None:
        """
        Initializes the PersonDetect service.

        Args:
            model_path: The path to the YOLO model.
            name: The name of the service. Defaults to 'person_detect'.
            threshold: The confidence threshold for person detection. Defaults to 0.75.
        """
        super().__init__(name=name)
        self.model = YOLO(model_path)
        self.threshold = threshold

    def inference(self, frame: Any) -> List[List[float]]:
        """
        Detects people in a given frame.

        Args:
            frame: The frame to process.

        Returns:
            A list of bounding boxes for detected people. Each bounding box is a list of [x1, y1, x2, y2, confidence, class_index].
        """

        # Perform person detection using the YOLO model
        model_results = self.model.predict(
            frame, classes=0, conf=self.threshold, verbose=False
        )

        if not model_results:
            return []

        # Extract and convert bounding boxes for detected people
        person_bboxes = model_results[0].boxes.data.cpu().numpy()
        person_bboxes[:, :4] = person_bboxes[:, :4].astype(
            int
        )  # Convert coordinates to integers

        return person_bboxes
