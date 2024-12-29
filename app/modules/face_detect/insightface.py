import httpx

from app.common.utils.file import image_to_base64
from app.interface import ServiceInterface


class FaceInsightExtractor(ServiceInterface):
    """
    A class for extracting face insights from an image frame using a remote service.
    """

    def __init__(
        self,
        url: str = "http://192.168.103.81:18080/extract",
        name: str = "insightface_service",
    ):
        """
        Initializes the FaceInsightExtractor.
        """
        super().__init__(name=name)
        self.url = url
        self.default_params = {
            "threshold": 0.6,
            "extract_ga": True,
            "extract_embedding": True,
            "return_face_data": False,
            "return_landmarks": True,
            "embed_only": False,
            "limit_faces": 0,
            "detect_masks": True,
            "msgpack": False,
        }

    def inference(self, frame, is_pretty: bool = True) -> list:
        """
        Extracts insights from an image frame.

        """

        try:
            response = httpx.post(
                url=self.url,
                json={
                    "images": {"data": [image_to_base64(frame)]},
                    **self.default_params,
                },
            )
            response.raise_for_status()  # Raise an exception for non-2xx status codes
        except httpx.HTTPStatusError as e:
            print(f"Error getting insights: {e}")
            return []

        data = response.json()

        if is_pretty:
            return data.get("data", [{}])[0].get("faces", [])
        else:
            return data
