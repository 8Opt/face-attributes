import cv2
import base64
import httpx
from urllib.parse import urlparse

import numpy as np


def read_image(path_or_url):
    """
    Reads an image from a file or a URL using OpenCV.

    Args:
        path_or_url (str): The file path or URL of the image.

    Returns:
        numpy.ndarray: The loaded image as a NumPy array.
    """
    try:
        # Check if input is a URL
        parsed = urlparse(path_or_url)
        if parsed.scheme in ("http", "https"):
            # Read image from URL
            with httpx.stream("GET", path_or_url, timeout=10) as response:
                if response.status_code == 200:
                    # Read the response content as bytes
                    image_array = np.asarray(bytearray(response.read()), dtype="uint8")
                    # Decode the image using OpenCV
                    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
                    if image is None:
                        raise ValueError("Failed to decode image from URL.")
                    return image
                else:
                    raise ValueError(
                        f"Failed to fetch image from URL: {response.status_code}"
                    )
        else:
            # Read image from local file
            image = cv2.imread(path_or_url, cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("Failed to read image from the file path.")
            return image
    except Exception as e:
        print(f"Error: {e}")
        return None


def image_to_base64(image, image_format=".jpg"):
    """
    Encodes an image to a Base64 string.

    Args:
        image_path (str): The file path of the image.

    Returns:
        str: The Base64-encoded string of the image.
    """
    try:
        # Read the image file
        # Encode the image as a binary buffer
        _, buffer = cv2.imencode(image_format, image)

        # Convert the binary buffer to a Base64 string
        base64_string = base64.b64encode(buffer).decode("utf-8")

        return base64_string
    except Exception as e:
        print(f"Error: {e}")
        return None
