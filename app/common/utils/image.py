import cv2


# ===== BOX =====
def xyxy_to_xywh(xyxy):
    """
    Converts a bounding box from (x_min, y_min, x_max, y_max) format to (x, y, width, height) format.

    Args:
        xyxy (tuple): A tuple (x_min, y_min, x_max, y_max).

    Returns:
        tuple: A tuple (x, y, width, height).
    """
    x_min, y_min, x_max, y_max = xyxy
    width = x_max - x_min
    height = y_max - y_min
    return (x_min, y_min, width, height)


def crop_image(image, bbox):
    """
    Crops an image using a bounding box in (x_min, y_min, x_max, y_max) format.
    Ensures the bounding box coordinates are integers.

    Args:
        image (numpy.ndarray): The input image as a NumPy array.
        bbox (tuple): The bounding box as (x_min, y_min, x_max, y_max).

    Returns:
        numpy.ndarray: The cropped image.
    """
    # Convert bounding box to integers
    x_min, y_min, x_max, y_max = map(int, bbox)

    # Ensure the coordinates are within the image boundaries
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(image.shape[1], x_max)
    y_max = min(image.shape[0], y_max)

    # Crop the image
    cropped_image = image[y_min:y_max, x_min:x_max]
    return cropped_image


def adjust_bbox(person_bbox, face_bbox):
    """
    Aligns the face bounding box (box2) with the person bounding box (box1)
    by shifting it to the center of the person bounding box.

    Args:
        person_bbox (list): Original person bounding box (x1, y1, x2, y2).
        face_bbox (list): Original face bounding box (x1, y1, x2, y2).

    Returns:
        list: Adjusted face bounding box (x1, y1, x2, y2).
    """

    # Calculate center of person bounding box
    person_center_x = (person_bbox[0] + person_bbox[2]) / 2
    person_center_y = (person_bbox[1] + person_bbox[3]) / 2

    # Calculate center of face bounding box
    face_center_x = (face_bbox[0] + face_bbox[2]) / 2
    face_center_y = (face_bbox[1] + face_bbox[3]) / 2

    # Calculate shift amounts
    x_shift = person_center_x - face_center_x
    y_shift = person_center_y - face_center_y

    # Shift face bounding box coordinates
    aligned_face_bbox = [
        face_bbox[0] + x_shift,
        face_bbox[1] + y_shift,
        face_bbox[2] + x_shift,
        face_bbox[3] + y_shift,
    ]

    return aligned_face_bbox


def adjust_landmarks(landmarks, face_bbox):
    """
    Adjusts landmark coordinates based on the face bounding box.

    Args:
        landmarks (list): List of landmark coordinates (x, y).
        face_bbox (list): Face bounding box (x1, y1, x2, y2).

    Returns:
        list: Adjusted landmark coordinates.
    """
    face_width = face_bbox[2] - face_bbox[0]
    face_height = face_bbox[3] - face_bbox[1]

    adjusted_landmarks = []
    for x, y in landmarks:
        # Normalize landmark coordinates to [0, 1] range within the face bbox
        normalized_x = (x - face_bbox[0]) / face_width
        normalized_y = (y - face_bbox[1]) / face_height

        # Adjust landmarks based on the new face bbox position (assuming aligned)
        adjusted_x = normalized_x * face_width + face_bbox[0]
        adjusted_y = normalized_y * face_height + face_bbox[1]

        adjusted_landmarks.append((adjusted_x, adjusted_y))

    return adjusted_landmarks


def draw_bounding_box(
    image,
    bbox,
    caption=None,
    bbox_color=(0, 255, 0),
    text_color=(0, 0, 0),
    thickness=2,
    font_scale=0.5,
):
    "Draw the bounding box"
    x, y, w, h = map(int, bbox)
    cv2.rectangle(image, (x, y), (x + w, y + h), bbox_color, thickness)

    "Draw the caption above the bounding box"
    if caption:
        # Calculate text size
        (text_width, text_height), _ = cv2.getTextSize(
            caption, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
        )
        # Background rectangle for the caption
        cv2.rectangle(
            image, (x, y - text_height - 10), (x + text_width, y), bbox_color, -1
        )
        # Add text on top of the rectangle
        cv2.putText(
            image,
            caption,
            (x, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            text_color,
            1,
            lineType=cv2.LINE_AA,
        )

    return image


def draw_landmarks(image, landmarks, color=(0, 0, 255), radius=2):
    """Draws landmarks on the frame."""
    for landmark in landmarks:
        x, y = landmark
        cv2.circle(image, (int(x), int(y)), radius, color, -1)
    return image
