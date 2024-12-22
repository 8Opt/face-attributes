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


# ===== DRAWING =====
def drawing(
    image,
    bbox,
    landmarks,
    caption=None,
    bbox_color=(0, 255, 0),
    landmark_color=(0, 0, 255),
    text_color=(0, 0, 0),
    thickness=2,
    radius=3,
    font_scale=0.5,
):
    """
    Draws a bounding box, landmarks, and an optional caption on an image.

    Args:
        image (numpy.ndarray): The image on which to draw.
        bbox (tuple): Bounding box as (x, y, w, h).
        landmarks (list of tuples): List of (x, y) coordinates for the landmarks.
        caption (str): Optional text to display above the bounding box.
        bbox_color (tuple): Color of the bounding box in BGR (default: green).
        landmark_color (tuple): Color of the landmarks in BGR (default: red).
        text_color (tuple): Color of the caption text in BGR (default: white).
        thickness (int): Thickness of the bounding box lines.
        radius (int): Radius of the landmarks.
        font_scale (float): Font scale for the caption text.

    Returns:
        numpy.ndarray: The image with the bounding box, landmarks, and caption drawn.
    """
    # Draw the bounding box
    x, y, w, h = bbox
    cv2.rectangle(image, (x, y), (x + w, y + h), bbox_color, thickness)

    # Draw the landmarks
    for landmark in landmarks:
        lx, ly = landmark
        cv2.circle(image, (int(lx), int(ly)), radius, landmark_color, -1)

    # Draw the caption above the bounding box
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
