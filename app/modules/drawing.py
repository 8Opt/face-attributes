import cv2

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
