import cv2

from app.modules import (FaceInsightExtractor, PersonDetect, Tracking,
                         draw_bounding_box, draw_landmarks)


def test_inference():
    """Performs inference on a test image using face, person, and tracking services."""

    # Load image
    image = cv2.imread(filename="./examples/test_00.jpg")

    # Person detection
    person_boxes = PersonDetect(model_path='./weights/yolo11n.pt').inference(frame=image)

    # Tracking (assuming detections are already in person_boxes)
    track_resp = Tracking().inference(detections=person_boxes, frame=image)
    print(track_resp)

    for resp in track_resp: 
        # Face extraction (assuming a single person is detected)
        face_resp = FaceInsightExtractor().inference(frame=image)[0]
        track_person = int(resp[4])
        print(f"Track ID: {track_person}\n-> Detection Rate: {face_resp.get('prob')}\
              \nBounding box: {face_resp.get('bbox')}")
        # Draw bounding box and landmarks (if available)
        image = draw_bounding_box(
            image=image,
            bbox=face_resp.get('bbox'),
            caption=f"Track ID: {int(resp[4])}"
        )

        if face_resp.get('landmarks'):
            image = draw_landmarks(image=image, landmarks=face_resp.get('landmarks'))

    # Display image
    cv2.imshow("Test Pipeline", image)
    cv2.waitKey(0)  # Wait for key press to close the window
    cv2.destroyAllWindows()


if __name__ == "__main__":
    test_inference()