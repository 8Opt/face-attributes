import cv2
import time

from app.modules import (FaceInsightExtractor, PersonDetect, Tracking)

from app.common.utils.image import (adjust_bbox, adjust_landmarks, xyxy_to_xywh, 
                                    draw_bounding_box, draw_landmarks, 
                                    crop_image)


model = PersonDetect(model_path='./weights/yolo11n.pt')
tracker = Tracking()
insightface = FaceInsightExtractor()

def test_video(video_path: str = ""):
    """
    Processes a video stream for person detection, tracking, and face recognition.

    Args:
        video_path: The path to the video file (optional). Defaults to webcam capture.
    """

    # Initialize capture device (webcam by default)
    cap = cv2.VideoCapture(video_path or 0)

    # Performance tracking
    start_time = time.time()
    track_counter = {}

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Detect people
        person_boxes = model.inference(frame=frame)

        track_resp = tracker.inference(detections=person_boxes, frame=frame)

        # Process each tracked person
        if len(track_resp) != 0:
            for resp in track_resp:
                person_frame = crop_image(image=frame, bbox=resp[:4])  # Crop using track_resp

                # Extract face insights
                face_resp = insightface.inference(frame=person_frame)
                if face_resp:
                    face_resp = face_resp[0]

                    # Adjust bounding box and landmarks (if available)
                    bbox = face_resp.get("bbox")
                    bbox = adjust_bbox(resp[:4], bbox)
                    landmarks = face_resp.get("landmarks")
                    landmarks = adjust_landmarks(landmarks, bbox)

                    # Create caption (optional)
                    caption = f"Track ID: {resp[4]}\nFace Detection Rate: {str(face_resp.get('prob'))}\n"

                    # Update track information
                    track_counter[resp[4]] = (bbox, landmarks, caption)

                    # Draw bounding box and landmarks (if available)
                    if bbox:
                        frame = draw_bounding_box(frame, xyxy_to_xywh(bbox), caption=caption)
                    # if landmarks:
                    #     frame = draw_landmarks(frame, landmarks)

        # Display frame with bounding boxes and landmarks (if drawn)
        cv2.imshow("Tracking Person and Get Face Info", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    print(f"Total processing time: {time.time() - start_time}")


if __name__ == "__main__":
    test_video(video_path='./examples/videos/face_detection.mp4')  
    """
    Original clip's length: 38s
    
    Inference cost: 
    1/ 504.13942646980286s => load model YOLOv11 every inference. 
    2/ Load model YOLOv11 before every inference
        1. 354.20252084732056
        2. 347.8046991825104
        3. 458.28852438926697
    """