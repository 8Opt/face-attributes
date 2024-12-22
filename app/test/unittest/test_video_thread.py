import cv2
import time
import threading
from queue import Queue, Empty
from app.modules import FaceInsightExtractor, PersonDetect, Tracking
from app.common.utils.image import (
    adjust_bbox,
    adjust_landmarks,
    xyxy_to_xywh,
    draw_bounding_box,
    crop_image,
)

# Initialize models
model = PersonDetect(model_path="./weights/yolo11n.pt")
tracker = Tracking()
insightface = FaceInsightExtractor()


def capture_frames(video_path: str, frame_queue, stop_event):
    cap = cv2.VideoCapture(video_path or 0)
    while cap.isOpened() and not stop_event.is_set():
        success, frame = cap.read()
        if not success:
            break
        if not frame_queue.full():
            frame_queue.put(frame)
        else:
            time.sleep(0.01)  # Avoid spinning

    cap.release()
    stop_event.set()


def process_frames(frame_queue, result_queue, stop_event):
    while not stop_event.is_set():
        frame = frame_queue.get()  # Wait for frames

        # Detect people
        person_boxes = model.inference(frame=frame)
        track_resp = tracker.inference(detections=person_boxes, frame=frame)

        # Process each tracked person
        if len(track_resp) != 0:
            for resp in track_resp:
                person_frame = crop_image(image=frame, bbox=resp[:4])

                # Extract face insights
                face_resp = insightface.inference(frame=person_frame)
                if face_resp:
                    face_resp = face_resp[0]
                    bbox = face_resp.get("bbox")
                    bbox = adjust_bbox(resp[:4], bbox)

                    # Create caption
                    face_detection_prob = "{:.3f}".format(float(face_resp.get("prob")))
                    caption = f"Track ID: {int(resp[4])}-Face Detection Rate: {str(face_detection_prob)}"

                    if bbox:
                        frame = draw_bounding_box(
                            frame, xyxy_to_xywh(bbox), caption=caption
                        )

        # Put processed frame into the result queue
        if not result_queue.full():
            result_queue.put(frame)


def display_frames(result_queue, stop_event):
    while not stop_event.is_set():
        frame = result_queue.get()  # Wait for frames
        cv2.imshow("Tracking Person and Get Face Info", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            stop_event.set()
            break

    cv2.destroyAllWindows()


def main(
    video_path: str,
    frame_queue: Queue,
    result_queue: Queue,
    stop_event: threading.Event,
):
    start_time = time.time()

    capture_thread = threading.Thread(
        target=capture_frames, args=(video_path, frame_queue, stop_event)
    )
    process_thread = threading.Thread(
        target=process_frames, args=(frame_queue, result_queue, stop_event)
    )
    display_thread = threading.Thread(
        target=display_frames, args=(result_queue, stop_event)
    )

    capture_thread.start()
    process_thread.start()
    display_thread.start()

    capture_thread.join()
    process_thread.join()
    display_thread.join()

    print(f"Total processing time: {time.time() - start_time}")


if __name__ == "__main__":
    frame_queue = Queue(maxsize=10)
    result_queue = Queue(maxsize=10)
    stop_event = threading.Event()

    main(
        video_path="./examples/videos/face_detection.mp4",
        frame_queue=frame_queue,
        result_queue=result_queue,
        stop_event=stop_event,
    )

    """
    Original Length is 38s
    1/ Test with YOLOv8 => detect more accurate
        1. 11.272840738296509
        2. 11.265024423599243
        3. 11.573846340179443
        4. 11.272615909576416
        5. 11.24490237236023

    2/ Test with YOLOv11
        1. 11.3360595703125
        2. 11.224684238433838
        3. 11.339110374450684
        4. 11.085111856460571
        5. 11.240816831588745
    """
