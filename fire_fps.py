import cv2
import time
from ultralytics import YOLO
import os

def main():
    model_path = r"C:\Users\dalab\Desktop\azimjaan21\SafeFactory System\AI-COMS\KCL_test\TC2\fire.pt"
    video_path = r"C:\Users\dalab\Desktop\azimjaan21\SafeFactory System\AI-COMS\KCL_test\TC2\fire1.mp4"
    output_path = r"C:\Users\dalab\Desktop\azimjaan21\SafeFactory System\AI-COMS\KCL_test\TC2\fire_output_result3.mp4"

    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return

    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video: {video_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    total_inference_time = 0.0
    frame_count = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        inference_start = time.time()
        results = model.predict(frame, conf=0.5, verbose=False)
        inference_time = time.time() - inference_start
        total_inference_time += inference_time
        frame_count += 1

        # Real-time FPS
        inference_fps = 1.0 / inference_time if inference_time > 0 else 0
        annotated_frame = results[0].plot()
        cv2.putText(annotated_frame, f"Current FPS: {inference_fps:.1f}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        out.write(annotated_frame)
        cv2.imshow('PPE Detection', annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):

            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    total_time = time.time() - start_time
    if frame_count > 0:
        avg_fps = frame_count / total_inference_time
        print("\n" + "="*50)
        print(f"Average Inference FPS: {avg_fps:.2f}")
        print(f"Total Processing Time: {total_time:.2f}s")
        print(f"Processed Frames: {frame_count}")
        print(f"Output Saved To: {output_path}")
        print("="*50)
    else:
        print("Error: No frames processed")

if __name__ == "__main__":
    main()
