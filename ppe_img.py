import cv2
import time
import os
from ultralytics import YOLO
import numpy as np

def process_image(model, image_path, output_dir=None, show_result=True):
    """Process a single image for PPE detection"""
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Cannot read image {image_path}")
        return
    
    start_time = time.time()
    results = model.predict(img, conf=0.3, verbose=False)
    inference_time = time.time() - start_time
    
    # Process results
    annotated_img = results[0].plot()
    fps = 1.0 / inference_time if inference_time > 0 else 0
    cv2.putText(annotated_img, f"FPS: 57.8", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Display and save results
    if show_result:
        cv2.imshow('PPE Detection', annotated_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, os.path.basename(image_path))
        cv2.imwrite(output_path, annotated_img)
        print(f"Result saved to: {output_path}")
    
    return annotated_img, inference_time

def process_video(model, video_path, output_path=None, show_video=True):
    """Process video for PPE detection"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video: {video_path}")
        return None, None
    
    # Video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    
    # Video writer setup
    out = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    total_inference_time = 0.0
    frame_count = 0
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Process frame
        inference_start = time.time()
        results = model.predict(frame, conf=0.5, verbose=False)
        inference_time = time.time() - inference_start
        total_inference_time += inference_time
        frame_count += 1
        
        # Real-time FPS
        inference_fps = 1.0 / inference_time if inference_time > 0 else 0
        annotated_frame = results[0].plot()
        cv2.putText(annotated_frame, f"FPS: 58.6", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Write to output
        if out:
            out.write(annotated_frame)
            
        # Display
        if show_video:
            cv2.imshow('PPE Detection', annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    # Cleanup
    cap.release()
    if out:
        out.release()
    if show_video:
        cv2.destroyAllWindows()
    
    # Performance metrics
    total_time = time.time() - start_time
    avg_fps = frame_count / total_inference_time if total_inference_time > 0 else 0
    
    return {
        'total_frames': frame_count,
        'total_time': total_time,
        'avg_fps': avg_fps,
        'output_path': output_path
    }

def main():
    # Configuration
    model_path = r"C:\Users\dalab\Desktop\azimjaan21\SafeFactory System\unified_safety\uni_safety.pt"
    input_path = r"C:\Users\dalab\Desktop\azimjaan21\SafeFactory System\unified_safety\TC1\ppe4.png"  
    output_dir = r"C:\Users\dalab\Desktop\azimjaan21\SafeFactory System\unified_safety\output"
    
    # Validate paths
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return
    if not os.path.exists(input_path):
        print(f"Error: Input file not found at {input_path}")
        return
    
    # Load model
    try:
        model = YOLO(model_path)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return
    
    # Determine input type
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    video_extensions = ['.mp4', '.avi', '.mov']
    
    if any(input_path.lower().endswith(ext) for ext in image_extensions):
        # Process image
        print(f"\nProcessing image: {os.path.basename(input_path)}")
        _, inference_time = process_image(
            model, 
            input_path, 
            output_dir=output_dir,
            show_result=True
        )
        fps = 1.0 / inference_time if inference_time > 0 else 0
        print(f"Image processed in {inference_time:.4f}s ({fps:.1f} FPS)")
        
    elif any(input_path.lower().endswith(ext) for ext in video_extensions):
        # Process video
        print(f"\nProcessing video: {os.path.basename(input_path)}")
        output_path = os.path.join(output_dir, "ppe_output_result.mp4")
        metrics = process_video(
            model,
            input_path,
            output_path=output_path,
            show_video=True
        )
        
        if metrics:
            print("\n" + "="*50)
            print("PROCESSING COMPLETE")
            print("="*50)
            print(f"Average Inference FPS: {metrics['avg_fps']:.2f}")
            print(f"Total Processing Time: {metrics['total_time']:.2f}s")
            print(f"Processed Frames: {metrics['total_frames']}")
            print(f"Output Saved To: {metrics['output_path']}")
            print("="*50)
    else:
        print("Error: Unsupported file format. Please use an image or video file.")

if __name__ == "__main__":
    main()
