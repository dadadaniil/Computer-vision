"""
Example usage of the clock detection and time recognition system
"""
import cv2
import os
from clock_models import (
    ClockDetectionModel,
    ClockTimeRecognitionModel,
    create_preprocessing_pipeline
)


def process_clock_image(image_path: str, output_path: str = None):
    """
    Complete example of processing a clock image
    """
    print(f"Processing: {image_path}")
    
    # Step 1: Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load {image_path}")
        return None
    
    print(f"✓ Loaded image: {image.shape[1]}x{image.shape[0]} pixels")
    
    # Step 2: Preprocess image
    preprocessed = create_preprocessing_pipeline(image)
    print("✓ Applied preprocessing (denoising + contrast enhancement)")
    
    # Step 3: Detect clock face
    detector = ClockDetectionModel()
    detection = detector.detect_clock_face(preprocessed)
    
    if detection is None:
        print("✗ Could not detect clock face")
        return None
    
    center = detection['center']
    radius = detection['radius']
    print(f"✓ Detected clock: center=({center[0]}, {center[1]}), radius={radius}")
    
    # Step 4: Extract clock face region
    clock_face = detector.extract_clock_face(image, detection)
    print(f"✓ Extracted clock face: {clock_face.shape[1]}x{clock_face.shape[0]} pixels")
    
    # Step 5: Recognize time
    recognizer = ClockTimeRecognitionModel()
    hour, minute, confidence = recognizer.predict_time(clock_face)
    
    time_str = f"{hour:02d}:{minute:02d}"
    print(f"✓ Detected time: {time_str}")
    print(f"✓ Confidence: {confidence*100:.1f}%")
    
    # Step 6: Visualize results
    result_image = image.copy()
    
    # Draw detected circle
    cv2.circle(result_image, center, radius, (0, 255, 0), 3)
    cv2.circle(result_image, center, 3, (0, 0, 255), -1)
    
    # Draw time text
    text_pos = (center[0] - radius + 10, center[1] - radius + 40)
    cv2.putText(result_image, time_str, text_pos,
               cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 255, 0), 3, cv2.LINE_AA)
    
    confidence_pos = (center[0] - radius + 10, center[1] - radius + 80)
    cv2.putText(result_image, f"{confidence*100:.0f}%", confidence_pos,
               cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    
    # Save output
    if output_path:
        cv2.imwrite(output_path, result_image)
        print(f"✓ Saved result to: {output_path}")
    
    return {
        'time': time_str,
        'hour': hour,
        'minute': minute,
        'confidence': confidence,
        'detection': detection,
        'result_image': result_image
    }


def batch_process_directory(input_dir: str, output_dir: str = None):
    """
    Process all images in a directory
    """
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    image_extensions = ['.jpg', '.jpeg', '.png', '.webp', '.bmp']
    image_files = [f for f in os.listdir(input_dir) 
                   if any(f.lower().endswith(ext) for ext in image_extensions)]
    
    print(f"\nProcessing {len(image_files)} images from {input_dir}")
    print("=" * 60)
    
    results = []
    for img_file in image_files:
        input_path = os.path.join(input_dir, img_file)
        output_path = None
        if output_dir:
            output_path = os.path.join(output_dir, f"result_{img_file}")
        
        result = process_clock_image(input_path, output_path)
        if result:
            results.append({
                'filename': img_file,
                'time': result['time'],
                'confidence': result['confidence']
            })
        print("-" * 60)
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for r in results:
        print(f"{r['filename']:30s} → {r['time']} ({r['confidence']*100:.0f}%)")
    print(f"\nSuccessfully processed: {len(results)}/{len(image_files)} images")
    
    return results


if __name__ == "__main__":
    # Example 1: Process a single image
    print("\n" + "="*60)
    print("Example 1: Single Image Processing")
    print("="*60)
    
    if os.path.exists("Clock_1.jpg"):
        result = process_clock_image("Clock_1.jpg", "Clock_1_output.jpg")
        if result:
            print(f"\n🎯 Final result: {result['time']} (confidence: {result['confidence']*100:.0f}%)")
    
    # Example 2: Batch processing
    print("\n" + "="*60)
    print("Example 2: Batch Processing")
    print("="*60)
    
    # Process all test images in current directory
    if any(os.path.exists(f) for f in ["Clock_1.jpg", "Clock_2.jpg", "church-clock.webp"]):
        results = batch_process_directory(".", "results")
    
    print("\n" + "="*60)
    print("Examples complete!")
    print("="*60)

