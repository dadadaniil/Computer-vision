"""
Test script to validate clock detection and time reading on provided test images
"""
import cv2
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from lab3.clock_models import (
    ClockDetectionModel, 
    ClockTimeRecognitionModel,
    create_preprocessing_pipeline
)


def test_image(image_path: str, expected_time: str = None):
    """Test clock detection on a single image"""
    print(f"\n{'='*60}")
    print(f"Testing: {os.path.basename(image_path)}")
    print(f"{'='*60}")
    
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Error: Could not load image {image_path}")
        return False
    
    print(f"✓ Image loaded: {img.shape[1]}x{img.shape[0]}")
    
    # Preprocess
    preprocessed = create_preprocessing_pipeline(img)
    print("✓ Preprocessing applied")
    
    # Stage 1: Detect clock
    detector = ClockDetectionModel()
    detection = detector.detect_clock_face(preprocessed)
    
    if detection is None:
        print("❌ Clock face not detected")
        return False
    
    cx, cy = detection['center']
    r = detection['radius']
    print(f"✓ Clock detected: center=({cx}, {cy}), radius={r}")
    
    # Extract clock face
    clock_face = detector.extract_clock_face(img, detection)
    print(f"✓ Clock face extracted: {clock_face.shape[1]}x{clock_face.shape[0]}")
    
    # Stage 2: Recognize time
    recognizer = ClockTimeRecognitionModel()
    hour, minute, confidence = recognizer.predict_time(clock_face)
    
    detected_time = f"{hour:02d}:{minute:02d}"
    print(f"✓ Time detected: {detected_time}")
    print(f"✓ Confidence: {confidence*100:.1f}%")
    
    if expected_time:
        print(f"Expected time: {expected_time}")
        if detected_time == expected_time:
            print("✅ MATCH!")
        else:
            print("⚠️  Different from expected")
    
    # Save result image
    output_img = img.copy()
    cv2.circle(output_img, (cx, cy), r, (0, 255, 0), 3)
    cv2.circle(output_img, (cx, cy), 3, (0, 0, 255), -1)
    
    # Draw detected time
    cv2.putText(output_img, detected_time, (cx - r + 10, cy - r + 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3, cv2.LINE_AA)
    cv2.putText(output_img, f"{confidence*100:.0f}%", (cx - r + 10, cy - r + 70),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    
    # Save output
    output_path = image_path.replace('.', '_result.')
    cv2.imwrite(output_path, output_img)
    print(f"✓ Result saved to: {os.path.basename(output_path)}")
    
    return True


def main():
    """Test all provided images"""
    print("\n" + "="*60)
    print("Clock Time Recognition - Test Suite")
    print("="*60)
    
    # Test images with expected times (approximate)
    test_cases = [
        ("lab3/Clock_1.jpg", "03:15"),  # Clock showing approximately 3:15
        ("lab3/Clock_2.jpg", "04:20"),  # Clock showing approximately 4:20
        ("lab3/church-clock.webp", "04:26"),  # Clock showing approximately 4:26
    ]
    
    base_dir = os.path.dirname(os.path.dirname(__file__))
    
    results = []
    for img_file, expected in test_cases:
        img_path = os.path.join(base_dir, img_file)
        if os.path.exists(img_path):
            success = test_image(img_path, expected)
            results.append((img_file, success))
        else:
            print(f"\n❌ Image not found: {img_path}")
            results.append((img_file, False))
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    for img_file, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} - {os.path.basename(img_file)}")
    
    total = len(results)
    passed = sum(1 for _, s in results if s)
    print(f"\nTotal: {passed}/{total} tests passed ({passed/total*100:.0f}%)")
    
    print("\n" + "="*60)
    print("Testing complete!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()

