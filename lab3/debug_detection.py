"""
Debug script to visualize hand detection
"""
import cv2
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from lab3.clock_models import ClockDetectionModel, create_preprocessing_pipeline


def visualize_hands(image_path):
    """Visualize detected hands with angles"""
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not load {image_path}")
        return
    
    print(f"\n{'='*60}")
    print(f"Analyzing: {os.path.basename(image_path)}")
    print('='*60)
    
    # Preprocess
    preprocessed = create_preprocessing_pipeline(img)
    
    # Detect clock
    detector = ClockDetectionModel()
    detection = detector.detect_clock_face(preprocessed)
    
    if detection is None:
        print("Clock not detected!")
        return
    
    cx, cy = detection['center']
    r = detection['radius']
    
    print(f"Clock center: ({cx}, {cy}), radius: {r}")
    
    # Extract clock face
    clock_face = detector.extract_clock_face(img, detection)
    
    # Get clock face dimensions
    h, w = clock_face.shape[:2]
    cf_cx, cf_cy = w // 2, h // 2
    cf_r = min(w, h) // 2 - 10
    
    # Visualize color masks
    if len(clock_face.shape) == 3:
        hsv = cv2.cvtColor(clock_face, cv2.COLOR_BGR2HSV)
        
        # Yellow/Gold mask
        lower_yellow = np.array([15, 50, 50])
        upper_yellow = np.array([35, 255, 255])
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
        
        # Black mask  
        lower_black = np.array([0, 0, 0])
        upper_black = np.array([180, 255, 50])
        mask_black = cv2.inRange(hsv, lower_black, upper_black)
        
        # Combined
        mask_combined = cv2.bitwise_or(mask_yellow, mask_black)
        
        # Find contours
        contours, _ = cv2.findContours(mask_combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        print(f"\nFound {len(contours)} contours")
        
        # Visualize
        vis = clock_face.copy()
        
        # Filter contours that could be hands
        hand_candidates = []
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area < 50:
                continue
            
            if len(contour) < 5:
                continue
            
            # Get contour properties
            M = cv2.moments(contour)
            if M["m00"] == 0:
                continue
                
            cx_cont = int(M["m10"] / M["m00"])
            cy_cont = int(M["m01"] / M["m00"])
            dist_to_center = np.hypot(cx_cont - cf_cx, cy_cont - cf_cy)
            
            # Fit line
            [vx, vy, x0, y0] = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01)
            
            # Calculate angle
            dx = float(vx)
            dy = -float(vy)  # Invert for image coords
            angle_rad = np.arctan2(dy, dx)
            angle_deg = np.degrees(angle_rad)
            angle_clock = (90 - angle_deg) % 360  # 0° at 12 o'clock
            
            rect = cv2.minAreaRect(contour)
            length = max(rect[1])
            
            print(f"\nContour {i}:")
            print(f"  Area: {area:.0f}, Length: {length:.1f}")
            print(f"  Distance from center: {dist_to_center:.1f}")
            print(f"  Angle: {angle_clock:.1f}°")
            
            # Draw contour
            cv2.drawContours(vis, [contour], -1, (0, 255, 0), 2)
            
            # Draw fitted line
            line_len = cf_r
            x1 = int(x0 + vx * line_len)
            y1 = int(y0 + vy * line_len)
            x2 = int(x0 - vx * line_len)
            y2 = int(y0 - vy * line_len)
            cv2.line(vis, (x1, y1), (x2, y2), (255, 0, 0), 2)
            
            # Draw angle text
            cv2.putText(vis, f"{angle_clock:.0f}deg", (cx_cont, cy_cont),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            hand_candidates.append({
                'area': area,
                'length': length,
                'dist': dist_to_center,
                'angle': angle_clock
            })
        
        # Show visualization
        cv2.circle(vis, (cf_cx, cf_cy), 5, (0, 0, 255), -1)
        
        # Save
        output_path = image_path.replace('.', '_debug.')
        cv2.imwrite(output_path, vis)
        cv2.imwrite(output_path.replace('_debug', '_mask'), mask_combined)
        
        print(f"\nVisualization saved to: {os.path.basename(output_path)}")
        print(f"Mask saved to: {os.path.basename(output_path.replace('_debug', '_mask'))}")
        
        # Determine which are hour and minute hands
        if len(hand_candidates) >= 2:
            hand_candidates.sort(key=lambda x: x['length'], reverse=True)
            print(f"\nLikely minute hand: {hand_candidates[0]['angle']:.1f}° (longer)")
            print(f"Likely hour hand: {hand_candidates[1]['angle']:.1f}° (shorter)")
            
            # Convert to time
            minute = int(round(hand_candidates[0]['angle'] / 6.0)) % 60
            hour_raw = (hand_candidates[1]['angle'] - minute * 0.5) / 30.0
            hour = int(round(hour_raw)) % 12
            
            print(f"\n→ Interpreted time: {hour:02d}:{minute:02d}")


if __name__ == "__main__":
    images = [
        "lab3/Clock_1.jpg",
        "lab3/Clock_2.jpg",
        "lab3/church-clock.webp"
    ]
    
    for img_path in images:
        if os.path.exists(img_path):
            visualize_hands(img_path)

