"""
Simpler, more reliable clock reading algorithm
Based on template matching and geometric analysis
"""
import cv2
import numpy as np
from typing import Tuple, Optional


def read_clock_time(clock_face: np.ndarray) -> Tuple[int, int, float]:
    """
    Read time from clock face using a simpler, more reliable method
    
    Strategy:
    1. Use Hough Transform to find ALL lines radiating from center
    2. Filter lines by strict geometric criteria
    3. Select the two most prominent lines as hands
    4. Identify which is hour vs minute based on length/thickness
    5. Calculate time from angles
    """
    if len(clock_face.shape) == 3:
        gray = cv2.cvtColor(clock_face, cv2.COLOR_BGR2GRAY)
    else:
        gray = clock_face.copy()
    
    h, w = gray.shape
    cx, cy = w // 2, h // 2
    radius = min(w, h) // 2 - 10
    
    # Create tight mask around clock center
    mask = np.zeros_like(gray)
    cv2.circle(mask, (cx, cy), int(radius * 0.9), 255, -1)
    
    # Aggressive preprocessing to isolate hands
    # Use morphological gradient to find edges of thick objects (hands)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    morph_grad = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
    morph_grad = cv2.bitwise_and(morph_grad, morph_grad, mask=mask)
    
    # Threshold to get strong edges
    _, binary = cv2.threshold(morph_grad, 30, 255, cv2.THRESH_BINARY)
    
    # Also try standard Canny
    blurred = cv2.GaussianBlur(gray, (5, 5), 1)
    canny = cv2.Canny(blurred, 40, 120)
    canny = cv2.bitwise_and(canny, canny, mask=mask)
    
    # Combine both
    combined = cv2.bitwise_or(binary, canny)
    
    # Detect lines using standard Hough Transform (not probabilistic)
    # This gives us lines through the entire image
    lines = cv2.HoughLines(combined, 1, np.pi/180, threshold=int(radius*0.3))
    
    if lines is None or len(lines) < 2:
        return 0, 0, 0.0
    
    # Convert Hough lines to angle representation
    line_angles = []
    for line in lines:
        rho, theta = line[0]
        
        # Convert to endpoint representation
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        
        # Check if line passes near center
        dist_to_center = abs(rho - np.sqrt(cx**2 + cy**2) * np.cos(theta - np.arctan2(cy, cx)))
        
        if dist_to_center > radius * 0.2:
            continue
        
        # Calculate angle of the line (0° = 12 o'clock, clockwise)
        # Theta from Hough is perpendicular to the line, so rotate by 90°
        line_angle = np.degrees(theta - np.pi/2)
        
        # Normalize to 0-360
        line_angle = (90 - line_angle) % 360
        
        line_angles.append(line_angle)
    
    if len(line_angles) < 2:
        # Fallback to probabilistic Hough
        lines_p = cv2.HoughLinesP(combined, 1, np.pi/180, 50, 
                                   minLineLength=int(radius*0.4), 
                                   maxLineGap=int(radius*0.2))
        
        if lines_p is None or len(lines_p) < 2:
            return 0, 0, 0.0
        
        hand_candidates = []
        for line in lines_p:
            x1, y1, x2, y2 = line[0]
            
            # Check if passes through center
            d1 = np.hypot(x1 - cx, y1 - cy)
            d2 = np.hypot(x2 - cx, y2 - cy)
            
            if min(d1, d2) > radius * 0.25:
                continue
            
            if max(d1, d2) < radius * 0.4:
                continue
            
            # Get tip angle
            tip_x, tip_y = (x1, y1) if d1 > d2 else (x2, y2)
            angle = calculate_angle(tip_x, tip_y, cx, cy)
            length = np.hypot(x2 - x1, y2 - y1)
            
            hand_candidates.append({
                'angle': angle,
                'length': length,
                'reach': max(d1, d2)
            })
        
        if len(hand_candidates) < 2:
            return 0, 0, 0.0
        
        # Remove duplicates
        hand_candidates.sort(key=lambda x: x['reach'], reverse=True)
        unique = []
        for cand in hand_candidates:
            is_dup = False
            for existing in unique:
                diff = abs(cand['angle'] - existing['angle'])
                diff = min(diff, 360 - diff)
                if diff < 20:
                    is_dup = True
                    break
            if not is_dup:
                unique.append(cand)
        
        if len(unique) < 2:
            return 0, 0, 0.0
        
        # Minute hand reaches farther
        minute_hand = unique[0]
        hour_hand = unique[1]
        
        minute_angle = minute_hand['angle']
        hour_angle = hour_hand['angle']
    else:
        # Remove duplicate angles from Hough
        line_angles.sort()
        unique_angles = []
        for angle in line_angles:
            is_dup = False
            for existing in unique_angles:
                diff = abs(angle - existing)
                diff = min(diff, 360 - diff)
                if diff < 20:
                    is_dup = True
                    break
            if not is_dup:
                unique_angles.append(angle)
        
        if len(unique_angles) < 2:
            return 0, 0, 0.0
        
        # Take first two as potential hands
        # We need to determine which is which
        # Use the fact that hands should be at specific angular relationships
        
        # Try all combinations and pick the one that makes most sense
        best_hour = 0
        best_minute = 0
        best_score = float('inf')
        
        for i in range(len(unique_angles)):
            for j in range(i+1, len(unique_angles)):
                # Try both assignments
                for minute_ang, hour_ang in [(unique_angles[i], unique_angles[j]),
                                              (unique_angles[j], unique_angles[i])]:
                    minute_val = int(round(minute_ang / 6.0)) % 60
                    hour_val = int(round((hour_ang - minute_val * 0.5) / 30.0)) % 12
                    
                    # Check consistency
                    expected_hour_ang = (hour_val * 30 + minute_val * 0.5) % 360
                    error = min(abs(expected_hour_ang - hour_ang), 360 - abs(expected_hour_ang - hour_ang))
                    
                    if error < best_score:
                        best_score = error
                        best_hour = hour_val
                        best_minute = minute_val
        
        if best_score > 30:
            return 0, 0, 0.0
        
        return best_hour, best_minute, max(0.5, 1.0 - best_score / 60.0)
    
    # Calculate time
    minute = int(round(minute_angle / 6.0)) % 60
    hour = int(round((hour_angle - minute * 0.5) / 30.0)) % 12
    
    # Validate
    expected_hour_angle = (hour * 30 + minute * 0.5) % 360
    error = min(abs(expected_hour_angle - hour_angle), 360 - abs(expected_hour_angle - hour_angle))
    
    # Try neighbors if error is large
    if error > 25:
        for test_h in [(hour - 1) % 12, (hour + 1) % 12]:
            test_angle = (test_h * 30 + minute * 0.5) % 360
            test_error = min(abs(test_angle - hour_angle), 360 - abs(test_angle - hour_angle))
            if test_error < error:
                error = test_error
                hour = test_h
    
    confidence = max(0.5, 1.0 - error / 60.0)
    
    return hour, minute, confidence


def calculate_angle(x: float, y: float, cx: float, cy: float) -> float:
    """Calculate angle from center, 0° at 12 o'clock, clockwise"""
    dx = x - cx
    dy = cy - y  # Invert y
    angle = np.degrees(np.arctan2(dy, dx))
    angle = (90 - angle) % 360
    return angle

