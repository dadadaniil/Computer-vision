import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                             QLabel, QFileDialog, QGroupBox, QGridLayout, QComboBox, 
                             QCheckBox, QSpinBox, QProgressBar, QTextEdit, QTabWidget)
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtCore import Qt, QThread, pyqtSignal
import os
import sys

# Add lab3 directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'lab3'))

try:
    from lab3.clock_models import (ClockDetectionModel, ClockTimeRecognitionModel, 
                                   create_preprocessing_pipeline)
except ImportError:
    # Fallback if import fails
    ClockDetectionModel = None
    ClockTimeRecognitionModel = None
    create_preprocessing_pipeline = None


class ClockReaderApp(QWidget):
    def __init__(self):
        super().__init__()
        self.original_image = None
        self.processed_image = None
        self.gray_image = None
        
        # Initialize models
        self.detection_model = ClockDetectionModel() if ClockDetectionModel else None
        self.recognition_model = ClockTimeRecognitionModel() if ClockTimeRecognitionModel else None

        self.setWindowTitle('Лабораторная работа №3: Определение времени по стрелочным часам (Enhanced)')
        self.setGeometry(100, 100, 1400, 800)

        main_layout = QHBoxLayout()

        # Left controls
        controls_layout = QVBoxLayout()
        controls_widget = QWidget()
        controls_widget.setFixedWidth(360)
        controls_widget.setLayout(controls_layout)

        # Load button
        self.btn_load = QPushButton('Загрузить изображение')
        self.btn_load.clicked.connect(self.load_image)
        controls_layout.addWidget(self.btn_load)

        # Detection settings group
        group_settings = QGroupBox('Настройки обнаружения')
        settings_layout = QVBoxLayout()
        
        # Detection method selection
        method_layout = QHBoxLayout()
        method_layout.addWidget(QLabel('Метод:'))
        self.combo_method = QComboBox()
        self.combo_method.addItems(['Улучшенный (Enhanced)', 'Классический (Basic)', 'Гибридный (Hybrid)'])
        method_layout.addWidget(self.combo_method)
        settings_layout.addLayout(method_layout)
        
        # Preprocessing option
        self.check_preprocess = QCheckBox('Применить предобработку')
        self.check_preprocess.setChecked(True)
        settings_layout.addWidget(self.check_preprocess)
        
        # Show intermediate steps
        self.check_show_steps = QCheckBox('Показать промежуточные шаги')
        self.check_show_steps.setChecked(False)
        settings_layout.addWidget(self.check_show_steps)
        
        group_settings.setLayout(settings_layout)
        controls_layout.addWidget(group_settings)

        # Actions group
        group_actions = QGroupBox('Действия')
        group_layout = QVBoxLayout()
        
        self.btn_detect = QPushButton('Определить время')
        self.btn_detect.clicked.connect(self.detect_time)
        self.btn_detect.setEnabled(False)
        self.btn_detect.setStyleSheet("QPushButton { font-weight: bold; padding: 10px; }")
        group_layout.addWidget(self.btn_detect)

        self.label_result = QLabel('Результат: —')
        self.label_result.setAlignment(Qt.AlignCenter)
        result_font = QFont()
        result_font.setPointSize(14)
        result_font.setBold(True)
        self.label_result.setFont(result_font)
        self.label_result.setStyleSheet("QLabel { padding: 10px; background-color: #f0f0f0; border-radius: 5px; }")
        group_layout.addWidget(self.label_result)
        
        self.label_confidence = QLabel('Уверенность: —')
        self.label_confidence.setAlignment(Qt.AlignCenter)
        group_layout.addWidget(self.label_confidence)

        group_actions.setLayout(group_layout)
        controls_layout.addWidget(group_actions)
        
        # Info section
        group_info = QGroupBox('Информация')
        info_layout = QVBoxLayout()
        self.text_info = QTextEdit()
        self.text_info.setReadOnly(True)
        self.text_info.setMaximumHeight(200)
        self.text_info.setPlainText("Загрузите изображение для начала работы.\n\nПрограмма поддерживает:\n- Обнаружение циферблатов\n- Определение времени\n- Работу с декоративными элементами\n- Различные методы обработки")
        info_layout.addWidget(self.text_info)
        group_info.setLayout(info_layout)
        controls_layout.addWidget(group_info)
        
        controls_layout.addStretch()

        # Right images
        images_layout = QGridLayout()
        self.image_label_orig = QLabel('Исходное изображение')
        self.image_label_orig.setAlignment(Qt.AlignCenter)
        self.image_label_orig.setStyleSheet('border: 1px solid gray;')
        self.image_label_proc = QLabel('Обработанное изображение')
        self.image_label_proc.setAlignment(Qt.AlignCenter)
        self.image_label_proc.setStyleSheet('border: 1px solid gray;')
        images_layout.addWidget(self.image_label_orig, 0, 0)
        images_layout.addWidget(self.image_label_proc, 0, 1)

        main_layout.addWidget(controls_widget)
        main_layout.addLayout(images_layout)
        self.setLayout(main_layout)

    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, 'Открыть изображение', '', 'Image Files (*.png *.jpg *.jpeg *.bmp *.webp)')
        if not file_path:
            return
        img = cv2.imread(file_path)
        if img is None:
            return
        self.original_image = img
        self.gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.processed_image = None
        self.display_image(self.original_image, 'original')
        self.image_label_proc.clear()
        self.image_label_proc.setText('Обработанное изображение')
        self.label_result.setText('Результат: —')
        self.btn_detect.setEnabled(True)

    def display_image(self, image, where):
        if image is None:
            return
        if len(image.shape) == 2:
            qimg = QImage(image.data, image.shape[1], image.shape[0], image.strides[0], QImage.Format_Grayscale8)
        else:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            qimg = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        if where == 'original':
            self.image_label_orig.setPixmap(pixmap.scaled(self.image_label_orig.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            self.image_label_proc.setPixmap(pixmap.scaled(self.image_label_proc.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    @staticmethod
    def angle_from_vector(x, y, cx, cy):
        vx = x - cx
        vy = cy - y  # invert y for standard math orientation
        angle = np.degrees(np.arctan2(vy, vx))
        angle = (90 - angle) % 360  # 0 deg at 12 o'clock, clockwise
        return angle

    def detect_time(self):
        if self.original_image is None:
            return

        method = self.combo_method.currentText()
        use_preprocess = self.check_preprocess.isChecked()
        show_steps = self.check_show_steps.isChecked()
        
        # Update info
        self.text_info.setPlainText(f"Обработка изображения...\nМетод: {method}\nПредобработка: {'Да' if use_preprocess else 'Нет'}")
        
        try:
            # Preprocessing
            img = self.original_image.copy()
            if use_preprocess and create_preprocessing_pipeline:
                img = create_preprocessing_pipeline(img)
                self.text_info.append("✓ Предобработка выполнена")
            
            # Stage 1: Clock Detection
            if 'Улучшенный' in method and self.detection_model:
                detection = self.detection_model.detect_clock_face(img)
            else:
                detection = self._detect_clock_classic(img)
            
            if detection is None:
                self.label_result.setText('Результат: Циферблат не найден')
                self.label_confidence.setText('Уверенность: 0%')
                self.text_info.append("✗ Циферблат не обнаружен")
                return
            
            self.text_info.append(f"✓ Циферблат найден: центр ({detection['center'][0]}, {detection['center'][1]}), радиус {detection['radius']}")
            
            # Visualize detection
            vis_img = img.copy()
            cx, cy = detection['center']
            r = detection['radius']
            cv2.circle(vis_img, (cx, cy), r, (0, 255, 0), 3)
            cv2.circle(vis_img, (cx, cy), 3, (0, 0, 255), -1)
            
            # Extract clock face
            if self.detection_model:
                clock_face = self.detection_model.extract_clock_face(img, detection)
            else:
                x, y, w, h = detection['bbox']
                clock_face = img[y:y+h, x:x+w]
            
            # Stage 2: Time Recognition
            if 'Улучшенный' in method and self.recognition_model:
                hour, minute, confidence = self.recognition_model.predict_time(clock_face)
            else:
                hour, minute, confidence = self._detect_time_classic(img, detection)
            
            self.text_info.append(f"✓ Время определено: {hour:02d}:{minute:02d}")
            self.text_info.append(f"✓ Уверенность: {confidence*100:.1f}%")
            
            # Draw hands visualization
            vis_img = self._draw_time_visualization(vis_img, detection, hour, minute)
            
            # Display results
            time_text = f"{hour:02d}:{minute:02d}"
            self.label_result.setText(f'Результат: {time_text}')
            self.label_confidence.setText(f'Уверенность: {confidence*100:.0f}%')
            
            # Add time text to image
            cv2.putText(vis_img, time_text, (cx - r + 10, cy - r + 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3, cv2.LINE_AA)
            cv2.putText(vis_img, f"{confidence*100:.0f}%", (cx - r + 10, cy - r + 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            
            self.processed_image = vis_img
            self.display_image(self.processed_image, 'processed')
            
        except Exception as e:
            self.label_result.setText('Результат: Ошибка обработки')
            self.label_confidence.setText('Уверенность: 0%')
            self.text_info.append(f"✗ Ошибка: {str(e)}")
            print(f"Error in detect_time: {e}")
            import traceback
            traceback.print_exc()
    
    def _detect_clock_classic(self, img):
        """Classic clock detection using Hough Circles"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Try multiple parameter sets
        param_sets = [
            {'dp': 1.2, 'minDist': 100, 'param1': 120, 'param2': 40},
            {'dp': 1.0, 'minDist': 100, 'param1': 100, 'param2': 35},
            {'dp': 1.5, 'minDist': 80, 'param1': 150, 'param2': 45},
        ]
        
        all_circles = []
        for params in param_sets:
            circles = cv2.HoughCircles(
                blurred, cv2.HOUGH_GRADIENT,
                dp=params['dp'], minDist=params['minDist'],
                param1=params['param1'], param2=params['param2'],
                minRadius=30, maxRadius=min(gray.shape) // 2
            )
            if circles is not None:
                all_circles.extend(circles[0])
        
        if not all_circles:
            return None
        
        # Select largest circle
        circles_sorted = sorted(all_circles, key=lambda c: c[2], reverse=True)
        cx, cy, r = circles_sorted[0]
        
        return {
            'center': (int(cx), int(cy)),
            'radius': int(r),
            'bbox': (int(cx - r), int(cy - r), int(2 * r), int(2 * r))
        }
    
    def _detect_time_classic(self, img, detection):
        """Classic time detection using line detection - IMPROVED"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        cx, cy = detection['center']
        r = detection['radius']
        
        # Enhanced preprocessing
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        
        # Create mask for dial
        mask = np.zeros_like(gray)
        cv2.circle(mask, (cx, cy), int(r * 0.98), 255, -1)
        
        # Strong edge detection
        edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
        edges = cv2.bitwise_and(edges, edges, mask=mask)
        
        # Strengthen edges
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        
        # Detect lines
        lines = cv2.HoughLinesP(
            edges, 
            rho=1,
            theta=np.pi / 180,
            threshold=30,
            minLineLength=int(r * 0.25),
            maxLineGap=int(r * 0.15)
        )
        
        if lines is None or len(lines) < 2:
            return 0, 0, 0.0
        
        # Find hand candidates
        hand_candidates = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            d1 = np.hypot(x1 - cx, y1 - cy)
            d2 = np.hypot(x2 - cx, y2 - cy)
            inner = min(d1, d2)
            outer = max(d1, d2)
            
            # Must pass through center and extend outward
            if inner < r * 0.2 and outer > r * 0.35:
                tip_x, tip_y = (x1, y1) if d1 > d2 else (x2, y2)
                angle = self.angle_from_vector(tip_x, tip_y, cx, cy)
                length = np.hypot(x2 - x1, y2 - y1)
                
                hand_candidates.append({
                    'angle': angle,
                    'length': length,
                    'outer': outer,
                    'coords': (x1, y1, x2, y2)
                })
        
        if len(hand_candidates) < 2:
            return 0, 0, 0.0
        
        # Remove duplicates
        hand_candidates.sort(key=lambda x: (x['length'], x['outer']), reverse=True)
        unique_hands = []
        for cand in hand_candidates:
            is_dup = False
            for existing in unique_hands:
                diff = abs(cand['angle'] - existing['angle'])
                diff = min(diff, 360 - diff)
                if diff < 15:
                    if cand['length'] > existing['length']:
                        unique_hands.remove(existing)
                        unique_hands.append(cand)
                    is_dup = True
                    break
            if not is_dup:
                unique_hands.append(cand)
        
        if len(unique_hands) < 2:
            return 0, 0, 0.0
        
        # Sort by outer distance
        unique_hands.sort(key=lambda x: x['outer'], reverse=True)
        
        minute_hand = unique_hands[0]
        hour_hand = unique_hands[1]
        
        # Convert angles to time
        minute_angle = minute_hand['angle']
        hour_angle = hour_hand['angle']
        
        # Calculate minute
        minute = int(round(minute_angle / 6.0)) % 60
        
        # Calculate hour (considering minute position)
        hour_from_angle = (hour_angle - minute * 0.5) / 30.0
        hour = int(round(hour_from_angle)) % 12
        
        # Validate
        expected_hour_angle = (hour * 30 + minute * 0.5) % 360
        error = min(abs(expected_hour_angle - hour_angle), 360 - abs(expected_hour_angle - hour_angle))
        
        # Try neighbors if error is large
        if error > 20:
            best_hour = hour
            min_error = error
            for test_h in [(hour - 1) % 12, (hour + 1) % 12]:
                test_angle = (test_h * 30 + minute * 0.5) % 360
                test_error = min(abs(test_angle - hour_angle), 360 - abs(test_angle - hour_angle))
                if test_error < min_error:
                    min_error = test_error
                    best_hour = test_h
            hour = best_hour
            error = min_error
        
        # Confidence based on error
        if error < 10:
            confidence = 0.9
        elif error < 20:
            confidence = 0.75
        elif error < 30:
            confidence = 0.6
        else:
            confidence = 0.4
        
        return hour, minute, confidence
    
    def _draw_time_visualization(self, img, detection, hour, minute):
        """Draw clock hands based on detected time"""
        cx, cy = detection['center']
        r = detection['radius']
        
        # Calculate hand angles
        minute_angle = np.radians(minute * 6 - 90)
        hour_angle = np.radians(((hour % 12) * 30 + minute * 0.5) - 90)
        
        # Draw minute hand
        minute_len = r * 0.8
        mx = int(cx + minute_len * np.cos(minute_angle))
        my = int(cy + minute_len * np.sin(minute_angle))
        cv2.line(img, (cx, cy), (mx, my), (255, 255, 0), 4)
        cv2.circle(img, (mx, my), 6, (255, 255, 0), -1)
        
        # Draw hour hand
        hour_len = r * 0.5
        hx = int(cx + hour_len * np.cos(hour_angle))
        hy = int(cy + hour_len * np.sin(hour_angle))
        cv2.line(img, (cx, cy), (hx, hy), (255, 0, 0), 6)
        cv2.circle(img, (hx, hy), 6, (255, 0, 0), -1)
        
        return img


if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = ClockReaderApp()
    w.show()
    sys.exit(app.exec_())


