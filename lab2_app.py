import sys
import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                             QLabel, QFileDialog, QGroupBox, QMessageBox)
from PyQt5.QtGui import QImage, QPixmap, QCursor
from PyQt5.QtCore import Qt, pyqtSignal


class ClickableLabel(QLabel):
    clicked = pyqtSignal(int, int)

    def __init__(self, parent=None):
        super(ClickableLabel, self).__init__(parent)
        self.setCursor(QCursor(Qt.CrossCursor))

    def mousePressEvent(self, event):
        self.clicked.emit(event.x(), event.y())
        super().mousePressEvent(event)


class ImageProcessorApp(QWidget):
    def __init__(self):
        super().__init__()
        self.original_image = None
        self.processed_image = None
        self.grayscale_image = None
        self.is_segmentation_mode = False
        self.original_pixmap_size = None
        self.setWindowTitle('Лабораторная работа №1-2: Обработка изображений')
        self.setGeometry(100, 100, 1200, 700)
        main_layout = QHBoxLayout()
        controls_layout = QVBoxLayout()
        controls_widget = QWidget()
        controls_widget.setFixedWidth(300)
        controls_widget.setLayout(controls_layout)
        self.load_btn = QPushButton('Загрузить изображение')
        self.load_btn.clicked.connect(self.load_image)
        controls_layout.addWidget(self.load_btn)
        group1 = QGroupBox("ЛР1: Предварительная обработка")
        group1_layout = QVBoxLayout()
        group1.setLayout(group1_layout)
        controls_layout.addWidget(group1)
        group_lab2 = QGroupBox("ЛР2: Выделение признаков")
        group_lab2_layout = QVBoxLayout()
        btn_hough_lines = QPushButton('a. Найти линии (Хаф)')
        btn_hough_lines.clicked.connect(self.find_hough_lines)
        group_lab2_layout.addWidget(btn_hough_lines)
        btn_hough_circles = QPushButton('a. Найти окружности (Хаф)')
        btn_hough_circles.clicked.connect(self.find_hough_circles)
        group_lab2_layout.addWidget(btn_hough_circles)
        btn_texture_segment = QPushButton('c. Сегментация по текстуре')
        btn_texture_segment.clicked.connect(self.start_texture_segmentation)
        group_lab2_layout.addWidget(btn_texture_segment)
        group_lab2.setLayout(group_lab2_layout)
        controls_layout.addWidget(group_lab2)
        controls_layout.addStretch()
        images_layout = QVBoxLayout()
        self.image_label_orig = ClickableLabel('Исходное изображение')
        self.image_label_orig.setAlignment(Qt.AlignCenter)
        self.image_label_orig.setStyleSheet("border: 1px solid gray;")
        self.image_label_orig.clicked.connect(self.image_clicked)
        self.image_label_proc = QLabel('Обработанное изображение')
        self.image_label_proc.setAlignment(Qt.AlignCenter)
        self.image_label_proc.setStyleSheet("border: 1px solid gray;")
        images_layout.addWidget(self.image_label_orig)
        images_layout.addWidget(self.image_label_proc)
        main_layout.addWidget(controls_widget)
        main_layout.addLayout(images_layout)
        self.setLayout(main_layout)

    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, 'Открыть изображение', '', 'Image Files (*.png *.jpg *.bmp)')
        if file_path:
            self.original_image = cv2.imread(file_path)
            self.grayscale_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
            self.display_image(self.original_image, 'original')
            self.image_label_proc.clear()
            self.image_label_proc.setText('Обработанное изображение')
            self.is_segmentation_mode = False

    def find_hough_lines(self):
        if self.grayscale_image is not None:
            edges = cv2.Canny(self.grayscale_image, 50, 150, apertureSize=3)
            lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)
            output_image = self.original_image.copy()
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    cv2.line(output_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            self.processed_image = output_image
            self.display_image(self.processed_image, 'processed')

    def find_hough_circles(self):
        if self.grayscale_image is not None:
            img_blur = cv2.medianBlur(self.grayscale_image, 5)
            circles = cv2.HoughCircles(img_blur, cv2.HOUGH_GRADIENT, 1, 20,
                                       param1=50, param2=30, minRadius=0, maxRadius=0)
            output_image = self.original_image.copy()
            if circles is not None:
                circles = np.uint16(np.around(circles))
                for i in circles[0, :]:
                    cv2.circle(output_image, (i[0], i[1]), i[2], (0, 255, 0), 2)
                    cv2.circle(output_image, (i[0], i[1]), 2, (0, 0, 255), 3)
            self.processed_image = output_image
            self.display_image(self.processed_image, 'processed')

    def start_texture_segmentation(self):
        if self.original_image is None:
            QMessageBox.warning(self, "Ошибка", "Сначала загрузите изображение.")
            return
        self.is_segmentation_mode = True
        QMessageBox.information(self, "Сегментация", "Выберите 'затравочную' точку на исходном изображении.")

    def image_clicked(self, x_label, y_label):
        if not self.is_segmentation_mode or self.original_image is None:
            return
        h_orig, w_orig, _ = self.original_image.shape
        w_label = self.image_label_orig.width()
        h_label = self.image_label_orig.height()
        pixmap = self.image_label_orig.pixmap()
        if pixmap is None: return
        w_pix = pixmap.width()
        h_pix = pixmap.height()
        x = int(x_label * w_orig / w_pix)
        y = int(y_label * h_orig / h_pix)
        if 0 <= x < w_orig and 0 <= y < h_orig:
            self.perform_region_growing(x, y)
        self.is_segmentation_mode = False

    def perform_region_growing(self, seed_x, seed_y):
        patch_size = 21
        radius = 2
        n_points = 8 * radius
        half_patch = patch_size // 2
        lbp = local_binary_pattern(self.grayscale_image, n_points, radius, 'uniform')
        seed_patch_lbp = lbp[seed_y - half_patch: seed_y + half_patch + 1,
        seed_x - half_patch: seed_x + half_patch + 1]
        target_hist, _ = np.histogram(seed_patch_lbp, bins=n_points + 2, range=(0, n_points + 2), density=True)
        h, w = self.grayscale_image.shape
        mask = np.zeros((h, w), dtype=np.uint8)
        queue = [(seed_x, seed_y)]
        similarity_threshold = 0.5
        while queue:
            x, y = queue.pop(0)
            if not (0 <= x < w and 0 <= y < h) or mask[y, x] != 0:
                continue
            current_patch_lbp = lbp[y - half_patch: y + half_patch + 1,
            x - half_patch: x + half_patch + 1]
            if current_patch_lbp.size == 0: continue
            current_hist, _ = np.histogram(current_patch_lbp, bins=n_points + 2, range=(0, n_points + 2), density=True)
            score = cv2.compareHist(target_hist.astype(np.float32), current_hist.astype(np.float32), cv2.HISTCMP_CORREL)
            if score > similarity_threshold:
                mask[y, x] = 255
                queue.append((x + 1, y))
                queue.append((x - 1, y))
                queue.append((x, y + 1))
                queue.append((x, y - 1))
        output_image = self.original_image.copy()
        output_image[mask == 255] = [0, 255, 0]
        self.processed_image = output_image
        self.display_image(self.processed_image, 'processed')

    def display_image(self, image, window_type):
        if image is None: return

        if len(image.shape) == 3:
            h, w, ch = image.shape
            q_format = QImage.Format_BGR888
            q_img = QImage(image.data, w, h, ch * w, q_format)
        else:
            h, w = image.shape
            q_format = QImage.Format_Grayscale8
            q_img = QImage(image.data, w, h, w, q_format)

        pixmap = QPixmap.fromImage(q_img)

        if window_type == 'original':
            scaled_pixmap = pixmap.scaled(self.image_label_orig.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.image_label_orig.setPixmap(scaled_pixmap)
        elif window_type == 'processed':
            scaled_pixmap = pixmap.scaled(self.image_label_proc.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.image_label_proc.setPixmap(scaled_pixmap)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_app = ImageProcessorApp()
    main_app.show()
    sys.exit(app.exec_())