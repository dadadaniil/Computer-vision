python
import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                             QLabel, QFileDialog, QSizePolicy, QGroupBox, QGridLayout)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt

class ImageProcessorApp(QWidget):
    def __init__(self):
        super().__init__()
        # Initialize image variables
        self.original_image = None
        self.processed_image = None
        self.grayscale_image = None

        # Set window properties
        self.setWindowTitle('Лабораторная работа №1: Обработка изображений')
        self.setGeometry(100, 100, 1200, 700)

        # Main layout
        main_layout = QHBoxLayout()

        # Controls layout (left panel)
        controls_layout = QVBoxLayout()
        controls_widget = QWidget()
        controls_widget.setFixedWidth(300)
        controls_widget.setLayout(controls_layout)

        # Button to load image
        self.load_btn = QPushButton('Загрузить изображение')
        self.load_btn.clicked.connect(self.load_image)
        controls_layout.addWidget(self.load_btn)

        # Group 1: Preprocessing operations
        group1 = QGroupBox("a. Предварительная обработка")
        group1_layout = QVBoxLayout()
        btn_gray_avg = QPushButton('Оттенки серого (усреднение)')
        btn_gray_avg.clicked.connect(self.convert_to_gray_avg)
        group1_layout.addWidget(btn_gray_avg)
        btn_gray_hsv = QPushButton('Оттенки серого (стандартный)')
        btn_gray_hsv.clicked.connect(self.convert_to_gray_standard)
        group1_layout.addWidget(btn_gray_hsv)
        btn_binarize_thresh = QPushButton('Бинаризация (порог=127)')
        btn_binarize_thresh.clicked.connect(self.binarize_threshold)
        group1_layout.addWidget(btn_binarize_thresh)
        btn_binarize_otsu = QPushButton('Бинаризация (метод Отсу)')
        btn_binarize_otsu.clicked.connect(self.binarize_otsu)
        group1_layout.addWidget(btn_binarize_otsu)
        btn_hist_eq = QPushButton('Эквализация гистограммы')
        btn_hist_eq.clicked.connect(self.equalize_histogram)
        group1_layout.addWidget(btn_hist_eq)
        btn_hist_stretch = QPushButton('Растяжение гистограммы')
        btn_hist_stretch.clicked.connect(self.stretch_histogram)
        group1_layout.addWidget(btn_hist_stretch)
        group1.setLayout(group1_layout)
        controls_layout.addWidget(group1)

        # Group 2: Convolution operations
        group2 = QGroupBox("b. Операции свёртки")
        group2_layout = QVBoxLayout()
        btn_blur = QPushButton('Размытие (Гаусс 5x5)')
        btn_blur.clicked.connect(self.gaussian_blur)
        group2_layout.addWidget(btn_blur)
        btn_sharpen = QPushButton('Повышение чёткости (Лапласиан)')
        btn_sharpen.clicked.connect(self.sharpen_laplacian)
        group2_layout.addWidget(btn_sharpen)
        btn_sobel = QPushButton('Выделение краёв (Собель)')
        btn_sobel.clicked.connect(self.sobel_edges)
        group2_layout.addWidget(btn_sobel)
        group2.setLayout(group2_layout)
        controls_layout.addWidget(group2)

        # Group 3: Geometric transformations
        group3 = QGroupBox("c. Геометрические преобразования")
        group3_layout = QVBoxLayout()
        btn_shift = QPushButton('Циклический сдвиг (50, 50)')
        btn_shift.clicked.connect(self.cyclic_shift)
        group3_layout.addWidget(btn_shift)
        btn_rotate = QPushButton('Поворот (45°, центр)')
        btn_rotate.clicked.connect(self.rotate_image)
        group3_layout.addWidget(btn_rotate)
        group3.setLayout(group3_layout)
        controls_layout.addWidget(group3)

        controls_layout.addStretch()

        # Layout for displaying images
        images_layout = QGridLayout()
        self.image_label_orig = QLabel('Исходное изображение')
        self.image_label_orig.setAlignment(Qt.AlignCenter)
        self.image_label_orig.setStyleSheet("border: 1px solid gray;")
        self.image_label_proc = QLabel('Обработанное изображение')
        self.image_label_proc.setAlignment(Qt.AlignCenter)
        self.image_label_proc.setStyleSheet("border: 1px solid gray;")
        images_layout.addWidget(self.image_label_orig, 0, 0)
        images_layout.addWidget(self.image_label_proc, 0, 1)

        # Add widgets to main layout
        main_layout.addWidget(controls_widget)
        main_layout.addLayout(images_layout)
        self.setLayout(main_layout)

    def load_image(self):
        # Open file dialog to select image
        file_path, _ = QFileDialog.getOpenFileName(self, 'Открыть изображение', '', 'Image Files (*.png *.jpg *.bmp)')
        if file_path:
            self.original_image = cv2.imread(file_path)
            self.grayscale_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
            self.display_image(self.original_image, 'original')
            self.image_label_proc.clear()
            self.image_label_proc.setText('Обработанное изображение')

    def convert_to_gray_avg(self):
        # Convert image to grayscale using average method
        if self.original_image is not None:
            b, g, r = cv2.split(self.original_image)
            avg_gray = ((b.astype(float) + g.astype(float) + r.astype(float)) / 3).astype(np.uint8)
            self.processed_image = avg_gray
            self.display_image(self.processed_image, 'processed')

    def convert_to_gray_standard(self):
        # Convert image to grayscale using OpenCV standard method
        if self.original_image is not None:
            self.processed_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
            self.display_image(self.processed_image, 'processed')

    def binarize_threshold(self):
        # Binarize grayscale image with fixed threshold
        if self.grayscale_image is not None:
            _, self.processed_image = cv2.threshold(self.grayscale_image, 127, 255, cv2.THRESH_BINARY)
            self.display_image(self.processed_image, 'processed')

    def binarize_otsu(self):
        # Binarize grayscale image using Otsu's method
        if self.grayscale_image is not None:
            _, self.processed_image = cv2.threshold(self.grayscale_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            self.display_image(self.processed_image, 'processed')

    def equalize_histogram(self):
        # Apply histogram equalization to grayscale image
        if self.grayscale_image is not None:
            self.processed_image = cv2.equalizeHist(self.grayscale_image)
            self.display_image(self.processed_image, 'processed')

    def stretch_histogram(self):
        # Stretch histogram of grayscale image
        if self.grayscale_image is not None:
            self.processed_image = cv2.normalize(self.grayscale_image, None, 0, 255, cv2.NORM_MINMAX)
            self.display_image(self.processed_image, 'processed')

    def gaussian_blur(self):
        # Apply Gaussian blur to grayscale image
        if self.grayscale_image is not None:
            self.processed_image = cv2.GaussianBlur(self.grayscale_image, (5, 5), 0)
            self.display_image(self.processed_image, 'processed')

    def sharpen_laplacian(self):
        # Sharpen grayscale image using Laplacian filter
        if self.grayscale_image is not None:
            laplacian = cv2.Laplacian(self.grayscale_image, cv2.CV_64F)
            laplacian_abs = np.uint8(np.absolute(laplacian))
            self.processed_image = cv2.add(self.grayscale_image, laplacian_abs)
            self.display_image(self.processed_image, 'processed')

    def sobel_edges(self):
        # Detect edges using Sobel operator
        if self.grayscale_image is not None:
            sobelx = cv2.Sobel(self.grayscale_image, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(self.grayscale_image, cv2.CV_64F, 0, 1, ksize=3)
            sobel = np.hypot(sobelx, sobely)
            sobel = np.uint8(np.clip(sobel, 0, 255))
            self.processed_image = sobel
            self.display_image(self.processed_image, 'processed')

    def cyclic_shift(self):
        # Apply cyclic shift to original image
        if self.original_image is not None:
            shifted = np.roll(self.original_image, shift=50, axis=(0, 1))
            self.processed_image = shifted
            self.display_image(self.processed_image, 'processed')

    def rotate_image(self):
        # Rotate original image by 45 degrees around its center
        if self.original_image is not None:
            (h, w) = self.original_image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, 45, 1.0)
            rotated = cv2.warpAffine(self.original_image, M, (w, h))
            self.processed_image = rotated
            self.display_image(self.processed_image, 'processed')

    def display_image(self, image, label_type):
        # Display image in the appropriate QLabel
        if len(image.shape) == 2:
            qimg = QImage(image.data, image.shape[1], image.shape[0], image.strides[0], QImage.Format_Grayscale8)
        else:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            qimg = QImage(rgb_image.data, rgb_image.shape[1], rgb_image.shape[0], rgb_image.strides[0], QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        if label_type == 'original':
            self.image_label_orig.setPixmap(pixmap.scaled(self.image_label_orig.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            self.image_label_proc.setPixmap(pixmap.scaled(self.image_label_proc.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

if __name__ == '__main__':
    # Run the application
    app = QApplication(sys.argv)
    window = ImageProcessorApp()
    window.show()
    sys.exit(app.exec_())
