# Quick Start Guide - Analog Clock Time Recognition

## 🚀 Getting Started in 5 Minutes

### Step 1: Install Dependencies

```bash
cd /Users/daniil.anishchanka/PycharmProjects/Computer-vision
source venv/bin/activate
pip install -r requirements.txt
```

### Step 2: Run the Application

```bash
python lab3_app.py
```

### Step 3: Use the GUI

1. Click **"Загрузить изображение"** (Load Image)
2. Select a clock image (Clock_1.jpg, Clock_2.jpg, or church-clock.webp)
3. Choose detection method (Enhanced recommended)
4. Click **"Определить время"** (Detect Time)
5. View results!

## 📊 Quick Examples

### Example 1: Process Single Image (Python API)

```python
from lab3.clock_models import ClockDetectionModel, ClockTimeRecognitionModel
import cv2

# Load image
image = cv2.imread('lab3/Clock_1.jpg')

# Detect clock
detector = ClockDetectionModel()
detection = detector.detect_clock_face(image)

# Extract and recognize
clock_face = detector.extract_clock_face(image, detection)
recognizer = ClockTimeRecognitionModel()
hour, minute, confidence = recognizer.predict_time(clock_face)

print(f"Time: {hour:02d}:{minute:02d} (confidence: {confidence*100:.0f}%)")
```

### Example 2: Run Automated Tests

```bash
cd lab3
python test_images.py
```

### Example 3: Generate Training Data

```python
from lab3.synthetic_clock_generator import SyntheticClockGenerator

gen = SyntheticClockGenerator("my_dataset")
gen.generate_dataset(num_samples=100, clock_size=300)
```

## 🎯 What You Get

- ✅ **GUI Application** - User-friendly desktop interface
- ✅ **Clock Detection** - Automatically finds clocks in images
- ✅ **Time Reading** - Extracts hour and minute
- ✅ **Synthetic Data** - Generate unlimited training images
- ✅ **Multiple Methods** - Classic, Enhanced, and Hybrid approaches
- ✅ **Testing Suite** - Validate on test images
- ✅ **Documentation** - Complete guides and examples

## 📁 Key Files

| File | Purpose |
|------|---------|
| `lab3_app.py` | Main GUI application |
| `lab3/clock_models.py` | Detection & recognition models |
| `lab3/synthetic_clock_generator.py` | Training data generator |
| `lab3/test_images.py` | Automated testing |
| `lab3/example_usage.py` | Code examples |
| `lab3/README.md` | Full documentation |

## 🛠️ Troubleshooting

### Problem: GUI doesn't open
**Solution:** Make sure PyQt5 is installed:
```bash
pip install PyQt5
```

### Problem: Image not loading
**Solution:** Check supported formats: .jpg, .jpeg, .png, .webp, .bmp

### Problem: Clock not detected
**Solution:** Try different detection methods in the GUI settings

### Problem: Import errors
**Solution:** Ensure you're in the project root:
```bash
cd /Users/daniil.anishchanka/PycharmProjects/Computer-vision
source venv/bin/activate
```

## 📖 Next Steps

1. **Read Full Documentation:** See `lab3/README.md`
2. **Review Implementation:** See `lab3/IMPLEMENTATION_REPORT.md`
3. **Study Examples:** Run `lab3/example_usage.py`
4. **Generate Data:** Create training datasets
5. **Customize:** Modify detection parameters in code

## 💡 Tips

- Use **Enhanced** mode for best results
- Enable **preprocessing** for better detection
- Check **confidence** scores to validate results
- Test with different clock styles
- Generate synthetic data to understand the system

## 🎓 Learning Resources

The implementation demonstrates:
- Hough Transform (circles and lines)
- Edge detection (Canny, Adaptive Thresholding)
- Geometric filtering
- Angle calculations
- Image preprocessing (CLAHE, denoising)
- PyQt5 GUI development
- Deep learning architecture design

## 📧 Support

For issues or questions:
1. Check documentation in `lab3/` directory
2. Review test output in `lab3/test_images.py`
3. Examine example code in `lab3/example_usage.py`

---

**Status:** ✅ Fully Functional

**Last Updated:** October 2025

**Version:** 1.0.0

