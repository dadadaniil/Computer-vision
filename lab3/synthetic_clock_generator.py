"""
Synthetic Clock Data Generator
Generates diverse analog clock images for training deep learning models
"""
import os
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import json
from typing import Tuple, List, Dict


class SyntheticClockGenerator:
    """Generate synthetic analog clock images with annotations"""
    
    def __init__(self, output_dir: str = "synthetic_clocks"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "annotations"), exist_ok=True)
        
    def generate_clock_face(self, size: int, style: str = "modern") -> Image.Image:
        """Generate a clock face with various styles"""
        img = Image.new('RGBA', (size, size), (255, 255, 255, 0))
        draw = ImageDraw.Draw(img)
        center = size // 2
        radius = size // 2 - 10
        
        # Random colors
        face_color = self._random_color()
        border_color = self._random_color()
        number_color = self._random_color()
        
        # Draw clock face
        draw.ellipse([center - radius, center - radius, 
                     center + radius, center + radius], 
                     fill=face_color, outline=border_color, width=3)
        
        # Draw hour markers
        if random.random() > 0.3:  # 70% chance of having markers
            if style == "roman":
                self._draw_roman_numerals(draw, center, radius, number_color, size)
            elif style == "arabic":
                self._draw_arabic_numerals(draw, center, radius, number_color, size)
            else:
                self._draw_tick_marks(draw, center, radius, border_color)
        
        return img
    
    def _random_color(self) -> Tuple[int, int, int]:
        """Generate random color"""
        return (random.randint(50, 255), 
                random.randint(50, 255), 
                random.randint(50, 255))
    
    def _draw_roman_numerals(self, draw, center, radius, color, size):
        """Draw Roman numerals"""
        numerals = ['XII', 'I', 'II', 'III', 'IV', 'V', 'VI', 
                   'VII', 'VIII', 'IX', 'X', 'XI']
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size // 15)
        except:
            font = ImageFont.load_default()
            
        for i, numeral in enumerate(numerals):
            angle = np.radians(i * 30 - 90)
            x = center + int((radius * 0.75) * np.cos(angle))
            y = center + int((radius * 0.75) * np.sin(angle))
            bbox = draw.textbbox((x, y), numeral, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            draw.text((x - text_width // 2, y - text_height // 2), 
                     numeral, fill=color, font=font)
    
    def _draw_arabic_numerals(self, draw, center, radius, color, size):
        """Draw Arabic numerals"""
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size // 12)
        except:
            font = ImageFont.load_default()
            
        for i in range(12):
            number = str(i if i == 0 else i)
            if i == 0:
                number = "12"
            angle = np.radians(i * 30 - 90)
            x = center + int((radius * 0.75) * np.cos(angle))
            y = center + int((radius * 0.75) * np.sin(angle))
            bbox = draw.textbbox((x, y), number, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            draw.text((x - text_width // 2, y - text_height // 2), 
                     number, fill=color, font=font)
    
    def _draw_tick_marks(self, draw, center, radius, color):
        """Draw tick marks"""
        for i in range(60):
            angle = np.radians(i * 6 - 90)
            if i % 5 == 0:  # Hour marks
                start_r = radius * 0.85
                end_r = radius * 0.95
                width = 3
            else:  # Minute marks
                start_r = radius * 0.90
                end_r = radius * 0.95
                width = 1
                
            x1 = center + int(start_r * np.cos(angle))
            y1 = center + int(start_r * np.sin(angle))
            x2 = center + int(end_r * np.cos(angle))
            y2 = center + int(end_r * np.sin(angle))
            draw.line([x1, y1, x2, y2], fill=color, width=width)
    
    def draw_clock_hands(self, img: Image.Image, hour: int, minute: int) -> Tuple[Image.Image, List]:
        """Draw hour and minute hands"""
        draw = ImageDraw.Draw(img)
        size = img.size[0]
        center = size // 2
        radius = size // 2 - 10
        
        hand_color = self._random_color()
        
        # Calculate angles (0 degrees is 12 o'clock, clockwise)
        minute_angle = np.radians(minute * 6 - 90)
        hour_angle = np.radians(((hour % 12) * 30 + minute * 0.5) - 90)
        
        # Draw minute hand (longer)
        minute_length = radius * random.uniform(0.75, 0.85)
        minute_width = random.randint(3, 6)
        mx = center + int(minute_length * np.cos(minute_angle))
        my = center + int(minute_length * np.sin(minute_angle))
        draw.line([center, center, mx, my], fill=hand_color, width=minute_width)
        
        # Draw hour hand (shorter)
        hour_length = radius * random.uniform(0.45, 0.55)
        hour_width = random.randint(4, 8)
        hx = center + int(hour_length * np.cos(hour_angle))
        hy = center + int(hour_length * np.sin(hour_angle))
        draw.line([center, center, hx, hy], fill=hand_color, width=hour_width)
        
        # Draw center dot
        dot_radius = random.randint(5, 10)
        draw.ellipse([center - dot_radius, center - dot_radius,
                     center + dot_radius, center + dot_radius],
                     fill=hand_color)
        
        # Return hand positions for annotation
        hands = [
            {"type": "hour", "angle": float(np.degrees(hour_angle + np.pi/2))},
            {"type": "minute", "angle": float(np.degrees(minute_angle + np.pi/2))}
        ]
        
        return img, hands
    
    def add_background(self, clock_img: Image.Image, bg_size: Tuple[int, int]) -> Tuple[Image.Image, Tuple[int, int, int, int]]:
        """Add background with distractors and place clock"""
        bg = Image.new('RGB', bg_size, self._random_color())
        draw = ImageDraw.Draw(bg)
        
        # Add random lines as distractors
        num_lines = random.randint(5, 20)
        for _ in range(num_lines):
            x1, y1 = random.randint(0, bg_size[0]), random.randint(0, bg_size[1])
            x2, y2 = random.randint(0, bg_size[0]), random.randint(0, bg_size[1])
            color = self._random_color()
            width = random.randint(1, 5)
            draw.line([x1, y1, x2, y2], fill=color, width=width)
        
        # Add random circles as distractors
        num_circles = random.randint(2, 8)
        for _ in range(num_circles):
            cx = random.randint(0, bg_size[0])
            cy = random.randint(0, bg_size[1])
            r = random.randint(20, 100)
            color = self._random_color()
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], 
                        outline=color, width=random.randint(2, 5))
        
        # Place clock on background
        clock_size = clock_img.size[0]
        max_x = bg_size[0] - clock_size
        max_y = bg_size[1] - clock_size
        x_offset = random.randint(0, max_x) if max_x > 0 else 0
        y_offset = random.randint(0, max_y) if max_y > 0 else 0
        
        bg.paste(clock_img, (x_offset, y_offset), clock_img)
        
        # Bounding box (x, y, width, height)
        bbox = (x_offset, y_offset, clock_size, clock_size)
        
        return bg, bbox
    
    def apply_augmentations(self, img: Image.Image) -> Image.Image:
        """Apply random augmentations"""
        # Random rotation
        if random.random() > 0.5:
            angle = random.uniform(-15, 15)
            img = img.rotate(angle, expand=False, fillcolor=(128, 128, 128))
        
        # Random blur
        if random.random() > 0.5:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 2.0)))
        
        # Random brightness
        if random.random() > 0.5:
            enhancer = Image.Enhance.Brightness(img)
            img = enhancer.enhance(random.uniform(0.7, 1.3))
        
        return img
    
    def generate_dataset(self, num_samples: int = 1000, 
                        clock_size: int = 300, 
                        image_size: Tuple[int, int] = (640, 640)):
        """Generate complete dataset with annotations"""
        annotations = []
        
        for i in range(num_samples):
            # Random time
            hour = random.randint(0, 11)
            minute = random.randint(0, 59)
            
            # Random clock style
            style = random.choice(["modern", "roman", "arabic"])
            
            # Generate clock
            clock = self.generate_clock_face(clock_size, style)
            clock, hands = self.draw_clock_hands(clock, hour, minute)
            
            # Add background with distractors
            final_img, bbox = self.add_background(clock, image_size)
            
            # Apply augmentations
            final_img = self.apply_augmentations(final_img)
            
            # Save image
            img_filename = f"clock_{i:05d}.png"
            img_path = os.path.join(self.output_dir, "images", img_filename)
            final_img.save(img_path)
            
            # Create annotation
            annotation = {
                "filename": img_filename,
                "hour": hour,
                "minute": minute,
                "time": f"{hour:02d}:{minute:02d}",
                "bbox": bbox,
                "hands": hands,
                "image_size": image_size
            }
            annotations.append(annotation)
            
            if (i + 1) % 100 == 0:
                print(f"Generated {i + 1}/{num_samples} images")
        
        # Save annotations
        annotations_path = os.path.join(self.output_dir, "annotations.json")
        with open(annotations_path, 'w') as f:
            json.dump(annotations, f, indent=2)
        
        print(f"\nDataset generation complete!")
        print(f"Images saved to: {os.path.join(self.output_dir, 'images')}")
        print(f"Annotations saved to: {annotations_path}")


if __name__ == "__main__":
    # Example usage
    generator = SyntheticClockGenerator("synthetic_clocks_data")
    generator.generate_dataset(num_samples=100, clock_size=300, image_size=(640, 640))

