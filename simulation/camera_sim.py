import cv2
import numpy as np
import os
from pathlib import Path
import random
from . import config

class CameraSimulator:
    """Simulates 3 cameras capturing images at different heights"""
    
    def __init__(self, mode="crop"):
        self.mode = mode
        self.background_image = None
        self.dummy_images = []
        self._load_resources()
    
    def _load_resources(self):
        """Load background image or dataset images"""
        # Load real dataset images
        self.dataset_images = self._load_dataset_images()
        
        if self.mode == "crop":
            bg_path = config.BACKGROUND_IMAGE
            if os.path.exists(bg_path):
                self.background_image = cv2.imread(bg_path)
            else:
                # Generate synthetic background
                self.background_image = self._generate_background()
        else:
            # Load dummy images
            dummy_path = Path(config.DUMMY_IMAGES_PATH)
            if dummy_path.exists():
                self.dummy_images = list(dummy_path.glob("*.jpg")) + list(dummy_path.glob("*.png"))
            
            if not self.dummy_images:
                # Generate dummy images
                self._generate_dummy_images()
    
    def _load_dataset_images(self):
        """Load images from dataset directory"""
        dataset_images = []
        images_dir = Path('images')
        
        if images_dir.exists():
            for component_dir in images_dir.iterdir():
                if component_dir.is_dir():
                    imgs = list(component_dir.glob('*.jpg')) + list(component_dir.glob('*.png'))
                    dataset_images.extend(imgs)
        
        print(f"Loaded {len(dataset_images)} images from dataset")
        return dataset_images
    
    def _generate_background(self):
        """Generate a synthetic tall background image"""
        height = 3000  # Tall image for vertical scanning
        width = config.IMAGE_WIDTH
        
        # Create gradient background
        bg = np.zeros((height, width, 3), dtype=np.uint8)
        for i in range(height):
            color_val = int(100 + (i / height) * 100)
            bg[i, :] = [color_val, color_val - 20, color_val - 40]
        
        # Add some texture/patterns
        for _ in range(50):
            x = random.randint(0, width - 100)
            y = random.randint(0, height - 100)
            w = random.randint(50, 200)
            h = random.randint(50, 200)
            color = (random.randint(80, 120), random.randint(80, 120), random.randint(80, 120))
            cv2.rectangle(bg, (x, y), (x + w, y + h), color, -1)
        
        return bg
    
    def _generate_dummy_images(self):
        """Generate dummy images for random mode"""
        Path(config.DUMMY_IMAGES_PATH).mkdir(parents=True, exist_ok=True)
        
        for i in range(20):
            img = np.random.randint(50, 200, (config.IMAGE_HEIGHT, config.IMAGE_WIDTH, 3), dtype=np.uint8)
            # Add some patterns
            cv2.circle(img, (random.randint(100, config.IMAGE_WIDTH - 100), 
                           random.randint(100, config.IMAGE_HEIGHT - 100)), 
                      random.randint(30, 100), 
                      (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)), -1)
            
            path = os.path.join(config.DUMMY_IMAGES_PATH, f"dummy_{i:03d}.jpg")
            cv2.imwrite(path, img)
            self.dummy_images.append(path)
    
    def capture_at_height(self, height, amr_x=0):
        """
        Capture 3 camera images at specified height using dataset images
        
        Args:
            height: Current height in meters
            amr_x: AMR horizontal position in meters
            
        Returns:
            tuple: (left_image, center_image, right_image)
        """
        # Always use dataset images if available
        if self.dataset_images and len(self.dataset_images) >= 3:
            return self._capture_from_dataset(height, amr_x)
        elif self.mode == "crop":
            return self._capture_crop(height, amr_x)
        else:
            return self._capture_random(height)
    
    def _capture_crop(self, height, amr_x):
        """Crop from background image"""
        if self.background_image is None:
            self.background_image = self._generate_background()
        
        bg_height, bg_width = self.background_image.shape[:2]
        
        # Calculate vertical position based on height
        # Map height (0.3 to 2.3m) to image rows
        height_ratio = (height - 0.3) / (2.3 - 0.3)
        start_y = int(height_ratio * (bg_height - config.IMAGE_HEIGHT))
        start_y = max(0, min(start_y, bg_height - config.IMAGE_HEIGHT))
        
        # Crop region
        crop = self.background_image[start_y:start_y + config.IMAGE_HEIGHT, :bg_width]
        
        # Ensure crop is correct size
        if crop.shape[0] < config.IMAGE_HEIGHT:
            crop = cv2.resize(crop, (bg_width, config.IMAGE_HEIGHT))
        
        # Split into 3 cameras
        cam_width = crop.shape[1] // 3
        left = crop[:, :cam_width]
        center = crop[:, cam_width:2*cam_width]
        right = crop[:, 2*cam_width:]
        
        # Resize to standard size
        left = cv2.resize(left, (config.IMAGE_WIDTH // 3, config.IMAGE_HEIGHT))
        center = cv2.resize(center, (config.IMAGE_WIDTH // 3, config.IMAGE_HEIGHT))
        right = cv2.resize(right, (config.IMAGE_WIDTH // 3, config.IMAGE_HEIGHT))
        
        return left, center, right
    
    def _capture_random(self, height):
        """Select random images"""
        if not self.dummy_images:
            self._generate_dummy_images()
        
        selected = random.sample(self.dummy_images, min(3, len(self.dummy_images)))
        
        images = []
        for img_path in selected:
            if isinstance(img_path, str):
                img = cv2.imread(img_path)
            else:
                img = cv2.imread(str(img_path))
            
            if img is not None:
                img = cv2.resize(img, (config.IMAGE_WIDTH // 3, config.IMAGE_HEIGHT))
                images.append(img)
        
        # Ensure we have 3 images
        while len(images) < 3:
            images.append(images[0].copy() if images else np.zeros((config.IMAGE_HEIGHT, config.IMAGE_WIDTH // 3, 3), dtype=np.uint8))
        
        return images[0], images[1], images[2]
    
    def _capture_from_dataset(self, height, amr_x):
        """Capture images from real dataset - returns full resolution images for AI detection"""
        if not self.dataset_images:
            # Fallback to crop mode
            return self._capture_crop(height, amr_x)
        
        # Select 3 random images from dataset
        selected = random.sample(self.dataset_images, min(3, len(self.dataset_images)))
        
        images = []
        for img_path in selected:
            img = cv2.imread(str(img_path))
            if img is not None:
                # Keep original resolution for better AI detection
                # Resize to standard size if needed
                h, w = img.shape[:2]
                if h != config.IMAGE_HEIGHT or w != config.IMAGE_WIDTH:
                    img = cv2.resize(img, (config.IMAGE_WIDTH, config.IMAGE_HEIGHT))
                images.append(img)
        
        # Ensure we have 3 images
        while len(images) < 3:
            if images:
                images.append(images[0].copy())
            else:
                # Generate blank image
                images.append(np.zeros((config.IMAGE_HEIGHT, config.IMAGE_WIDTH, 3), dtype=np.uint8))
        
        return images[0], images[1], images[2]
    
    def save_images(self, left, center, right, height, scan_id):
        """Save captured images"""
        Path(config.RESULTS_PATH).mkdir(parents=True, exist_ok=True)
        
        paths = {}
        for name, img in [("left", left), ("center", center), ("right", right)]:
            filename = f"scan_{scan_id}_h{height:.1f}m_{name}.jpg"
            filepath = os.path.join(config.RESULTS_PATH, filename)
            cv2.imwrite(filepath, img)
            paths[name] = filepath
        
        return paths
