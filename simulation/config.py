# Configuration for AMR Scanning Simulation

# Height levels for vertical scanning (meters)
HEIGHT_LEVELS = [0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.1, 2.3]

# AMR horizontal positions (meters)
AMR_POSITIONS = [0, 1, 2]

# 4-Side scanning configuration
SIDES = {
    1: {"name": "Front", "position": (0, 0, 3), "rotation": 0},
    2: {"name": "Right", "position": (3, 0, 0), "rotation": -90},
    3: {"name": "Back", "position": (0, 0, -3), "rotation": 180},
    4: {"name": "Left", "position": (-3, 0, 0), "rotation": 90}
}

# System specifications
ASSEMBLY_HEIGHT = 2.3  # meters
CAMERA_REACH_MIN = 1.7  # meters
CAMERA_REACH_MAX = 1.8  # meters

# Timing (seconds)
MOVEMENT_DELAY = 0.5
STABILIZATION_DELAY = 0.3
CAPTURE_DELAY = 0.2

# Camera simulation
NUM_CAMERAS = 3
IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1080

# Paths
DUMMY_IMAGES_PATH = "data/dummy_images"
RESULTS_PATH = "data/results"
BACKGROUND_IMAGE = "data/background.jpg"

# Detection parameters
CONFIDENCE_THRESHOLD = 0.5
DEFECT_TYPES = ["crack", "corrosion", "deformation", "missing_part"]

# Simulation modes
CAMERA_MODE = "crop"  # "crop" or "random"
