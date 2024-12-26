from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Database Configuration
DATABASE_CONFIG = {
    "host": os.getenv("DB_HOST"),
    "port": int(os.getenv("DB_PORT", 5432)),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "database": os.getenv("DB_NAME"),
}

print("DB_HOST:", os.getenv("DB_HOST"))
print("DB_PORT:", os.getenv("DB_PORT"))
print("DB_USER:", os.getenv("DB_USER"))
print("DB_PASSWORD:", os.getenv("DB_PASSWORD"))
print("DB_NAME:", os.getenv("DB_NAME"))



# Storage Paths for Images
STORAGE_PATHS = {
    "original": "./data/original/",  # Path to store original images
    "enhanced": "./data/enhanced/",  # Path to store enhanced images
    "logs": "./data/logs/"           # Path for processing logs
}

# Image Quality Standards
IMAGE_QUALITY_THRESHOLDS = {
    "min_resolution": (1024, 1024),  # Minimum resolution for images
    "aspect_ratio": (1, 1),          # Aspect ratio (default: square images)
    "sharpness_threshold": 100.0,    # Laplacian variance threshold for clarity
    "allowed_background_colors": [(255, 255, 255)],  # Allowed background colors (e.g., white)
}

# Batch Processing Configuration
BATCH_PROCESSING = {
    "batch_size": 5,              # Number of images to process in one batch
    "parallel_workers": 4,           # Number of parallel workers for multiprocessing
}

# AWS S3 Configuration (Optional Cloud Storage)
AWS_S3_CONFIG = {
    "enabled": bool(os.getenv("AWS_S3_ENABLED", False)),  # Enable or disable S3 integration
    "bucket_name": os.getenv("AWS_S3_BUCKET", "image-quality-tool"),
    "region": os.getenv("AWS_S3_REGION", "us-east-1"),
    "access_key": os.getenv("AWS_ACCESS_KEY_ID", ""),
    "secret_key": os.getenv("AWS_SECRET_ACCESS_KEY", ""),
}

# Logging Configuration
LOGGING_CONFIG = {
    "level": os.getenv("LOG_LEVEL", "INFO"),  # Logging level (e.g., DEBUG, INFO, WARNING)
    "log_file": "./data/logs/app.log",       # Path to the log file
}

# API Configuration
API_CONFIG = {
    "host": os.getenv("API_HOST", "0.0.0.0"),
    "port": int(os.getenv("API_PORT", 5000)),
    "debug": bool(os.getenv("API_DEBUG", True)),
}

# Ensure required directories exist
REQUIRED_DIRECTORIES = [STORAGE_PATHS["original"], STORAGE_PATHS["enhanced"], STORAGE_PATHS["logs"]]
for directory in REQUIRED_DIRECTORIES:
    os.makedirs(directory, exist_ok=True)

# Additional Constants
ALLOWED_IMAGE_FORMATS = ["jpg", "jpeg", "png"]  # Supported image formats
ENHANCEMENT_MODEL_PATH = "../models/esrgan/weights/"  # Path to ESRGAN model weights
