import os
import logging
import tempfile
from datetime import datetime
from typing import Dict, List

def setup_logging():
    """Set up logging configuration"""
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f"{log_dir}/medical_assistant.log"),
            logging.StreamHandler()
        ]
    )

def create_temp_file(data: bytes, suffix: str = ".wav") -> str:
    """Create temporary file from bytes data"""
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    temp_file.write(data)
    temp_file.close()
    return temp_file.name

def cleanup_temp_file(file_path: str):
    """Clean up temporary file"""
    try:
        if os.path.exists(file_path):
            os.unlink(file_path)
    except Exception as e:
        logging.error(f"Failed to cleanup temp file: {e}")

def format_timestamp(seconds: float) -> str:
    """Format seconds to MM:SS format"""
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    return f"{minutes:02d}:{seconds:02d}"

def validate_api_key(api_key: str) -> bool:
    """Validate AssemblyAI API key format"""
    return bool(api_key and len(api_key) > 20 and api_key.startswith(('aa-', 'assemblyai-')))