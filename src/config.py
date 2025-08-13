import os
from dotenv import load_dotenv

load_dotenv()

# API Configuration
ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY")

# File paths
TEMP_DIR = "temp"
DATA_DIR = "data"
LOGS_DIR = "logs"

# Supported audio formats
SUPPORTED_FORMATS = ['wav', 'mp3', 'm4a', 'flac', 'ogg', 'aac']

# Medical entity categories
ENTITY_CATEGORIES = {
    'phi': {'color': 'red', 'label': 'Personal Health Information'},
    'medical_condition': {'color': 'lightgreen', 'label': 'Medical Conditions'},
    'anatomy': {'color': 'lightblue', 'label': 'Anatomy'},
    'medication': {'color': 'yellow', 'label': 'Medications'},
    'procedure': {'color': 'lightcyan', 'label': 'Procedures & Tests'}
}