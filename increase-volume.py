#!/usr/bin/env python3
"""
Audio Volume Processor

A comprehensive audio processing tool that can increase volume of audio files
with various safety features and options.

Features:
- Supports multiple audio formats (WAV, FLAC, OGG, etc.)
- Configurable volume adjustment
- Parallel processing for better performance
- Recursive directory processing
- Backup and safety features
- Detailed logging
- Dry-run mode for testing

Usage Examples:
--------------
1. Basic usage (default 20% volume increase):
    python increase-volume.py

2. Specify volume increase:
    python increase-volume.py --volume-factor 1.5

3. Process specific folder with backups:
    python increase-volume.py --folder /path/to/audio --backup-folder /path/to/backup

4. Recursive processing with multiple formats:
    python increase-volume.py --recursive --formats wav,flac,ogg

5. Dry run with logging:
    python increase-volume.py --dry-run --log-file process.log

6. Process recent files in parallel:
    python increase-volume.py --recent --workers 4

For more options, use: python increase-volume.py --help
"""

import os
import sys
import hashlib
import logging
import argparse
import soundfile as sf
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Dict, Union, Set
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import shutil

# Custom Exceptions
class AudioProcessingError(Exception):
    """Base exception for audio processing errors."""
    pass

class ValidationError(AudioProcessingError):
    """Raised when input validation fails."""
    pass

class BackupError(AudioProcessingError):
    """Raised when backup operations fail."""
    pass

@dataclass
class ProcessingConfig:
    """Configuration for audio processing."""
    volume_factor: float
    input_folder: Path
    backup_folder: Optional[Path]
    formats: Set[str]
    recursive: bool
    dry_run: bool
    recent_only: bool
    workers: int
    log_file: Optional[str]

    def __post_init__(self):
        if self.volume_factor <= 0:
            raise ValidationError("Volume factor must be positive")
        if not self.input_folder.exists():
            raise ValidationError(f"Input folder does not exist: {self.input_folder}")
        if self.backup_folder and not self.backup_folder.exists():
            os.makedirs(self.backup_folder)

class Logger:
    """Handles all logging operations."""
    def __init__(self, log_file: Optional[str] = None):
        self.logger = logging.getLogger('AudioProcessor')
        self.logger.setLevel(logging.INFO)
        
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # File handler if specified
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

    def info(self, msg: str) -> None:
        self.logger.info(msg)

    def error(self, msg: str) -> None:
        self.logger.error(msg)

    def warning(self, msg: str) -> None:
        self.logger.warning(msg)

class AudioProcessor:
    """Handles audio file processing operations."""
    def __init__(self, config: ProcessingConfig, logger: Logger):
        self.config = config
        self.logger = logger

    def calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA-256 checksum of a file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def process_file(self, file_path: Path) -> bool:
        """Process a single audio file."""
        try:
            if self.config.dry_run:
                self.logger.info(f"Would process: {file_path}")
                return True

            # Create backup if needed
            if self.config.backup_folder:
                backup_path = self.config.backup_folder / file_path.name
                original_checksum = self.calculate_checksum(file_path)
                shutil.copy2(file_path, backup_path)
                backup_checksum = self.calculate_checksum(backup_path)
                
                if original_checksum != backup_checksum:
                    raise BackupError(f"Backup verification failed for {file_path}")

            # Process audio
            data, samplerate = sf.read(file_path)
            volume_increased = data * self.config.volume_factor

            # Normalize if needed
            max_amplitude = np.abs(volume_increased).max()
            if max_amplitude > 1.0:
                volume_increased = volume_increased / max_amplitude

            # Write processed audio
            sf.write(file_path, volume_increased, samplerate)
            self.logger.info(f"Processed: {file_path}")
            return True

        except Exception as e:
            self.logger.error(f"Error processing {file_path}: {str(e)}")
            return False

class FileHandler:
    """Handles file system operations."""
    def __init__(self, config: ProcessingConfig, logger: Logger):
        self.config = config
        self.logger = logger

    def get_files_to_process(self) -> List[Path]:
        """Get list of audio files to process based on configuration."""
        files = []
        
        def should_process(path: Path) -> bool:
            if not path.suffix[1:].lower() in self.config.formats:
                return False
            if self.config.recent_only:
                file_age = datetime.now() - datetime.fromtimestamp(path.stat().st_ctime)
                if file_age > timedelta(hours=24):
                    return False
            return True

        if self.config.recursive:
            for path in self.config.input_folder.rglob("*"):
                if path.is_file() and should_process(path):
                    files.append(path)
        else:
            for path in self.config.input_folder.iterdir():
                if path.is_file() and should_process(path):
                    files.append(path)

        return files

def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description='Process audio files to adjust volume with various options and safety features'
    )
    
    parser.add_argument('--folder', type=Path,
                    default=Path('/Users/brian/Documents/VSTLive/projects/audio/Gig'),
                    help='Folder path containing audio files')
    parser.add_argument('--backup-folder', type=Path,
                    help='Folder path to store backup copies')
    parser.add_argument('--volume-factor', type=float, default=1.2,
                    help='Volume increase factor (e.g., 1.2 for 20% increase)')
    parser.add_argument('--formats', type=str, default='wav',
                    help='Comma-separated list of audio formats to process')
    parser.add_argument('--recursive', action='store_true',
                    help='Process files in subdirectories')
    parser.add_argument('--recent', action='store_true',
                    help='Only process files from last 24 hours')
    parser.add_argument('--workers', type=int, default=1,
                    help='Number of worker processes for parallel processing')
    parser.add_argument('--dry-run', action='store_true',
                    help='Show what would be done without making changes')
    parser.add_argument('--log-file', type=str,
                    help='Path to log file for detailed logging')
    
    args = parser.parse_args()


try:
    # Initialize configuration
    config = ProcessingConfig(
        volume_factor=args.volume_factor,
        input_folder=args.folder,
        backup_folder=args.backup_folder,
        formats=set(args.formats.split(',')),
        recursive=args.recursive,
        dry_run=args.dry_run,
        recent_only=args.recent,
        workers=args.workers,
        log_file=args.log_file
    )
except ValidationError as e:
    print(f"Error: {str(e)}")
    sys.exit(1)

# Initialize logger
logger = Logger(log_file=config.log_file)

# Initialize file handler
file_handler = FileHandler(config, logger)

# Initialize audio processor
audio_processor = AudioProcessor(config, logger)

# Get list of files to process
files_to_process = file_handler.get_files_to_process()

# Process files in parallel
with ProcessPoolExecutor(max_workers=config.workers) as executor:
    futures = []
    for file_path in files_to_process:
        futures.append(executor.submit(audio_processor.process_file, file_path))

    for future in tqdm(as_completed(futures), total=len(files_to_process)):
        if future.result() is False:
            sys.exit(1)