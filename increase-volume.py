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
import sys
import os
import logging
import hashlib
import argparse
from pathlib import Path
from typing import List, Optional, Dict, Union, Set
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import soundfile as sf
from tqdm import tqdm
import shutil
from dataclasses import dataclass

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
    """Audio processing configuration.

    Attributes:
        volume_factor: Factor to multiply audio volume by (must be > 0)
        input_folder: Path to folder containing audio files
        backup_folder: Optional path for backup copies
        formats: Set of supported audio formats
        recursive: Whether to process subdirectories
        dry_run: Whether to perform dry run without changes
        recent_only: Whether to only process recent files (<24h)
        workers: Number of parallel workers
        log_file: Optional path for log file
    """
    volume_factor: float
    input_folder: Path
    backup_folder: Optional[Path]
    formats: Set[str]
    recursive: bool
    dry_run: bool
    recent_only: bool
    workers: int
    log_file: Optional[str]
    volume_increase: Optional[float] = None

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
        
        # Remove any existing handlers to avoid duplicates
        self.logger.handlers = []
        
        # Create formatter for consistent output
        formatter = logging.Formatter('%(message)s')  # Simplified format for clearer output

        # Always add console handler for immediate feedback
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(logging.INFO)
        self.logger.addHandler(console_handler)
        
        # Add file handler if specified
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        
        # Initial session message
        self.logger.info("=== Audio Processing Session Started ===")

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
        """Calculate SHA-256 checksum of a file.

        Args:
            file_path: Path to file to checksum
        
        Returns:
            String containing hexadecimal SHA-256 checksum
        """
        """Calculate SHA-256 checksum of a file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def process_file(self, file_path: Path) -> bool:
        """Process a single audio file with volume adjustment.

        Args:
            file_path: Path to audio file to process

        Returns:
            True if processing succeeded, False otherwise
        """
        try:
            self.logger.info(f"Starting to process: {file_path}")
            
            if self.config.dry_run:
                self.logger.info(f"[DRY RUN] Would process: {file_path}")
                self.logger.info(f"[DRY RUN] Would increase volume by {self.config.volume_factor}x")
                self.logger.info(f"[DRY RUN] Would create backup in: {self.config.backup_folder}")
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
    def __init__(self, config: ProcessingConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger

    def get_files_to_process(self) -> List[Path]:
        """Get list of audio files to process."""
        self.logger.debug(f"Searching for audio files in: {self.config.input_folder}")
        self.logger.debug(f"File formats to process: {self.config.formats}")
        
        if not self.config.input_folder.exists():
            self.logger.error(f"Input folder does not exist: {self.config.input_folder}")
            return []

        if not self.config.input_folder.is_dir():
            self.logger.error(f"Input path is not a directory: {self.config.input_folder}")
            return []

        files = []
        for format in self.config.formats:
            pattern = f"*.{format}"
            self.logger.debug(f"Searching with pattern: {pattern}")
            
            if self.config.recursive:
                matches = list(self.config.input_folder.rglob(pattern))
            else:
                matches = list(self.config.input_folder.glob(pattern))
            
            self.logger.debug(f"Found {len(matches)} files matching {pattern}")
            for match in matches:
                self.logger.debug(f"Found file: {match}")
            files.extend(matches)

        if self.config.recent_only:
            cutoff_time = datetime.now() - timedelta(hours=24)
            self.logger.debug(f"Filtering for files modified after: {cutoff_time}")
            files = [f for f in files if f.stat().st_mtime > cutoff_time.timestamp()]
            self.logger.debug(f"After time filter: {len(files)} files remaining")

        self.logger.debug(f"Total files to process: {len(files)}")
        return files

def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Process audio files to adjust volume with various options and safety features'
    )
    parser.add_argument('--folder', type=Path, required=True,
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
    return parser.parse_args()

def main() -> None:
    print("Starting audio processing script...")
    try:
        # Configure root logger first
        logging.basicConfig(level=logging.DEBUG)
        logger = logging.getLogger(__name__)
        
        # Parse arguments
        args = parse_arguments()
        logger.debug(f"Parsed arguments: {vars(args)}")
        
        # Create configuration
        config = ProcessingConfig(
            input_folder=Path(args.folder),
            backup_folder=args.backup_folder,
            formats=set(args.formats.lower().split(',')) if args.formats else {'wav'},
            dry_run=args.dry_run,
            recursive=args.recursive,
            recent_only=args.recent,
            workers=args.workers,
            log_file=args.log_file,
            volume_factor=args.volume_factor
        )
        
        logger.debug("Configuration created successfully")
        logger.debug(f"Input folder: {config.input_folder}")
        logger.debug(f"Formats: {config.formats}")
        
        # Configure file handler if log file specified
        if config.log_file:
            file_handler = logging.FileHandler(config.log_file)
            file_handler.setLevel(logging.DEBUG)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        # Initialize handlers
        file_handler = FileHandler(config, logger)
        files = file_handler.get_files_to_process()
        
        if not files:
            logger.info("No files found to process")
            return
            
        # Process files
        for file in files:
            if config.dry_run:
                logger.info(f"Would process: {file}")
                logger.info(f"Would increase volume by factor: {config.volume_factor}")
            else:
                logger.info(f"Processing: {file}")
            
    except Exception as e:
        print(f"ERROR: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
