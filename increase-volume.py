#!/usr/bin/env python3
"""
Audio Volume Processor

A comprehensive audio processing tool that can increase volume of audio files
with advanced dynamic processing, safety features and options.

Features:
- Intelligent volume processing with compression and limiting
- Dynamic range compression with configurable threshold and ratio
- Soft-knee limiting for peak protection
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
    volume_increase: Optional float tracking the actual volume increase achieved
                    after compression and limiting

The configuration includes settings for both the basic volume adjustment
and the advanced dynamic processing features (compression and limiting).
The volume_factor setting works in conjunction with dynamic processing
to achieve the desired volume increase while maintaining audio quality.
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
    """Handles all logging operations with support for multiple logging levels."""
    def __init__(self, log_file: Optional[str] = None):
        self.logger = logging.getLogger('AudioProcessor')
        self.logger.setLevel(logging.DEBUG)  # Set to DEBUG to allow all levels
        
        # Remove any existing handlers to avoid duplicates
        self.logger.handlers = []
        
        # Create formatter for consistent output
        formatter = logging.Formatter('%(levelname)s: %(message)s')

        # Always add console handler for immediate feedback
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(logging.INFO)  # Console shows INFO and above
        self.logger.addHandler(console_handler)
        
        # Add file handler if specified
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            file_handler.setLevel(logging.DEBUG)  # File logs everything
            self.logger.addHandler(file_handler)
        
        # Initial session message
        self.logger.info("=== Audio Processing Session Started ===")

    def debug(self, msg: str) -> None:
        """Log debug level message."""
        self.logger.debug(msg)

    def info(self, msg: str) -> None:
        """Log info level message."""
        self.logger.info(msg)

    def warning(self, msg: str) -> None:
        """Log warning level message."""
        self.logger.warning(msg)

    def error(self, msg: str) -> None:
        """Log error level message."""
        self.logger.error(msg)

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

            # Ensure data is 2D array (samples, channels)
            if len(data.shape) == 1:
                data = data.reshape(-1, 1)

            # Calculate original stats per channel
            # Calculate per-channel statistics
            original_max = np.max(np.abs(data), axis=0)
            original_rms = np.sqrt(np.mean(data**2, axis=0))

            # Convert to scalar values by taking mean across channels
            max_amp = float(np.mean(original_max))
            rms_amp = float(np.mean(original_rms))
            self.logger.info(f"Original max amplitude: {max_amp:.4f}, RMS: {rms_amp:.4f}")

            # Dynamic Range Compression Stage
            # Apply professional-grade dynamic range compression to control volume dynamics
            # while preserving audio quality. This stage involves:
            # 1. Converting to dB scale for proper compression processing
            # 2. Applying threshold-based compression with ratio
            # 3. Using attack/release envelope for smooth gain changes

            # Convert to dB scale while maintaining array dimensions
            eps = 1e-10  # Prevent log of zero
            db = 20 * np.log10(np.abs(data) + eps)

            # Compression parameters
            threshold_db = -12  # Compression threshold in dB
            ratio = 2.0  # Compression ratio (2:1)
            attack_ms = 5  # Attack time in milliseconds
            release_ms = 50  # Release time in milliseconds

            # Calculate gain reduction per channel
            gain_reduction_db = np.maximum(0, db - threshold_db) * (1 - 1/ratio)

            # Apply attack/release envelope
            attack_samples = int(samplerate * attack_ms / 1000)
            release_samples = int(samplerate * release_ms / 1000)

            # Create smoothing window
            window = np.concatenate([
                np.ones(attack_samples) / attack_samples,
                np.exp(-np.arange(release_samples) / (release_samples/3))
            ])
            window = window / np.sum(window)

            # Smooth gain reduction for each channel
            smoothed_reduction = np.zeros_like(gain_reduction_db)
            for channel in range(data.shape[1]):
                smoothed_reduction[:, channel] = np.convolve(
                    gain_reduction_db[:, channel], 
                    window, 
                    mode='same'
                )

            # Apply compression (maintains dimensions)
            compressed = data * np.power(10, -smoothed_reduction/20)

            # Volume increase
            volume_increased = compressed * self.config.volume_factor

            # Soft-knee Limiter Stage
            # Implement transparent limiting to prevent digital clipping while
            # maintaining natural sound. Features:
            # 1. Gradual gain reduction using soft knee for transparency
            # 2. Independent per-channel processing
            # 3. Automated makeup gain and peak protection

            limiter_threshold = 0.95  # Target maximum output level
            knee_width = 0.1  # Width of soft knee transition region
            max_amplitude = np.max(np.abs(volume_increased), axis=0)

            # Only limit if needed
            if np.any(max_amplitude > (limiter_threshold - knee_width)):
                self.logger.info(f"Applying soft-knee limiting above {limiter_threshold-knee_width:.2f}")
                
                # Calculate attenuation curve per channel
                gain_reduction = np.zeros_like(volume_increased)
                amplitude = np.abs(volume_increased)
                
                # Soft knee region
                knee_start = limiter_threshold - knee_width
                knee_end = limiter_threshold
                
                for channel in range(data.shape[1]):
                    knee_mask = (amplitude[:, channel] > knee_start) & (amplitude[:, channel] <= knee_end)
                    above_knee = amplitude[:, channel] > knee_end
                    
                    # Quadratic gain reduction in knee region
                    x = (amplitude[knee_mask, channel] - knee_start) / knee_width
                    gain_reduction[knee_mask, channel] = x * x * (knee_width / 2)
                    
                    # Hard limiting above knee
                    gain_reduction[above_knee, channel] = amplitude[above_knee, channel] - limiter_threshold
                
                # Apply gain reduction (maintains dimensions)
                volume_increased = volume_increased * np.power(10, -gain_reduction/20)

            # Squeeze back to mono if input was mono
            if data.shape[1] == 1:
                volume_increased = volume_increased.squeeze()

            # Calculate final stats
            # Calculate per-channel statistics for processed audio
            final_max_per_channel = np.max(np.abs(volume_increased), axis=0)
            final_rms_per_channel = np.sqrt(np.mean(volume_increased**2, axis=0))

            # Convert to scalar values for logging
            final_max = float(np.mean(final_max_per_channel))
            final_rms = float(np.mean(final_rms_per_channel))

            # Calculate ratios using mean values to avoid division issues
            rms_ratio = float(np.mean(final_rms_per_channel / np.maximum(original_rms, 1e-10)))
            peak_ratio = float(np.mean(final_max_per_channel / np.maximum(original_max, 1e-10)))

            self.logger.info(f"Final max amplitude: {final_max:.4f}, RMS: {final_rms:.4f}")
            self.logger.info(f"Volume increase: {rms_ratio:.2f}x (RMS), Peak: {peak_ratio:.2f}x")

            # Write processed audio
            sf.write(file_path, volume_increased, samplerate)
            self.logger.info(f"Processed: {file_path}")
            return True

        except Exception as e:
            self.logger.error(f"Error processing {file_path}: {str(e)}")
            return False

class FileHandler:
    """Handles file system operations for audio processing.
    
    This class manages all file-related operations including:
    - Recursive file discovery
    - Format filtering
    - Recent file filtering
    - Path validation
    - File enumeration
    
    It works closely with the ProcessingConfig to determine which files
    should be processed based on format, recursion settings, and time
    filters."""
    def __init__(self, config: ProcessingConfig, logger: logging.Logger):
        """
        Initialize the FileHandler with configuration and logger.

        This constructor sets up the FileHandler with the necessary configuration
        and logging capabilities for file system operations.

        Args:
            config (ProcessingConfig): Configuration object containing processing parameters
                                       and settings for file handling.
            logger (logging.Logger): Logger object for recording file handling operations
                                     and any relevant messages.

        The method doesn't return any value but initializes the instance variables.
        """
        self.config = config
        self.logger = logger

    def get_files_to_process(self) -> List[Path]:
        """Get list of audio files to process."""
        self.logger.info(f"Searching for audio files in: {self.config.input_folder}")
        self.logger.info(f"File formats to process: {self.config.formats}")
        
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
            
            self.logger.info(f"Found {len(matches)} files matching {pattern}")
            for match in matches:
                self.logger.debug(f"Found file: {match}")
            files.extend(matches)

        if self.config.recent_only:
            cutoff_time = datetime.now() - timedelta(hours=24)
            self.logger.info(f"Filtering for files modified after: {cutoff_time}")
            files = [f for f in files if f.stat().st_mtime > cutoff_time.timestamp()]
            self.logger.info(f"After time filter: {len(files)} files remaining")

        self.logger.info(f"Total files to process: {len(files)}")
        return files

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Audio processing tool to adjust volume of audio files"
    )
    parser.add_argument("--folder", type=Path, required=True,
                    help="Input directory containing audio files to process")
    parser.add_argument("--backup-folder", type=Path,
                    help="Directory to store backup copies of original files")
    parser.add_argument("--volume-factor", type=float, default=1.2,
                    help="Multiplication factor for volume adjustment. 1.2 means 20 percent increase")
    parser.add_argument("--formats", type=str, default="wav",
                    help="Comma-separated list of audio formats to process. Example: wav,mp3,ogg")
    parser.add_argument("--recursive", action="store_true",
                    help="Process audio files in all subdirectories recursively")
    parser.add_argument("--recent", action="store_true",
                    help="Only process files modified in the last 24 hours")
    parser.add_argument("--workers", type=int, default=1,
                    help="Number of parallel processing workers")
    parser.add_argument("--dry-run", action="store_true",
                    help="Preview changes without modifying any files")
    parser.add_argument("--log-file", type=str,
                    help="Output file for detailed processing logs")
    return parser.parse_args()

def main() -> None:
    print("Starting audio processing script...")
    try:
        # Parse arguments first
        args = parse_arguments()
        
        # Initialize logger with parsed arguments
        logger = Logger(args.log_file)
        
        # Log startup information
        logger.info("Audio Processing Script Started")
        logger.info(f"Parsed arguments: {vars(args)}")

        # Create backup directory if specified
        backup_path = None
        if args.backup_folder:
            backup_path = Path(args.backup_folder)
            try:
                logger.info(f"Setting up backup directory: {backup_path}")
                backup_path.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                logger.error(f"Failed to create backup directory: {e}")
                raise

        # Create configuration
        config = ProcessingConfig(
            input_folder=Path(args.folder),
            backup_folder=Path(args.backup_folder) if args.backup_folder else None,
            formats=set(args.formats.lower().split(',')) if args.formats else {'wav'},
            dry_run=args.dry_run,
            recursive=args.recursive,
            recent_only=args.recent,
            workers=args.workers,
            log_file=args.log_file,
            volume_factor=args.volume_factor
        )

        logger.info("Configuration created successfully")
        logger.info(f"Input folder: {config.input_folder}")
        logger.info(f"Backup folder: {config.backup_folder}")
        logger.info(f"Formats: {config.formats}")

        # Initialize handlers
        file_handler = FileHandler(config, logger)
        files = file_handler.get_files_to_process()
        
        if not files:
            logger.info("No files found to process")
            return

        # Initialize audio processor
        audio_processor = AudioProcessor(config, logger)
        
        # Process files
        with ProcessPoolExecutor(max_workers=config.workers) as executor:
            if config.dry_run:
                for file in files:
                    logger.info(f"Would process: {file}")
                    if config.backup_folder:
                        logger.info(f"Would backup to: {config.backup_folder / file.name}")
                    logger.info(f"Would increase volume by factor: {config.volume_factor}")
            else:
                # Submit all file processing tasks
                future_to_file = {executor.submit(audio_processor.process_file, file): file for file in files}
                
                # Process results as they complete
                for future in tqdm(as_completed(future_to_file), total=len(files), desc="Processing files"):
                    file = future_to_file[future]
                    try:
                        success = future.result()
                        if success:
                            logger.info(f"Successfully processed: {file}")
                        else:
                            logger.error(f"Failed to process: {file}")
                    except Exception as e:
                        logger.error(f"Error processing {file}: {str(e)}")

    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
if __name__ == '__main__':
    main()
