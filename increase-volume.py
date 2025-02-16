"""
Audio Volume Increaser

This script increases the volume of WAV files by a fixed percentage (default 20%),
with options for backup and filtering recent files.

Usage Examples:
--------------
1. Basic usage (uses default folder):
    python increase-volume.py

2. Process files in a specific folder:
    python increase-volume.py --folder /path/to/audio/files

3. Create backups before processing:
    python increase-volume.py --backup-folder /path/to/backups

4. Only process files from the last 24 hours:
    python increase-volume.py --recent

5. Combining multiple options:
    python increase-volume.py --folder /path/to/audio/files --backup-folder /path/to/backups --recent

Notes:
- The script processes all .wav files in the specified folder
- Volume is increased by 20% by default
- Audio is automatically normalized if needed to prevent clipping
- Original files are overwritten unless --backup-folder is specified
"""

import os
import soundfile as sf
import numpy as np
import logging
from tqdm import tqdm
import argparse
from datetime import datetime, timedelta

# Configuration
VOLUME_INCREASE_FACTOR = 1.2  # 20% increase in volume

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

def process_audio_file(file_path, backup_folder=None):
    try:
        # Create backup if backup folder is specified
        if backup_folder:
            try:
                backup_path = os.path.join(backup_folder, os.path.basename(file_path))
                if os.path.exists(backup_path):
                    logging.warning(f"Backup already exists: {backup_path}")
                else:
                    sf.read(file_path)  # Test if file is readable first
                    os.makedirs(backup_folder, exist_ok=True)
                    import shutil
                    shutil.copy2(file_path, backup_path)
                    logging.info(f"Created backup: {backup_path}")
            except Exception as e:
                logging.error(f"Failed to create backup for {file_path}: {str(e)}")
                return False
        
        # Read audio file
        data, samplerate = sf.read(file_path)
        
        # Increase volume by multiplying the audio data
        volume_increased_data = data * VOLUME_INCREASE_FACTOR
        
        # Ensure we don't clip the audio (normalize if needed)
        if np.abs(volume_increased_data).max() > 1.0:
            volume_increased_data = volume_increased_data / np.abs(volume_increased_data).max()
        
        # overwrite existing files    
        filename = os.path.basename(file_path)
        name, ext = os.path.splitext(filename)
        new_filename = f"{name}{ext}"
        new_file_path = os.path.join(os.path.dirname(file_path), new_filename)
        
        # Write the processed audio to a new file
        sf.write(new_file_path, volume_increased_data, samplerate, subtype='PCM_24')
        logging.info(f"Successfully processed: {filename} → {new_filename}")
        return True
    except Exception as e:
        logging.error(f"Error processing {file_path}: {str(e)}")
        return False

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Process WAV files to increase volume')
    parser.add_argument('--folder', default='/Users/brian/Documents/VSTLive/projects/audio/Gig',
                    help='Folder path containing WAV files')
    parser.add_argument('--backup-folder',
                    help='Folder path to store backup copies of original files before processing')
    parser.add_argument('--recent', action='store_true',
                    help='Only process WAV files created in the last 24 hours')
    args = parser.parse_args()
    
    if not os.path.exists(args.folder):
        logging.error(f"Folder not found: {args.folder}")
        return
    
    processed_count = 0
    error_count = 0
    
    # Get list of WAV files
    wav_files = [f for f in os.listdir(args.folder) if f.endswith('.wav')]
    
    # Filter for recent files if --recent flag is set
    if args.recent:
        current_time = datetime.now()
        wav_files = [f for f in wav_files if 
                    (current_time - datetime.fromtimestamp(
                        os.path.getctime(os.path.join(args.folder, f))
                    )) < timedelta(hours=24)]
        logging.info(f"Found {len(wav_files)} WAV files from the last 24 hours")

    # Process files with progress bar
    for filename in tqdm(wav_files, desc="Processing audio files"):
        file_path = os.path.join(args.folder, filename)
        if process_audio_file(file_path, args.backup_folder):
            processed_count += 1
        else:
            error_count += 1

    logging.info(f"\nProcessing complete:")
    logging.info(f"Successfully processed: {processed_count} files")
    if error_count > 0:
        logging.info(f"Errors encountered: {error_count} files")

if __name__ == "__main__":
    main()
