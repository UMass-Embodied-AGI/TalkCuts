#!/usr/bin/env python3
"""
YouTube Video Downloader using yt-dlp
Downloads videos from a CSV file containing YouTube URLs
"""

import csv
import os
import sys
from pathlib import Path
import subprocess


def extract_video_id(url):
    """Extract video ID from YouTube URL"""
    if 'v=' in url:
        return url.split('v=')[1].split('&')[0]
    elif 'youtu.be/' in url:
        return url.split('youtu.be/')[1].split('?')[0]
    return None


def download_videos(csv_file, target_folder):
    """
    Download videos from CSV file using yt-dlp
    
    Args:
        csv_file: Path to CSV file with columns: URL, Camera, Type
        target_folder: Folder where videos will be downloaded
    """
    # Create target folder if it doesn't exist
    target_path = Path(target_folder)
    target_path.mkdir(parents=True, exist_ok=True)
    
    # Check if yt-dlp is installed
    try:
        subprocess.run(['yt-dlp', '--version'], 
                      capture_output=True, 
                      check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: yt-dlp is not installed.")
        print("Install it with: pip install yt-dlp")
        sys.exit(1)
    
    # Read CSV file
    try:
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            videos = list(reader)
    except FileNotFoundError:
        print(f"Error: CSV file '{csv_file}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        sys.exit(1)
    
    print(f"Found {len(videos)} videos to download")
    print(f"Target folder: {target_path.absolute()}")
    print("-" * 60)
    
    # Download each video
    successful = 0
    failed = 0
    
    for idx, video in enumerate(videos, 1):
        url = video.get('URL', '').strip()
        if not url:
            print(f"[{idx}/{len(videos)}] Skipping empty URL")
            failed += 1
            continue
        
        # Extract video ID
        video_id = extract_video_id(url)
        if not video_id:
            print(f"[{idx}/{len(videos)}] Error: Could not extract video ID from {url}")
            failed += 1
            continue
        
        # Set output filename
        output_file = target_path / f"{video_id}.mp4"
        
        # Check if file already exists
        if output_file.exists():
            print(f"[{idx}/{len(videos)}] Skipping {video_id}.mp4 (already exists)")
            successful += 1
            continue
        
        print(f"[{idx}/{len(videos)}] Downloading {video_id}.mp4...")
        
        # Download using yt-dlp
        try:
            cmd = [
                'yt-dlp',
                '-f', 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
                '--merge-output-format', 'mp4',
                '-o', str(output_file),
                url
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            print(f"[{idx}/{len(videos)}] ✓ Successfully downloaded {video_id}.mp4")
            successful += 1
            
        except subprocess.CalledProcessError as e:
            print(f"[{idx}/{len(videos)}] ✗ Failed to download {video_id}: {e}")
            failed += 1
        except Exception as e:
            print(f"[{idx}/{len(videos)}] ✗ Unexpected error: {e}")
            failed += 1
    
    # Summary
    print("-" * 60)
    print(f"Download complete!")
    print(f"Successful: {successful}/{len(videos)}")
    print(f"Failed: {failed}/{len(videos)}")


def main():
    """Main function with command-line argument parsing"""
    if len(sys.argv) != 3:
        print("Usage: python youtube_downloader.py <csv_file> <target_folder>")
        print("\nExample:")
        print("  python youtube_downloader.py videos.csv ./downloads")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    target_folder = sys.argv[2]
    
    download_videos(csv_file, target_folder)


if __name__ == "__main__":
    main()