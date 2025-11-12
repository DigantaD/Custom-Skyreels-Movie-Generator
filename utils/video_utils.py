"""
Video Utilities Module
Helper functions for video manipulation and processing
"""

import cv2
import numpy as np
from typing import List, Optional, Tuple
from pathlib import Path
import subprocess
import os


def get_video_info(video_path: str) -> dict:
    """Get video information"""
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    info = {
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "duration": cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
    }
    
    cap.release()
    return info


def resize_video(input_path: str, output_path: str, target_resolution: Tuple[int, int], 
                 maintain_aspect: bool = True):
    """Resize video to target resolution"""
    cap = cv2.VideoCapture(input_path)
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {input_path}")
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    target_width, target_height = target_resolution
    
    if maintain_aspect:
        # Calculate aspect ratio
        aspect = width / height
        if target_width / target_height > aspect:
            target_width = int(target_height * aspect)
        else:
            target_height = int(target_width / aspect)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (target_width, target_height))
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        resized_frame = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_LANCZOS4)
        out.write(resized_frame)
    
    cap.release()
    out.release()


def concatenate_videos(video_paths: List[str], output_path: str, 
                      transition_duration: float = 0.0, method: str = "ffmpeg"):
    """
    Concatenate multiple videos into one
    
    Args:
        video_paths: List of video file paths
        output_path: Output video path
        transition_duration: Duration of transition between videos (in seconds)
        method: "ffmpeg" (fast) or "opencv" (slower but more control)
    """
    if method == "ffmpeg":
        _concatenate_with_ffmpeg(video_paths, output_path)
    else:
        _concatenate_with_opencv(video_paths, output_path, transition_duration)


def _concatenate_with_ffmpeg(video_paths: List[str], output_path: str):
    """Concatenate videos using ffmpeg (faster)"""
    # Create temporary file list for ffmpeg
    list_file = output_path.replace('.mp4', '_filelist.txt')
    
    with open(list_file, 'w') as f:
        for video_path in video_paths:
            f.write(f"file '{os.path.abspath(video_path)}'\n")
    
    try:
        # Use ffmpeg concat demuxer
        cmd = [
            'ffmpeg',
            '-f', 'concat',
            '-safe', '0',
            '-i', list_file,
            '-c', 'copy',
            output_path,
            '-y'  # Overwrite output file
        ]
        
        subprocess.run(cmd, check=True, capture_output=True)
    finally:
        # Clean up
        if os.path.exists(list_file):
            os.remove(list_file)


def _concatenate_with_opencv(video_paths: List[str], output_path: str, transition_duration: float):
    """Concatenate videos using OpenCV (more control, slower)"""
    if not video_paths:
        raise ValueError("No video paths provided")
    
    # Get info from first video
    first_video_info = get_video_info(video_paths[0])
    width = first_video_info["width"]
    height = first_video_info["height"]
    fps = first_video_info["fps"]
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    transition_frames = int(transition_duration * fps) if transition_duration > 0 else 0
    
    for i, video_path in enumerate(video_paths):
        cap = cv2.VideoCapture(video_path)
        
        # Resize if needed
        video_info = get_video_info(video_path)
        if video_info["width"] != width or video_info["height"] != height:
            # Need to resize frames
            resize_needed = True
        else:
            resize_needed = False
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if resize_needed:
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LANCZOS4)
            
            # Add transition effect if not first video and transition frames remain
            if i > 0 and transition_frames > 0 and frame_count < transition_frames:
                # Simple fade transition
                alpha = frame_count / transition_frames
                # This is a simplified transition - can be enhanced
                pass
            
            out.write(frame)
            frame_count += 1
        
        cap.release()
    
    out.release()


def add_transition(video1_path: str, video2_path: str, output_path: str, 
                   transition_type: str = "fade", duration: float = 0.5):
    """Add transition between two videos"""
    # This is a placeholder - can be enhanced with more transition types
    # For now, just concatenate
    concatenate_videos([video1_path, video2_path], output_path)


def extract_frame(video_path: str, frame_number: int, output_path: str):
    """Extract a specific frame from video"""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    
    ret, frame = cap.read()
    if ret:
        cv2.imwrite(output_path, frame)
    
    cap.release()
    return ret


def get_first_frame(video_path: str, output_path: Optional[str] = None) -> np.ndarray:
    """Get the first frame of a video"""
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        raise ValueError(f"Could not read first frame from {video_path}")
    
    if output_path:
        cv2.imwrite(output_path, frame)
    
    return frame


def get_last_frame(video_path: str, output_path: Optional[str] = None) -> np.ndarray:
    """Get the last frame of a video"""
    cap = cv2.VideoCapture(video_path)
    
    # Get total frame count
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Seek to last frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        raise ValueError(f"Could not read last frame from {video_path}")
    
    if output_path:
        cv2.imwrite(output_path, frame)
    
    return frame


def normalize_resolution(resolution_str: str) -> Tuple[int, int]:
    """Convert resolution string to (width, height) tuple"""
    resolution_map = {
        "360p": (640, 360),
        "480p": (854, 480),
        "540p": (960, 540),
        "720p": (1280, 720),
        "1080p": (1920, 1080)
    }
    
    resolution_str = resolution_str.lower()
    if resolution_str in resolution_map:
        return resolution_map[resolution_str]
    
    # Try to parse custom resolution (e.g., "1280x720")
    if 'x' in resolution_str:
        parts = resolution_str.split('x')
        if len(parts) == 2:
            try:
                return (int(parts[0]), int(parts[1]))
            except ValueError:
                pass
    
    # Default to 540p
    return resolution_map["540p"]

