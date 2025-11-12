"""
Post Processor Module
Basic video post-processing operations
"""

from pathlib import Path
from typing import Optional
import cv2
import numpy as np

from utils.video_utils import get_video_info, resize_video


class PostProcessor:
    """Post-processes generated videos"""
    
    def __init__(self):
        pass
    
    def normalize_videos(self, video_paths: list, target_resolution: tuple, 
                        target_fps: float, output_dir: str) -> list:
        """
        Normalize multiple videos to same resolution and FPS
        
        Args:
            video_paths: List of video paths
            target_resolution: (width, height) tuple
            target_fps: Target FPS
            output_dir: Directory for normalized videos
            
        Returns:
            List of normalized video paths
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        normalized_paths = []
        
        for idx, video_path in enumerate(video_paths):
            output_path = Path(output_dir) / f"normalized_{idx+1:03d}.mp4"
            
            # Resize video
            resize_video(video_path, str(output_path), target_resolution)
            
            # TODO: Resample FPS if needed (requires ffmpeg or more complex processing)
            
            normalized_paths.append(str(output_path))
        
        return normalized_paths
    
    def add_fade_in(self, video_path: str, output_path: str, duration: float = 0.5):
        """Add fade-in effect to video"""
        cap = cv2.VideoCapture(video_path)
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        fade_frames = int(duration * fps)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Apply fade-in
            if frame_idx < fade_frames:
                alpha = frame_idx / fade_frames
                frame = (frame * alpha).astype(np.uint8)
            
            out.write(frame)
            frame_idx += 1
        
        cap.release()
        out.release()
    
    def add_fade_out(self, video_path: str, output_path: str, duration: float = 0.5):
        """Add fade-out effect to video"""
        cap = cv2.VideoCapture(video_path)
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        fade_frames = int(duration * fps)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Apply fade-out
            if frame_idx >= total_frames - fade_frames:
                alpha = (total_frames - frame_idx) / fade_frames
                frame = (frame * alpha).astype(np.uint8)
            
            out.write(frame)
            frame_idx += 1
        
        cap.release()
        out.release()
    
    def adjust_brightness(self, video_path: str, output_path: str, factor: float):
        """
        Adjust video brightness
        
        Args:
            video_path: Input video
            output_path: Output video
            factor: Brightness factor (1.0 = no change, >1.0 = brighter, <1.0 = darker)
        """
        cap = cv2.VideoCapture(video_path)
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Adjust brightness
            adjusted = cv2.convertScaleAbs(frame, alpha=1, beta=(factor - 1.0) * 127)
            out.write(adjusted)
        
        cap.release()
        out.release()

