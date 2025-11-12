"""
Video Stitcher Module
Concatenates video segments into final video
"""

from pathlib import Path
from typing import List, Optional
import yaml

from utils.video_utils import concatenate_videos, get_video_info


class VideoStitcher:
    """Stitches video segments together"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.enable_transitions = self.config.get("video", {}).get("enable_transitions", True)
        self.transition_duration = self.config.get("video", {}).get("transition_duration", 0.5)
        self.output_format = self.config.get("video", {}).get("output_format", "mp4")
    
    def _load_config(self, config_path: Optional[str]) -> dict:
        """Load configuration"""
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        return {}
    
    def stitch_segments(self, segment_paths: List[str], output_path: str,
                       method: str = "ffmpeg") -> str:
        """
        Stitch multiple video segments into one video
        
        Args:
            segment_paths: List of paths to video segments
            output_path: Path for output video
            method: "ffmpeg" (fast) or "opencv" (more control)
            
        Returns:
            Path to stitched video
        """
        if not segment_paths:
            raise ValueError("No segment paths provided")
        
        # Validate all segments exist
        for path in segment_paths:
            if not Path(path).exists():
                raise FileNotFoundError(f"Segment not found: {path}")
        
        print(f"Stitching {len(segment_paths)} segments into final video...")
        
        # Ensure output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Concatenate videos
        transition_duration = self.transition_duration if self.enable_transitions else 0.0
        concatenate_videos(segment_paths, output_path, transition_duration, method)
        
        # Get final video info
        info = get_video_info(output_path)
        print(f"Final video created: {output_path}")
        print(f"  Duration: {info['duration']:.2f} seconds")
        print(f"  Resolution: {info['width']}x{info['height']}")
        print(f"  FPS: {info['fps']:.2f}")
        
        return output_path
    
    def validate_segments(self, segment_paths: List[str]) -> dict:
        """
        Validate that all segments are compatible for stitching
        
        Returns:
            Dictionary with validation results
        """
        if not segment_paths:
            return {"valid": False, "error": "No segments provided"}
        
        # Get info from first segment
        first_info = get_video_info(segment_paths[0])
        base_resolution = (first_info["width"], first_info["height"])
        base_fps = first_info["fps"]
        
        issues = []
        
        for idx, path in enumerate(segment_paths[1:], start=1):
            info = get_video_info(path)
            
            if (info["width"], info["height"]) != base_resolution:
                issues.append(f"Segment {idx+1}: Resolution mismatch "
                            f"({info['width']}x{info['height']} vs {base_resolution[0]}x{base_resolution[1]})")
            
            if abs(info["fps"] - base_fps) > 0.1:
                issues.append(f"Segment {idx+1}: FPS mismatch "
                            f"({info['fps']:.2f} vs {base_fps:.2f})")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "base_resolution": base_resolution,
            "base_fps": base_fps
        }

