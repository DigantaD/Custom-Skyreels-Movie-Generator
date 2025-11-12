"""
Segment Generator Module
Generates video segments using SkyReels-V2 T2V and I2V models
"""

import torch
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List
import cv2
from PIL import Image
import yaml
from tqdm import tqdm

from utils.model_loader import ModelLoader
from utils.video_utils import normalize_resolution


class SegmentGenerator:
    """Generates video segments from prompts"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.model_loader = ModelLoader(config_path)
        self.device = self.config.get("performance", {}).get("device", "cuda")
        
        # Generation parameters
        self.resolution = self.config.get("generation", {}).get("resolution", "540p")
        self.num_inference_steps = self.config.get("generation", {}).get("num_inference_steps", 50)
        self.guidance_scale = self.config.get("generation", {}).get("guidance_scale", 7.5)
        self.fps = self.config.get("generation", {}).get("fps", 8)
        self.seed = self.config.get("generation", {}).get("seed")
        
        # Models (lazy loaded)
        self.t2v_model = None
        self.i2v_model = None
    
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load configuration"""
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        return {}
    
    def _get_t2v_model(self):
        """Lazy load T2V model"""
        if self.t2v_model is None:
            print("Loading T2V model...")
            self.t2v_model = self.model_loader.get_t2v_model()
            print("T2V model loaded!")
        return self.t2v_model
    
    def _get_i2v_model(self):
        """Lazy load I2V model"""
        if self.i2v_model is None:
            print("Loading I2V model...")
            self.i2v_model = self.model_loader.get_i2v_model()
            print("I2V model loaded!")
        return self.i2v_model
    
    def generate_t2v_segment(self, prompt: str, output_path: str, 
                            duration: Optional[float] = None) -> str:
        """
        Generate a video segment from text prompt
        
        Args:
            prompt: Text prompt for video generation
            output_path: Path to save the generated video
            duration: Target duration in seconds (optional)
            
        Returns:
            Path to generated video
        """
        model = self._get_t2v_model()
        
        # Get resolution
        width, height = normalize_resolution(self.resolution)
        
        # Calculate number of frames
        if duration:
            num_frames = int(duration * self.fps)
        else:
            # Default: 3 seconds
            num_frames = int(3 * self.fps)
        
        print(f"Generating T2V segment: {prompt[:50]}...")
        print(f"Resolution: {width}x{height}, Frames: {num_frames}, FPS: {self.fps}")
        
        # Set seed for reproducibility
        generator = None
        if self.seed is not None:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(self.seed)
        
        try:
            # Generate video using the model
            # Try multiple API patterns based on common diffusion model interfaces
            
            # Pattern 1: Standard diffusers pipeline
            if hasattr(model, '__call__'):
                try:
                    # Try standard diffusers API
                    output = model(
                        prompt=prompt,
                        num_inference_steps=self.num_inference_steps,
                        guidance_scale=self.guidance_scale,
                        num_frames=num_frames,
                        height=height,
                        width=width,
                        generator=generator
                    )
                    
                    # Handle different output formats
                    if hasattr(output, 'frames'):
                        frames = output.frames
                    elif hasattr(output, 'videos'):
                        frames = output.videos[0]  # First video in batch
                    elif isinstance(output, (list, tuple)):
                        frames = output[0] if len(output) > 0 else output
                    elif hasattr(output, 'images'):
                        # Some models return images instead of frames
                        frames = output.images
                    else:
                        raise ValueError(f"Unknown output format: {type(output)}")
                    
                except Exception as e1:
                    # Pattern 2: Try with generate() method
                    if hasattr(model, 'generate'):
                        try:
                            output = model.generate(
                                prompt=prompt,
                                num_inference_steps=self.num_inference_steps,
                                guidance_scale=self.guidance_scale,
                                num_frames=num_frames,
                                height=height,
                                width=width,
                                generator=generator
                            )
                            frames = output if isinstance(output, list) else [output]
                        except Exception as e2:
                            raise RuntimeError(
                                f"Failed to generate video. Tried standard API ({e1}) and generate() method ({e2}). "
                                "Please check SkyReels-V2 integration in INTEGRATION_NOTES.md"
                            )
                    else:
                        raise
            
            # Convert frames to numpy arrays if needed
            frames = self._normalize_frames(frames)
            
            # Save video
            self._save_frames_as_video(frames, output_path, self.fps)
            
            print(f"✓ T2V segment saved to: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"✗ Error generating T2V segment: {e}")
            print("\n" + "="*60)
            print("TROUBLESHOOTING:")
            print("1. Check that SkyReels-V2 models are properly downloaded")
            print("2. Verify GPU availability and memory")
            print("3. Review INTEGRATION_NOTES.md for SkyReels integration")
            print("4. Try reducing resolution or num_frames if out of memory")
            print("="*60)
            raise
    
    def generate_i2v_segment(self, prompt: str, image_path: str, output_path: str,
                            duration: Optional[float] = None) -> str:
        """
        Generate a video segment from image and prompt
        
        Args:
            prompt: Text prompt for video generation
            image_path: Path to starting image
            output_path: Path to save the generated video
            duration: Target duration in seconds (optional)
            
        Returns:
            Path to generated video
        """
        model = self._get_i2v_model()
        
        # Load image
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        image = Image.open(image_path).convert("RGB")
        
        # Get resolution
        width, height = normalize_resolution(self.resolution)
        
        # Resize image if needed
        if image.size != (width, height):
            image = image.resize((width, height), Image.LANCZOS)
        
        # Calculate number of frames
        if duration:
            num_frames = int(duration * self.fps)
        else:
            num_frames = int(3 * self.fps)
        
        print(f"Generating I2V segment: {prompt[:50]}...")
        print(f"Resolution: {width}x{height}, Frames: {num_frames}, FPS: {self.fps}")
        
        # Set seed for reproducibility
        generator = None
        if self.seed is not None:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(self.seed)
        
        try:
            # Generate video using the model
            # Try multiple API patterns for I2V
            
            if hasattr(model, '__call__'):
                try:
                    # Try standard diffusers API with image
                    output = model(
                        prompt=prompt,
                        image=image,
                        num_inference_steps=self.num_inference_steps,
                        guidance_scale=self.guidance_scale,
                        num_frames=num_frames,
                        height=height,
                        width=width,
                        generator=generator
                    )
                    
                    # Handle different output formats
                    if hasattr(output, 'frames'):
                        frames = output.frames
                    elif hasattr(output, 'videos'):
                        frames = output.videos[0]
                    elif isinstance(output, (list, tuple)):
                        frames = output[0] if len(output) > 0 else output
                    elif hasattr(output, 'images'):
                        frames = output.images
                    else:
                        raise ValueError(f"Unknown output format: {type(output)}")
                        
                except Exception as e1:
                    if hasattr(model, 'generate'):
                        try:
                            output = model.generate(
                                prompt=prompt,
                                image=image,
                                num_inference_steps=self.num_inference_steps,
                                guidance_scale=self.guidance_scale,
                                num_frames=num_frames,
                                height=height,
                                width=width,
                                generator=generator
                            )
                            frames = output if isinstance(output, list) else [output]
                        except Exception as e2:
                            raise RuntimeError(
                                f"Failed to generate I2V. Tried standard API ({e1}) and generate() ({e2}). "
                                "Please check SkyReels-V2 integration."
                            )
                    else:
                        raise
            
            # Convert frames to numpy arrays if needed
            frames = self._normalize_frames(frames)
            
            # Save video
            self._save_frames_as_video(frames, output_path, self.fps)
            
            print(f"✓ I2V segment saved to: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"✗ Error generating I2V segment: {e}")
            print("\n" + "="*60)
            print("TROUBLESHOOTING:")
            print("1. Verify image path and format")
            print("2. Check model loading and GPU availability")
            print("3. Review INTEGRATION_NOTES.md")
            print("="*60)
            raise
    
    def _normalize_frames(self, frames):
        """Normalize frames to numpy arrays"""
        normalized = []
        
        for frame in frames:
            # Handle PIL Images
            if hasattr(frame, 'convert'):
                frame = np.array(frame.convert('RGB'))
            
            # Handle torch tensors
            elif hasattr(frame, 'cpu'):
                frame = frame.cpu().numpy()
                # Handle channel-first tensors (C, H, W) -> (H, W, C)
                if len(frame.shape) == 3 and frame.shape[0] == 3:
                    frame = frame.transpose(1, 2, 0)
            
            # Ensure it's a numpy array
            if not isinstance(frame, np.ndarray):
                frame = np.array(frame)
            
            # Normalize to 0-255 range if needed
            if frame.dtype == np.float32 or frame.dtype == np.float64:
                if frame.max() <= 1.0:
                    frame = (frame * 255).astype(np.uint8)
                else:
                    frame = frame.astype(np.uint8)
            
            normalized.append(frame)
        
        return normalized
    
    def _save_frames_as_video(self, frames: List[np.ndarray], output_path: str, fps: float):
        """Save frames as video file"""
        if not frames:
            raise ValueError("No frames to save")
        
        # Get frame dimensions
        first_frame = frames[0]
        if len(first_frame.shape) == 3:
            height, width = first_frame.shape[:2]
        else:
            raise ValueError(f"Invalid frame shape: {first_frame.shape}")
        
        # Create output directory if needed
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if not out.isOpened():
            raise RuntimeError(f"Failed to create video writer for {output_path}")
        
        # Write frames
        for idx, frame in enumerate(frames):
            # Ensure correct shape
            if len(frame.shape) != 3 or frame.shape[2] != 3:
                raise ValueError(f"Frame {idx} has invalid shape: {frame.shape}")
            
            # Convert RGB to BGR (OpenCV uses BGR)
            if frame.shape[2] == 3:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                frame_bgr = frame
            
            # Ensure uint8
            if frame_bgr.dtype != np.uint8:
                if frame_bgr.max() <= 1.0:
                    frame_bgr = (frame_bgr * 255).astype(np.uint8)
                else:
                    frame_bgr = frame_bgr.astype(np.uint8)
            
            # Resize if dimensions don't match
            if frame_bgr.shape[:2] != (height, width):
                frame_bgr = cv2.resize(frame_bgr, (width, height), interpolation=cv2.INTER_LANCZOS4)
            
            out.write(frame_bgr)
        
        out.release()
        print(f"  Saved {len(frames)} frames at {fps} FPS")
    
    def generate_batch(self, prompts: List[str], output_dir: str, mode: str = "t2v",
                      image_paths: Optional[List[str]] = None) -> List[str]:
        """
        Generate multiple segments in batch
        
        Args:
            prompts: List of prompts
            output_dir: Directory to save segments
            mode: "t2v" or "i2v"
            image_paths: List of image paths (required for i2v mode)
            
        Returns:
            List of output video paths
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        output_paths = []
        
        for idx, prompt in enumerate(tqdm(prompts, desc="Generating segments")):
            output_path = Path(output_dir) / f"segment_{idx+1:03d}.mp4"
            
            if mode == "t2v":
                self.generate_t2v_segment(prompt, str(output_path))
            elif mode == "i2v":
                if not image_paths or idx >= len(image_paths):
                    raise ValueError(f"Image path required for segment {idx+1}")
                self.generate_i2v_segment(prompt, image_paths[idx], str(output_path))
            else:
                raise ValueError(f"Invalid mode: {mode}")
            
            output_paths.append(str(output_path))
        
        return output_paths

