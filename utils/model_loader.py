"""
Model Loader Module
Handles loading SkyReels-V2 models from Hugging Face
"""

import torch
from pathlib import Path
from typing import Optional, Dict
import yaml
from huggingface_hub import snapshot_download
import os


class ModelLoader:
    """Loads and manages SkyReels-V2 models"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.device = self.config.get("performance", {}).get("device", "cuda")
        self.enable_offload = self.config.get("performance", {}).get("enable_offload", True)
        
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load configuration from YAML file"""
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        return {}
    
    def get_t2v_model(self, model_path: Optional[str] = None, hf_id: Optional[str] = None):
        """
        Load Text-to-Video model
        
        Args:
            model_path: Local path to model (if None, downloads from HF)
            hf_id: Hugging Face model ID
            
        Returns:
            Loaded model pipeline
        """
        # Determine model path
        if model_path and Path(model_path).exists():
            model_dir = model_path
        else:
            # Use config or default
            hf_id = hf_id or self.config.get("model", {}).get("t2v_hf_id", 
                "Skywork/SkyReels-V2-DF-14B-720P-Diffusers")
            
            # Check if model already exists locally
            local_dir = Path("./models/t2v")
            if local_dir.exists() and any(local_dir.iterdir()):
                print(f"Using existing T2V model from: {local_dir}")
                model_dir = str(local_dir)
            else:
                # Download from Hugging Face
                print(f"Downloading T2V model from Hugging Face: {hf_id}")
                print("This may take a while...")
                model_dir = snapshot_download(
                    repo_id=hf_id,
                    local_dir=str(local_dir),
                    local_dir_use_symlinks=False
                )
        
        # Try multiple loading strategies
        # Strategy 1: Try SkyReels custom pipeline (if available)
        try:
            # Check if SkyReels inference code is available
            import sys
            skyreels_path = Path("./skyreels_v2_infer")
            if not skyreels_path.exists():
                # Try to find it in parent directory or as installed package
                skyreels_path = Path("../SkyReels-V2/skyreels_v2_infer")
            
            if skyreels_path.exists():
                sys.path.insert(0, str(skyreels_path.parent))
                # This is a placeholder - adapt based on actual SkyReels code
                # from skyreels_v2_infer import SkyReelsPipeline
                # pipeline = SkyReelsPipeline.from_pretrained(model_dir, ...)
                print("SkyReels custom pipeline not yet integrated - using fallback")
                raise ImportError("SkyReels pipeline not integrated")
        except (ImportError, FileNotFoundError):
            # Strategy 2: Try diffusers pipeline
            try:
                from diffusers import DiffusionPipeline
                
                print("Loading model using diffusers...")
                pipeline = DiffusionPipeline.from_pretrained(
                    model_dir,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    device_map="auto" if self.enable_offload else None
                )
                
                if self.device == "cuda" and not self.enable_offload:
                    pipeline = pipeline.to(self.device)
                
                print("Model loaded successfully!")
                return pipeline
            except Exception as e:
                print(f"Error loading with diffusers: {e}")
                # Strategy 3: Manual loading (fallback)
                print("\n" + "="*60)
                print("IMPORTANT: SkyReels-V2 integration required!")
                print("="*60)
                print("Please:")
                print("1. Clone SkyReels-V2: git clone https://github.com/SkyworkAI/SkyReels-V2.git")
                print("2. Study their inference code in skyreels_v2_infer/")
                print("3. Adapt utils/model_loader.py with actual loading code")
                print("4. See INTEGRATION_NOTES.md for details")
                print("="*60)
                raise
    
    def get_i2v_model(self, model_path: Optional[str] = None, hf_id: Optional[str] = None):
        """
        Load Image-to-Video model
        
        Args:
            model_path: Local path to model (if None, downloads from HF)
            hf_id: Hugging Face model ID
            
        Returns:
            Loaded model pipeline
        """
        # Determine model path
        if model_path and Path(model_path).exists():
            model_dir = model_path
        else:
            # Use config or default
            hf_id = hf_id or self.config.get("model", {}).get("i2v_hf_id",
                "Skywork/SkyReels-V2-I2V-14B-720P")
            
            # Check if model already exists locally
            local_dir = Path("./models/i2v")
            if local_dir.exists() and any(local_dir.iterdir()):
                print(f"Using existing I2V model from: {local_dir}")
                model_dir = str(local_dir)
            else:
                # Download from Hugging Face
                print(f"Downloading I2V model from Hugging Face: {hf_id}")
                print("This may take a while...")
                model_dir = snapshot_download(
                    repo_id=hf_id,
                    local_dir=str(local_dir),
                    local_dir_use_symlinks=False
                )
        
        # Try multiple loading strategies (same as T2V)
        try:
            import sys
            skyreels_path = Path("./skyreels_v2_infer")
            if not skyreels_path.exists():
                skyreels_path = Path("../SkyReels-V2/skyreels_v2_infer")
            
            if skyreels_path.exists():
                sys.path.insert(0, str(skyreels_path.parent))
                print("SkyReels custom pipeline not yet integrated - using fallback")
                raise ImportError("SkyReels pipeline not integrated")
        except (ImportError, FileNotFoundError):
            try:
                from diffusers import DiffusionPipeline
                
                print("Loading I2V model using diffusers...")
                pipeline = DiffusionPipeline.from_pretrained(
                    model_dir,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    device_map="auto" if self.enable_offload else None
                )
                
                if self.device == "cuda" and not self.enable_offload:
                    pipeline = pipeline.to(self.device)
                
                print("I2V model loaded successfully!")
                return pipeline
            except Exception as e:
                print(f"Error loading I2V model: {e}")
                print("\n" + "="*60)
                print("IMPORTANT: SkyReels-V2 integration required!")
                print("See INTEGRATION_NOTES.md for integration steps")
                print("="*60)
                raise
    
    def check_gpu_availability(self) -> Dict:
        """Check GPU availability and memory"""
        info = {
            "cuda_available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "current_device": None,
            "device_name": None,
            "total_memory": None,
            "allocated_memory": None
        }
        
        if torch.cuda.is_available():
            info["current_device"] = torch.cuda.current_device()
            info["device_name"] = torch.cuda.get_device_name(0)
            info["total_memory"] = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
            info["allocated_memory"] = torch.cuda.memory_allocated(0) / (1024**3)  # GB
        
        return info

