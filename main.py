"""
Main Pipeline Orchestrator
Coordinates the entire video generation pipeline
"""

import argparse
from pathlib import Path
import json
from typing import Optional

from utils.prompt_parser import PromptParser
from scripts.segment_generator import SegmentGenerator
from scripts.video_stitcher import VideoStitcher
from scripts.post_processor import PostProcessor


class VideoGenerationPipeline:
    """Main pipeline for video generation"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or "configs/model_config.yaml"
        self.prompt_parser = PromptParser()
        self.segment_generator = SegmentGenerator(self.config_path)
        self.video_stitcher = VideoStitcher(self.config_path)
        self.post_processor = PostProcessor()
        
        # Create output directories
        Path("outputs/segments").mkdir(parents=True, exist_ok=True)
        Path("outputs/final").mkdir(parents=True, exist_ok=True)
    
    def generate_from_script(self, script_path: str, output_path: str, 
                            mode: str = "t2v", start_image: Optional[str] = None):
        """
        Generate video from script file
        
        Args:
            script_path: Path to script text file
            output_path: Path for output video
            mode: "t2v" or "i2v"
            start_image: Path to starting image (required for i2v mode)
        """
        # Read script
        with open(script_path, 'r', encoding='utf-8') as f:
            script_text = f.read()
        
        print("=" * 60)
        print("SkyReels Movie Generator - Pipeline Started")
        print("=" * 60)
        print(f"Script: {script_path}")
        print(f"Mode: {mode.upper()}")
        print(f"Output: {output_path}")
        print("=" * 60)
        
        # Parse script into segments
        print("\n[Step 1/4] Parsing script into segments...")
        segments = self.prompt_parser.parse_script(script_text)
        print(f"  → Found {len(segments)} segments")
        
        # Save segments for reference
        segments_json = Path("outputs/segments/segments.json")
        self.prompt_parser.save_segments(segments, str(segments_json))
        print(f"  → Segments saved to: {segments_json}")
        
        # Generate video segments
        print(f"\n[Step 2/4] Generating {len(segments)} video segments...")
        segment_paths = []
        
        for idx, segment in enumerate(segments):
            print(f"\n  Segment {idx+1}/{len(segments)}:")
            print(f"    Text: {segment.text[:80]}...")
            print(f"    Prompt: {segment.prompt[:80]}...")
            
            segment_output = Path("outputs/segments") / f"segment_{idx+1:03d}.mp4"
            
            try:
                if mode == "t2v":
                    self.segment_generator.generate_t2v_segment(
                        prompt=segment.prompt,
                        output_path=str(segment_output),
                        duration=segment.duration
                    )
                elif mode == "i2v":
                    if idx == 0 and start_image:
                        # Use provided start image for first segment
                        image_path = start_image
                    elif idx > 0:
                        # Use last frame of previous segment
                        from utils.video_utils import get_last_frame
                        prev_segment = segment_paths[-1]
                        image_path = f"outputs/segments/frame_{idx}.jpg"
                        get_last_frame(prev_segment, image_path)
                    else:
                        raise ValueError("Start image required for I2V mode")
                    
                    self.segment_generator.generate_i2v_segment(
                        prompt=segment.prompt,
                        image_path=image_path,
                        output_path=str(segment_output),
                        duration=segment.duration
                    )
                else:
                    raise ValueError(f"Invalid mode: {mode}")
                
                segment_paths.append(str(segment_output))
                print(f"    ✓ Generated: {segment_output}")
                
            except Exception as e:
                print(f"    ✗ Error generating segment {idx+1}: {e}")
                raise
        
        # Validate segments
        print(f"\n[Step 3/4] Validating segments...")
        validation = self.video_stitcher.validate_segments(segment_paths)
        if not validation["valid"]:
            print("  ⚠ Warning: Segment compatibility issues detected:")
            for issue in validation["issues"]:
                print(f"    - {issue}")
            print("  → Attempting to normalize segments...")
            # Normalize segments (resize to common resolution)
            target_res = validation["base_resolution"]
            normalized = self.post_processor.normalize_videos(
                segment_paths, target_res, validation["base_fps"],
                "outputs/segments/normalized"
            )
            segment_paths = normalized
        
        # Stitch segments
        print(f"\n[Step 4/4] Stitching segments into final video...")
        final_video = self.video_stitcher.stitch_segments(
            segment_paths, output_path
        )
        
        print("\n" + "=" * 60)
        print("Pipeline Complete!")
        print("=" * 60)
        print(f"Final video: {final_video}")
        print(f"Segments: {len(segment_paths)}")
        print("=" * 60)
        
        return final_video


def main():
    parser = argparse.ArgumentParser(description="SkyReels Movie Generator")
    
    parser.add_argument(
        "--script",
        type=str,
        required=True,
        help="Path to script text file"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/final/generated_video.mp4",
        help="Output video path"
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        choices=["t2v", "i2v"],
        default="t2v",
        help="Generation mode: t2v (text-to-video) or i2v (image-to-video)"
    )
    
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Starting image path (required for i2v mode)"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="configs/model_config.yaml",
        help="Path to configuration file"
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not Path(args.script).exists():
        print(f"Error: Script file not found: {args.script}")
        return
    
    if args.mode == "i2v" and not args.image:
        print("Error: --image required for i2v mode")
        return
    
    if args.mode == "i2v" and args.image and not Path(args.image).exists():
        print(f"Error: Image file not found: {args.image}")
        return
    
    # Create and run pipeline
    pipeline = VideoGenerationPipeline(config_path=args.config)
    
    try:
        pipeline.generate_from_script(
            script_path=args.script,
            output_path=args.output,
            mode=args.mode,
            start_image=args.image
        )
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

