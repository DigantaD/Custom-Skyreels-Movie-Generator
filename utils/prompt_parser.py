"""
Prompt Parser Module
Converts script text into structured video generation prompts
"""

import re
import json
from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class VideoSegment:
    """Represents a single video segment with its prompt and metadata"""
    segment_id: int
    text: str
    prompt: str
    duration: float
    camera_movement: Optional[str] = None
    scene_type: Optional[str] = None  # "opening", "product_showcase", "interaction", "closing"


class PromptParser:
    """Parses scripts into video generation prompts"""
    
    def __init__(self, style_modifiers: str = "cinematic, high quality, professional, 4K"):
        self.style_modifiers = style_modifiers
        self.camera_keywords = {
            "pan": "camera panning smoothly",
            "zoom": "camera zooming in",
            "wide": "wide angle shot",
            "close": "close-up shot"
        }
    
    def parse_script(self, script_text: str, default_duration: float = 3.0) -> List[VideoSegment]:
        """
        Parse a script into video segments
        
        Args:
            script_text: The full script text
            default_duration: Default duration for each segment in seconds
            
        Returns:
            List of VideoSegment objects
        """
        # Split script by paragraphs or explicit markers
        segments = self._split_script(script_text)
        
        video_segments = []
        for idx, segment_text in enumerate(segments):
            segment = self._create_segment(
                segment_id=idx + 1,
                text=segment_text,
                default_duration=default_duration
            )
            video_segments.append(segment)
        
        return video_segments
    
    def _split_script(self, script_text: str) -> List[str]:
        """Split script into logical segments"""
        # Remove extra whitespace
        script_text = re.sub(r'\s+', ' ', script_text.strip())
        
        # Split by double newlines or explicit markers like (Camera...)
        segments = []
        
        # First, identify camera action segments (in parentheses)
        pattern = r'\([^)]+\)'
        camera_actions = re.findall(pattern, script_text)
        
        # Split by paragraphs (double newline) or single newline
        parts = re.split(r'\n\n+|\n', script_text)
        
        current_segment = ""
        for part in parts:
            part = part.strip()
            if not part:
                continue
            
            # Check if this is a camera action
            if part.startswith('(') and part.endswith(')'):
                # Save previous segment if exists
                if current_segment:
                    segments.append(current_segment.strip())
                    current_segment = ""
                # Add camera action as separate segment
                segments.append(part)
            else:
                # Add to current segment
                if current_segment:
                    current_segment += " " + part
                else:
                    current_segment = part
        
        # Add last segment
        if current_segment:
            segments.append(current_segment.strip())
        
        # If no segments found, treat entire script as one segment
        if not segments:
            segments = [script_text]
        
        return segments
    
    def _create_segment(self, segment_id: int, text: str, default_duration: float) -> VideoSegment:
        """Create a VideoSegment from text"""
        # Detect camera movements
        camera_movement = self._detect_camera_movement(text)
        
        # Detect scene type
        scene_type = self._detect_scene_type(text)
        
        # Generate prompt
        prompt = self._generate_prompt(text, camera_movement, scene_type)
        
        # Estimate duration (rough estimate: 150 words per minute)
        word_count = len(text.split())
        estimated_duration = max(default_duration, (word_count / 150) * 60)
        
        return VideoSegment(
            segment_id=segment_id,
            text=text,
            prompt=prompt,
            duration=estimated_duration,
            camera_movement=camera_movement,
            scene_type=scene_type
        )
    
    def _detect_camera_movement(self, text: str) -> Optional[str]:
        """Detect camera movement from text"""
        text_lower = text.lower()
        
        if "pan" in text_lower or "pans" in text_lower:
            return "pan"
        elif "zoom" in text_lower:
            return "zoom"
        elif "wide" in text_lower or "wide shot" in text_lower:
            return "wide"
        elif "close" in text_lower or "close-up" in text_lower:
            return "close"
        
        return None
    
    def _detect_scene_type(self, text: str) -> Optional[str]:
        """Detect scene type from text"""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ["hey everyone", "welcome", "here at"]):
            return "opening"
        elif any(word in text_lower for word in ["shelves", "cookware", "serveware", "products"]):
            return "product_showcase"
        elif any(word in text_lower for word in ["picks up", "holding", "showing"]):
            return "interaction"
        elif any(word in text_lower for word in ["closes", "see you", "smiling", "final"]):
            return "closing"
        
        return None
    
    def _generate_prompt(self, text: str, camera_movement: Optional[str], scene_type: Optional[str]) -> str:
        """Generate a detailed prompt for video generation"""
        # Start with the main text content
        prompt_parts = [text]
        
        # Add visual description based on scene type
        if scene_type == "opening":
            prompt_parts.append("Professional influencer standing at store entrance, warm welcoming atmosphere")
        elif scene_type == "product_showcase":
            prompt_parts.append("Beautiful display of traditional Indian brass and copper cookware on shelves, well-lit store interior")
        elif scene_type == "interaction":
            prompt_parts.append("Close-up of hands holding traditional Indian brass cookware, showcasing craftsmanship")
        elif scene_type == "closing":
            prompt_parts.append("Wide shot of store interior with influencer, professional retail environment")
        
        # Add camera movement
        if camera_movement:
            camera_desc = self.camera_keywords.get(camera_movement, "")
            if camera_desc:
                prompt_parts.append(camera_desc)
        
        # Add style modifiers
        if self.style_modifiers:
            prompt_parts.append(self.style_modifiers)
        
        # Combine into final prompt
        final_prompt = ", ".join(prompt_parts)
        
        return final_prompt
    
    def save_segments(self, segments: List[VideoSegment], output_path: str):
        """Save segments to JSON file"""
        segments_dict = [
            {
                "segment_id": seg.segment_id,
                "text": seg.text,
                "prompt": seg.prompt,
                "duration": seg.duration,
                "camera_movement": seg.camera_movement,
                "scene_type": seg.scene_type
            }
            for seg in segments
        ]
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(segments_dict, f, indent=2, ensure_ascii=False)
    
    def load_segments(self, input_path: str) -> List[VideoSegment]:
        """Load segments from JSON file"""
        with open(input_path, 'r', encoding='utf-8') as f:
            segments_dict = json.load(f)
        
        segments = [
            VideoSegment(
                segment_id=seg["segment_id"],
                text=seg["text"],
                prompt=seg["prompt"],
                duration=seg["duration"],
                camera_movement=seg.get("camera_movement"),
                scene_type=seg.get("scene_type")
            )
            for seg in segments_dict
        ]
        
        return segments

