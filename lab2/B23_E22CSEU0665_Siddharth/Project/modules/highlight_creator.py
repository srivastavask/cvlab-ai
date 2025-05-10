import numpy as np
import logging
import os
import json
import cv2
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HighlightCreator:
    """Creates highlight segments from videos"""
    
    def __init__(self):
        """Initialize highlight creator with forced segment creation"""
        logger.info("Highlight creator initialized with FORCED MODE")
    
    def create_highlights(self, video_path, scenes=None, audio_highlights=None, 
                          action_highlights=None, target_duration=60):
        """
        FORCED highlight creation - ignores inputs and creates highlights directly
        
        Args:
            video_path: Path to original video
            scenes, audio_highlights, action_highlights: Ignored
            target_duration: Target duration in seconds
            
        Returns:
            List of selected highlight segments as (start_time, end_time) tuples
        """
        # Get video duration using OpenCV
        duration = self._get_video_duration(video_path)
        if duration <= 0:
            logger.error(f"Could not determine video duration for {video_path}")
            # Return empty to trigger fallback
            return []
            
        logger.info(f"FORCED HIGHLIGHT MODE: Video duration: {duration}s, Target: {target_duration}s")
        
        # HARD LIMIT: Ensure target is at most 40% of video duration
        target_duration = min(target_duration, duration * 0.4)
        
        # Force short segments
        segments = []
        
        # If video is very short, just take first half
        if duration <= 20:
            segments = [(0, duration / 2)]
        else:
            # Create 5-10 second segments at different parts of the video
            segment_points = [0.1, 0.25, 0.4, 0.6, 0.75, 0.9]  # Percentage points
            segment_duration = min(8.0, target_duration / len(segment_points))
            
            # Create segments distributed across the video
            current_total = 0
            for point in segment_points:
                if current_total >= target_duration:
                    break
                
                center = point * duration
                start = max(0, center - (segment_duration / 2))
                end = min(duration, start + segment_duration)
                
                # Don't add if too close to previous segment
                if segments and start - segments[-1][1] < 2.0:
                    continue
                
                segments.append((start, end))
                current_total += segment_duration
        
        # Debug log the segments clearly
        for i, (start, end) in enumerate(segments):
            logger.info(f"FORCED SEGMENT {i+1}: {start:.1f}s to {end:.1f}s (duration: {end-start:.1f}s)")
        
        # Save debug metadata
        self._save_debug_metadata(segments, video_path)
        
        logger.info(f"Created {len(segments)} FORCED segments, total duration: {sum(end-start for start,end in segments):.1f}s")
        return segments
    
    def _get_video_duration(self, video_path):
        """Get video duration using OpenCV"""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return 0
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if fps > 0 and frame_count > 0:
                duration = frame_count / fps
            else:
                duration = 0
                
            cap.release()
            return duration
        except Exception as e:
            logger.error(f"Error getting video duration: {str(e)}")
            return 0
    
    def _save_debug_metadata(self, segments, video_path):
        """Save debug metadata for visualization"""
        try:
            os.makedirs("output", exist_ok=True)
            
            # Extract filename without extension
            video_name = os.path.basename(video_path)
            video_name = os.path.splitext(video_name)[0]
            
            metadata = {
                "video": video_name,
                "original_duration": self._get_video_duration(video_path),
                "highlights": [
                    {
                        "start": start,
                        "end": end,
                        "duration": end - start,
                        "type": "forced"
                    }
                    for start, end in segments
                ]
            }
            
            with open(f"output/{video_name}_debug.json", "w") as f:
                json.dump(metadata, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save debug metadata: {str(e)}")