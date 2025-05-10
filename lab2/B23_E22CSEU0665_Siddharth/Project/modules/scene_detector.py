import cv2
import numpy as np
import logging
import os
import time
import subprocess
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SceneDetector:
    """
    Detects scene changes in videos using either PySceneDetect (if available)
    or falls back to OpenCV implementation
    """
    
    def __init__(self, threshold=0.5, min_scene_duration=1.0):
        """
        Initialize scene detector with sensitivity threshold
        
        Args:
            threshold: Detection threshold (0.0 to 1.0)
                Lower values detect more scenes (more sensitive)
            min_scene_duration: Minimum scene duration in seconds
        """
        self.threshold = threshold
        self.min_scene_duration = min_scene_duration
        
        # Check for PySceneDetect availability
        self.has_scenedetect = self._check_scenedetect()
        
        if self.has_scenedetect:
            logger.info(f"Scene detector initialized with PySceneDetect, threshold {threshold}")
        else:
            logger.info(f"Scene detector initialized with OpenCV fallback, threshold {threshold}")
    
    def _check_scenedetect(self):
        """
        Check if PySceneDetect is available
        
        Returns:
            bool: True if PySceneDetect is available, False otherwise
        """
        try:
            import scenedetect
            logger.info("Using PySceneDetect for scene detection")
            return True
        except ImportError:
            logger.info("PySceneDetect not available, using OpenCV fallback")
            return False
    
    def detect_scenes(self, video_path):
        """
        Detect scene changes in a video
        
        Args:
            video_path: Path to the video file
            
        Returns:
            List of (start_time, end_time) tuples in seconds
        """
        if not os.path.exists(video_path):
            logger.error(f"Video file not found: {video_path}")
            return []
            
        try:
            # Try PySceneDetect first if available
            if self.has_scenedetect:
                try:
                    return self._detect_scenes_pyscenedetect(video_path)
                except Exception as e:
                    logger.error(f"PySceneDetect error: {str(e)}")
                    logger.info("Falling back to OpenCV implementation")
            
            # Fall back to OpenCV implementation
            return self._detect_scenes_opencv(video_path)
            
        except Exception as e:
            logger.error(f"Error detecting scenes: {str(e)}")
            return []
    
    def _detect_scenes_pyscenedetect(self, video_path):
        """
        Detect scenes using PySceneDetect library
        
        Args:
            video_path: Path to video file
            
        Returns:
            List of (start_time, end_time) tuples
        """
        from scenedetect import VideoManager, SceneManager, StatsManager
        from scenedetect.detectors import ContentDetector
        
        # Adjust threshold for ContentDetector (inverse relationship)
        # Lower threshold value = more sensitive detection (more scenes)
        # Map our 0-1 threshold to ContentDetector's typical range (10-40)
        content_threshold = 35 - (self.threshold * 25)
        
        # Create video manager and scene manager
        video_manager = VideoManager([video_path])
        stats_manager = StatsManager()
        scene_manager = SceneManager(stats_manager)
        
        # Add content detector
        scene_manager.add_detector(
            ContentDetector(
                threshold=content_threshold,
                min_scene_len=int(self.min_scene_duration * 30)  # Assuming 30fps
            )
        )
        
        # Improve performance with downscaling
        video_manager.set_downscale_factor()
        
        # Start detection process
        video_manager.start()
        scene_manager.detect_scenes(frame_source=video_manager)
        
        # Get scene list and calculate time values
        scene_list = scene_manager.get_scene_list()
        fps = video_manager.get_framerate()
        
        # Convert frame numbers to timestamps
        scenes = []
        for scene in scene_list:
            start_frame, end_frame = scene
            start_time = start_frame.get_seconds()
            end_time = end_frame.get_seconds()
            
            # Skip very short scenes
            if end_time - start_time >= self.min_scene_duration:
                scenes.append((start_time, end_time))
        
        # Clean up
        video_manager.release()
        
        logger.info(f"PySceneDetect found {len(scenes)} scenes")
        return scenes
    
    def _detect_scenes_opencv(self, video_path):
        """
        Detect scenes using OpenCV (fallback method)
        
        Args:
            video_path: Path to video file
            
        Returns:
            List of (start_time, end_time) tuples
        """
        logger.info(f"Detecting scenes in {video_path} with OpenCV")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Could not open video {video_path}")
            return []
            
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30  # Default assumption
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        logger.info(f"Video properties: {width}x{height}, {fps} fps, {total_frames} frames")
        
        # Initialize variables
        scenes = []
        prev_frame = None
        curr_scene_start = 0
        frame_count = 0
        
        # Adjust sensitivity based on threshold
        # Higher threshold means less sensitive (fewer scenes)
        diff_threshold = 35.0 - (self.threshold * 25.0)
        
        # Sample rate (process every Nth frame for performance)
        sample_rate = max(1, int(fps / 4))  # Process 4 frames per second
        
        start_time = time.time()
        
        # Process frames
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Process at sample rate
            if frame_count % sample_rate == 0:
                # Resize large frames for performance
                if width > 640 or height > 480:
                    scale_factor = min(640 / width, 480 / height)
                    new_width = int(width * scale_factor)
                    new_height = int(height * scale_factor)
                    frame = cv2.resize(frame, (new_width, new_height))
                
                # Convert to grayscale and blur to reduce noise
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, (21, 21), 0)
                
                if prev_frame is not None:
                    # Calculate difference between frames
                    frame_diff = cv2.absdiff(prev_frame, gray)
                    mean_diff = np.mean(frame_diff)
                    
                    # Progress logging
                    if frame_count % (sample_rate * 100) == 0:
                        progress = (frame_count / total_frames) * 100 if total_frames > 0 else 0
                        elapsed = time.time() - start_time
                        logger.debug(f"Processing: {progress:.1f}% complete, frame {frame_count}")
                    
                    # If difference exceeds threshold, it's a new scene
                    if mean_diff > diff_threshold:
                        scene_end_time = frame_count / fps
                        scene_duration = scene_end_time - curr_scene_start
                        
                        # Only add if scene is long enough
                        if scene_duration >= self.min_scene_duration:
                            scenes.append((curr_scene_start, scene_end_time))
                            
                        # Start a new scene
                        curr_scene_start = scene_end_time
                
                # Update previous frame
                prev_frame = gray
                
            frame_count += 1
        
        # Add final scene if needed
        final_time = frame_count / fps
        if final_time - curr_scene_start >= self.min_scene_duration:
            scenes.append((curr_scene_start, final_time))
            
        # Clean up
        cap.release()
        
        # Merge very short adjacent scenes
        scenes = self._merge_short_scenes(scenes)
        
        logger.info(f"OpenCV method found {len(scenes)} scenes in {time.time() - start_time:.1f} seconds")
        return scenes
    
    def _merge_short_scenes(self, scenes, max_gap=1.5):
        """
        Merge scenes that are very close together
        
        Args:
            scenes: List of (start_time, end_time) tuples
            max_gap: Maximum gap in seconds between scenes to merge
            
        Returns:
            List of merged scenes
        """
        if not scenes or len(scenes) < 2:
            return scenes
            
        # Sort by start time (should already be sorted, but to be safe)
        sorted_scenes = sorted(scenes, key=lambda x: x[0])
        
        # Merge adjacent scenes with small gaps
        merged = [sorted_scenes[0]]
        
        for current in sorted_scenes[1:]:
            previous = merged[-1]
            
            # If gap between scenes is small, merge them
            if current[0] - previous[1] <= max_gap:
                merged[-1] = (previous[0], current[1])
            else:
                merged.append(current)
                
        return merged
    
    def get_scene_thumbnails(self, video_path, output_dir=None, max_thumbnails=10):
        """
        Generate thumbnail images for each detected scene
        
        Args:
            video_path: Path to video file
            output_dir: Directory to save thumbnails (default: same as video)
            max_thumbnails: Maximum number of thumbnails to generate
            
        Returns:
            List of paths to thumbnail images
        """
        try:
            # Detect scenes
            scenes = self.detect_scenes(video_path)
            
            # Limit number of thumbnails
            if len(scenes) > max_thumbnails:
                # Select scenes evenly distributed throughout the video
                step = len(scenes) / max_thumbnails
                selected_indices = [int(i * step) for i in range(max_thumbnails)]
                scenes = [scenes[i] for i in selected_indices]
            
            # Set output directory
            if output_dir is None:
                output_dir = os.path.dirname(video_path)
            
            os.makedirs(output_dir, exist_ok=True)
            
            # Extract base name from video path
            base_name = os.path.basename(video_path)
            name_without_ext = os.path.splitext(base_name)[0]
            
            # Open video
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return []
                
            thumbnails = []
            
            # Generate thumbnail for each scene
            for i, (start_time, _) in enumerate(scenes):
                # Set position to the middle of the scene
                cap.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000)
                
                # Read frame
                ret, frame = cap.read()
                if not ret:
                    continue
                    
                # Save thumbnail
                thumbnail_path = os.path.join(output_dir, f"{name_without_ext}_scene_{i}.jpg")
                cv2.imwrite(thumbnail_path, frame)
                thumbnails.append(thumbnail_path)
            
            # Clean up
            cap.release()
            
            return thumbnails
            
        except Exception as e:
            logger.error(f"Error generating scene thumbnails: {str(e)}")
            return []