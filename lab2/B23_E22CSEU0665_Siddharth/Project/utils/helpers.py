import os
import cv2
import numpy as np
import subprocess
import json
import logging
from datetime import timedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def format_time(seconds):
    """
    Format seconds to HH:MM:SS string format
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    if seconds is None:
        return "00:00"
        
    td = timedelta(seconds=seconds)
    hours, remainder = divmod(td.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    else:
        return f"{minutes:02d}:{seconds:02d}"

def get_video_info(video_path):
    """
    Get basic information about a video file
    
    Args:
        video_path: Path to video file
        
    Returns:
        Dictionary with video properties
    """
    if not os.path.exists(video_path):
        logger.error(f"Video file not found: {video_path}")
        return None
    
    try:
        # Try FFprobe first (more accurate)
        info = get_video_info_ffprobe(video_path)
        if info:
            return info
            
        # Fall back to OpenCV
        return get_video_info_opencv(video_path)
        
    except Exception as e:
        logger.error(f"Error getting video info: {str(e)}")
        return None

def get_video_info_ffprobe(video_path):
    """
    Get video information using FFprobe
    
    Args:
        video_path: Path to video file
        
    Returns:
        Dictionary with video properties
    """
    try:
        cmd = [
            "ffprobe",
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            video_path
        ]
        
        result = subprocess.run(
            cmd, 
            check=True, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE
        )
        
        info = json.loads(result.stdout)
        
        # Find video stream
        video_stream = None
        for stream in info.get("streams", []):
            if stream.get("codec_type") == "video":
                video_stream = stream
                break
        
        if not video_stream:
            return None
        
        # Parse frame rate
        fps = 30.0
        if "r_frame_rate" in video_stream:
            fps_parts = video_stream["r_frame_rate"].split('/')
            if len(fps_parts) == 2 and int(fps_parts[1]) != 0:
                fps = int(fps_parts[0]) / int(fps_parts[1])
        
        # Get duration
        duration = float(info.get("format", {}).get("duration", 0))
        
        return {
            "width": int(video_stream.get("width", 1280)),
            "height": int(video_stream.get("height", 720)),
            "duration": duration,
            "fps": fps,
            "codec": video_stream.get("codec_name"),
            "duration_formatted": format_time(duration)
        }
    
    except Exception as e:
        logger.error(f"FFprobe error: {str(e)}")
        return None

def get_video_info_opencv(video_path):
    """
    Get video information using OpenCV as fallback
    
    Args:
        video_path: Path to video file
        
    Returns:
        Dictionary with video properties
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Could not open video: {video_path}")
            return None
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        
        cap.release()
        
        return {
            'width': width,
            'height': height,
            'fps': fps,
            'frame_count': frame_count,
            'duration': duration,
            'duration_formatted': format_time(duration)
        }
    except Exception as e:
        logger.error(f"OpenCV error: {str(e)}")
        return None

def create_thumbnail(video_path, output_path=None, time_pos=None):
    """
    Create a thumbnail from a video.
    
    Args:
        video_path: Path to the video file
        output_path: Where to save the thumbnail, defaults to same directory
        time_pos: Position in seconds to capture thumbnail, defaults to middle
        
    Returns:
        Path to the generated thumbnail file
    """
    try:
        if not os.path.exists(video_path):
            logger.error(f"Video file not found: {video_path}")
            return None
            
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Could not open video: {video_path}")
            return None
            
        # Determine position to capture
        if time_pos is None:
            # Use middle of video
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            time_pos = (frame_count / fps) / 2 if frame_count > 0 and fps > 0 else 0
            
        # Set position and read frame
        cap.set(cv2.CAP_PROP_POS_MSEC, time_pos * 1000)
        ret, frame = cap.read()
        
        if not ret:
            logger.error(f"Could not read frame at position {time_pos}s")
            cap.release()
            return None
            
        # Generate output path if not provided
        if output_path is None:
            base_name = os.path.basename(video_path)
            name_without_ext = os.path.splitext(base_name)[0]
            output_path = os.path.join(os.path.dirname(video_path), 
                                      f"{name_without_ext}_thumbnail.jpg")
            
        # Save thumbnail
        cv2.imwrite(output_path, frame)
        cap.release()
        
        return output_path
        
    except Exception as e:
        logger.error(f"Error creating thumbnail: {str(e)}")
        return None