import os
import subprocess
import logging
import json
import tempfile
import cv2
import numpy as np
from datetime import datetime
import shutil

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VideoEditor:
    """
    Creates highlight videos using FFmpeg with smooth transitions and audio fades
    """
    
    def __init__(self):
        """Initialize video editor with FFmpeg check"""
        self.has_ffmpeg = self._check_ffmpeg()
        if self.has_ffmpeg:
            logger.info("Video editor initialized with FFmpeg backend")
        else:
            logger.warning("FFmpeg not found! Please install FFmpeg for video editing.")
    
    def _check_ffmpeg(self):
        """Check if FFmpeg is installed"""
        try:
            subprocess.run(
                ["ffmpeg", "-version"], 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                check=True
            )
            return True
        except (subprocess.SubprocessError, FileNotFoundError):
            return False
    
    def create_highlight_video(self, video_path, highlights, transition_type="Fade", 
                               output_dir="output", add_intro=True, resolution="original"):
        """
        Create a highlight video from selected segments using FFmpeg with enhanced transitions.
        
        Args:
            video_path: Path to original video
            highlights: List of (start_time, end_time) tuples
            transition_type: Type of transition ("Fade", "Cut", "Dissolve", "None")
            output_dir: Directory for output
            add_intro: Whether to add intro title
            resolution: Output resolution
            
        Returns:
            Path to output video or None on failure
        """
        if not self.has_ffmpeg:
            logger.error("FFmpeg not available. Cannot create highlight video.")
            return None
        
        # Debug log the highlights
        logger.info(f"Creating highlights from {len(highlights)} segments:")
        for i, (start, end) in enumerate(highlights):
            logger.info(f"  Segment {i}: {start:.2f}s - {end:.2f}s (duration: {end-start:.2f}s)")
        
        # Validate highlights
        if not highlights:
            logger.error("No highlights to process")
            return None
        
        # Create output dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Create temporary directory for processing files
        temp_dir = tempfile.mkdtemp()
        logger.info(f"Using temporary directory: {temp_dir}")
        
        try:
            # Generate output filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"highlights_{timestamp}.mp4"
            output_path = os.path.join(output_dir, output_filename)
            
            # Get video info for resolution
            video_info = self._get_video_info(video_path)
            if not video_info:
                logger.error("Could not determine video info")
                return None
                
            # Get target resolution
            target_resolution = self._get_target_resolution(resolution, video_info)
            
            # Extract segments with overlapping for transitions
            segment_files = []
            
            # Add intro if requested
            if add_intro:
                intro_path = self._create_intro(temp_dir, target_resolution, video_info)
                if intro_path:
                    segment_files.append(intro_path)
            
            # Add padding for transitions (0.25 seconds before and after each segment - REDUCED)
            padding = 0.25 if transition_type in ["Fade", "Dissolve"] else 0.0
            
            # Extract each segment with audio fades
            for i, (start, end) in enumerate(highlights):
                # Skip invalid segments
                if end <= start:
                    continue
                
                # Add padding for transitions (ensuring we don't go before start of video)
                padded_start = max(0, start - padding)
                padded_end = end + padding
                
                # Extract segment with audio fades
                segment_path = self._extract_segment_with_fades(
                    video_path, padded_start, padded_end, start, end, i, temp_dir, target_resolution
                )
                
                if segment_path:
                    segment_files.append(segment_path)
                    
            if not segment_files:
                logger.error("No valid segments extracted")
                return None
                
            # Create complex filter for advanced transitions
            if transition_type in ["Fade", "Dissolve"] and len(segment_files) > 1:
                output_path = self._create_video_with_transitions(
                    segment_files, transition_type, output_path, temp_dir
                )
            else:
                # Simple concatenation for "Cut" or "None" transitions
                output_path = self._concatenate_segments(segment_files, output_path, temp_dir)
            
            # Clean up temporary files
            try:
                for file in segment_files:
                    if os.path.exists(file):
                        os.remove(file)
                shutil.rmtree(temp_dir, ignore_errors=True)
            except Exception as e:
                logger.error(f"Error cleaning up temp files: {str(e)}")
            
            if output_path and os.path.exists(output_path):
                logger.info(f"Highlight video created at {output_path}")
                return output_path
            else:
                logger.error("Failed to create highlight video")
                return None
                
        except Exception as e:
            logger.error(f"Error creating highlight video: {str(e)}")
            shutil.rmtree(temp_dir, ignore_errors=True)
            return None
    
    def _extract_segment_with_fades(self, video_path, padded_start, padded_end, 
                                   actual_start, actual_end, index, temp_dir, target_resolution):
        """Extract segment with audio fades at beginning and end"""
        try:
            segment_path = os.path.join(temp_dir, f"segment_{index:03d}.mp4")
            
            # Calculate actual segment duration and fade durations
            segment_duration = padded_end - padded_start
            actual_duration = actual_end - actual_start
            
            # Calculate fade positions relative to padded segment
            fade_in_start = actual_start - padded_start
            fade_out_start = fade_in_start + actual_duration - 0.25  # REDUCED: Start fade 0.25s before end
            
            # Ensure fade durations are valid - REDUCED transition duration
            fade_in_duration = min(0.25, actual_duration / 5)
            fade_out_duration = min(0.25, actual_duration / 5)
            
            # Create video filter with fades
            video_filter = []
            
            # Add scaling if needed
            if target_resolution != "original":
                video_filter.append(f"scale={target_resolution}")
                
            # Add video fades
            video_filter.append(f"fade=in:st={fade_in_start}:d={fade_in_duration}")
            video_filter.append(f"fade=out:st={fade_out_start}:d={fade_out_duration}")
            
            # Join filters
            vf = ",".join(video_filter)
            
            # Create audio filter with fades
            audio_filter = f"afade=in:st={fade_in_start}:d={fade_in_duration},afade=out:st={fade_out_start}:d={fade_out_duration}"
            
            # Extract segment with fades
            cmd = [
                "ffmpeg", "-y",
                "-ss", str(padded_start),
                "-i", video_path,
                "-t", str(segment_duration),
                "-vf", vf,
                "-af", audio_filter,
                "-c:v", "libx264",
                "-c:a", "aac",
                "-b:a", "192k",
                segment_path
            ]
            
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            if os.path.exists(segment_path) and os.path.getsize(segment_path) > 1000:
                logger.info(f"Successfully extracted segment {index} with fades")
                return segment_path
            else:
                logger.error(f"Failed to extract segment {index}")
                return None
                
        except Exception as e:
            logger.error(f"Error extracting segment {index}: {str(e)}")
            return None
    
    def _create_video_with_transitions(self, segment_files, transition_type, output_path, temp_dir):
        """Create video with smooth transitions between segments"""
        try:
            # REDUCED: Use shorter transition duration (0.25 seconds instead of 0.5)
            transition_duration = 0.25
            
            if len(segment_files) <= 1:
                # No transitions needed for single file
                if len(segment_files) == 1:
                    shutil.copy(segment_files[0], output_path)
                return output_path
            
            # Choose transition effect
            if transition_type == "Dissolve":
                xfade_effect = "dissolve"
            else:  # default to crossfade
                xfade_effect = "fade"
            
            # Create complex filter
            filter_complex = []
            
            # First segment doesn't need input transition
            filter_complex.append(f"[0:v]setpts=PTS-STARTPTS[v0];")
            filter_complex.append(f"[0:a]asetpts=PTS-STARTPTS[a0];")
            
            # Process each transition
            for i in range(1, len(segment_files)):
                # Video transition
                filter_complex.append(f"[{i}:v]setpts=PTS-STARTPTS[v{i}];")
                filter_complex.append(f"[v{i-1}][v{i}]xfade=transition={xfade_effect}:duration={transition_duration}:offset={i*5-transition_duration}[xv{i}];")
                
                # Audio crossfade
                filter_complex.append(f"[{i}:a]asetpts=PTS-STARTPTS[a{i}];")
                filter_complex.append(f"[a{i-1}][a{i}]acrossfade=d={transition_duration}[xa{i}];")
            
            # Create input file list
            input_list = []
            for file in segment_files:
                input_list.extend(["-i", file])
            
            # Create filter file
            filter_file = os.path.join(temp_dir, "filter_complex.txt")
            with open(filter_file, "w") as f:
                f.write("".join(filter_complex))
            
            # Execute FFmpeg command
            cmd = [
                "ffmpeg", "-y",
                *input_list,
                "-filter_complex_script", filter_file,
                "-map", f"[xv{len(segment_files)-1}]",
                "-map", f"[xa{len(segment_files)-1}]",
                "-c:v", "libx264",
                "-c:a", "aac",
                "-b:a", "192k",
                output_path
            ]
            
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error creating video with transitions: {str(e)}")
            
            # Fallback to simple concatenation
            logger.warning("Falling back to simple concatenation")
            return self._concatenate_segments(segment_files, output_path, temp_dir)
    
    def _concatenate_segments(self, segment_files, output_path, temp_dir):
        """Simple concatenation of segments"""
        try:
            # Create concat file
            concat_file = os.path.join(temp_dir, "concat.txt")
            with open(concat_file, "w") as f:
                for file in segment_files:
                    # Use escaped absolute paths
                    abs_path = os.path.abspath(file).replace('\\', '\\\\')
                    f.write(f"file '{abs_path}'\n")
            
            # Run ffmpeg concat
            cmd = [
                "ffmpeg", "-y",
                "-f", "concat",
                "-safe", "0",
                "-i", concat_file,
                "-c", "copy",
                output_path
            ]
            
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error concatenating segments: {str(e)}")
            return None
    
    def _create_intro(self, temp_dir, target_resolution, video_info):
        """Create a simple but reliable intro with HIGHLIGHTS text"""
        try:
            intro_path = os.path.join(temp_dir, "intro.mp4")
            
            # Find video dimensions
            width = 1280
            height = 720
            
            # Get dimensions from original video or resolution setting
            if video_info:
                for stream in video_info.get("streams", []):
                    if stream.get("codec_type") == "video":
                        width = int(stream.get("width", 1280))
                        height = int(stream.get("height", 720))
                        break
            
            if target_resolution != "original":
                try:
                    width, height = map(int, target_resolution.split(':'))
                except:
                    pass
            
            # Create a single intro file with both the HIGHLIGHTS text and animated intro
            # Use a simpler, more reliable approach
            cmd = [
                "ffmpeg", "-y",
                "-f", "lavfi", 
                "-i", f"color=c=black:s={width}x{height}:d=4.5",  # 4.5 seconds total
                "-vf", (
                    # First show HIGHLIGHTS text (0-2s)
                    f"drawtext=text='HIGHLIGHTS':fontcolor=white:fontsize={height/6}:x=(w-text_w)/2:y=(h-text_h)/2:"
                    f"enable='between(t,0,2)':alpha='if(lt(t,0.3),t/0.3,if(lt(t,1.7),1,1-(t-1.7)/0.3))',"
                    
                    # Then show Video Highlights text (2-4.5s)
                    f"drawtext=text='Video Highlights':fontcolor=white:fontsize={height/10}:x=(w-text_w)/2:y=(h-text_h)/2-40:"
                    f"enable='between(t,2,4.5)':alpha='if(lt(t,2.3),(t-2)/0.3,if(lt(t,4),1,1-(t-4)/0.5))',"
                    
                    # Add date text based on user's provided date
                    f"drawtext=text='Created on May 08, 2025':fontcolor=white@0.8:fontsize={height/20}:"
                    f"x=(w-text_w)/2:y=(h-text_h)/2+40:enable='between(t,2.3,4.5)':alpha='if(lt(t,2.6),(t-2.3)/0.3,if(lt(t,4),1,1-(t-4)/0.5))',"
                    
                    # Add user name
                    f"drawtext=text='By PiyushTiwari10':fontcolor=white@0.7:fontsize={height/25}:"
                    f"x=(w-text_w)/2:y=(h-text_h)/2+80:enable='between(t,2.6,4.5)':alpha='if(lt(t,2.9),(t-2.6)/0.3,if(lt(t,4),1,1-(t-4)/0.5))'"
                ),
                # Add audio
                "-f", "lavfi",
                "-i", "aevalsrc=0.03*sin(440*2*PI*t):s=44100:d=4.5",
                "-af", "afade=in:st=0:d=0.5,afade=out:st=4:d=0.5",
                "-c:v", "libx264", 
                "-c:a", "aac",
                "-b:a", "192k",
                "-shortest",
                intro_path
            ]
            
            # Run command
            logger.info("Creating intro with HIGHLIGHTS screen...")
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # If the complex command fails, try a much simpler fallback
            if not (os.path.exists(intro_path) and os.path.getsize(intro_path) > 1000):
                logger.warning("Complex intro creation failed. Trying simple fallback...")
                
                # Create very simple intro with just text
                simple_cmd = [
                    "ffmpeg", "-y",
                    "-f", "lavfi", 
                    "-i", f"color=c=black:s={width}x{height}:d=3",
                    "-vf", f"drawtext=text='HIGHLIGHTS':fontcolor=white:fontsize=72:x=(w-text_w)/2:y=(h-text_h)/2",
                    "-f", "lavfi",
                    "-i", "anullsrc=r=44100:cl=stereo",
                    "-t", "3",
                    "-c:v", "libx264", 
                    "-c:a", "aac",
                    intro_path
                ]
                
                subprocess.run(simple_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            if os.path.exists(intro_path) and os.path.getsize(intro_path) > 1000:
                logger.info("Created intro successfully")
                return intro_path
                
            return None
        
        except Exception as e:
            logger.error(f"Error creating intro: {str(e)}")
            return None
    
    def _get_video_info(self, video_path):
        """Get video metadata using FFprobe"""
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
            return info
            
        except Exception as e:
            logger.error(f"Error getting video info: {str(e)}")
            return None
    
    def _get_target_resolution(self, resolution, video_info):
        """Convert resolution setting to FFmpeg scale parameter"""
        if resolution == "original":
            return "original"
        
        # Find video stream
        video_stream = None
        for stream in video_info.get("streams", []):
            if stream.get("codec_type") == "video":
                video_stream = stream
                break
                
        if not video_stream:
            return "original"
        
        width = int(video_stream.get("width", 1280))
        height = int(video_stream.get("height", 720))
        
        # Handle different target resolutions
        if resolution == "1080p":
            if height > width:  # Portrait video
                return "1080:1920" 
            return "1920:1080"
        elif resolution == "720p":
            if height > width:  # Portrait video
                return "720:1280"
            return "1280:720"
        elif resolution == "480p":
            if height > width:  # Portrait video
                return "480:854"
            return "854:480"
        
        return "original"