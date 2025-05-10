import os
import logging
import numpy as np
import librosa
import subprocess
import tempfile
from scipy.signal import medfilt

logger = logging.getLogger(__name__)

class AudioAnalyzer:
    """Analyzes audio to find exciting moments based on volume and other characteristics."""
    
    def __init__(self, sensitivity=0.5, min_segment_duration=1.0, max_segment_duration=8.0):
        self.sensitivity = sensitivity  # Higher values = more segments detected
        self.min_segment_duration = min_segment_duration
        self.max_segment_duration = max_segment_duration
        logger.info(f"AudioAnalyzer initialized with sensitivity {sensitivity}")
    
    def extract_audio(self, video_path):
        """Extract audio from video file to a temporary WAV file"""
        try:
            temp_audio = tempfile.NamedTemporaryFile(suffix='.wav', delete=False).name
            cmd = [
                "ffmpeg", "-y", "-i", video_path, 
                "-vn", "-acodec", "pcm_s16le", 
                "-ar", "44100", "-ac", "2", 
                temp_audio
            ]
            subprocess.run(cmd, capture_output=True, check=True)
            return temp_audio
        except Exception as e:
            logger.error(f"Audio extraction failed: {str(e)}")
            return None
    
    def find_exciting_moments(self, video_path):
        """Find exciting audio segments based on volume changes and energy"""
        try:
            logger.info(f"Analyzing audio in {video_path}")
            
            # Extract audio to temporary file
            temp_audio = self.extract_audio(video_path)
            if not temp_audio or not os.path.exists(temp_audio):
                logger.error("Audio extraction failed")
                return []
            
            # Load audio using librosa
            try:
                y, sr = librosa.load(temp_audio, sr=None, mono=True)
            except Exception as e:
                logger.error(f"Librosa failed to load audio: {str(e)}")
                return []
                
            # Clean up temp file
            try:
                os.unlink(temp_audio)
            except:
                pass
            
            # Calculate features
            # 1. RMS energy (volume)
            frame_length = int(sr * 0.05)  # 50ms windows
            hop_length = int(sr * 0.025)   # 25ms hop
            
            # Calculate RMS energy (volume)
            rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
            
            # Calculate spectral contrast (for excitement detection)
            contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=hop_length), axis=0)
            
            # Calculate spectral flux (for sudden changes)
            mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, hop_length=hop_length, n_mels=128)
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            flux = np.zeros_like(mel_spec_db[0, :-1])
            for i in range(len(mel_spec_db[0]) - 1):
                flux[i] = np.sum((mel_spec_db[:, i+1] - mel_spec_db[:, i]) > 0)
            
            # Normalize features
            rms_norm = (rms - np.min(rms)) / (np.max(rms) - np.min(rms) + 1e-6)
            contrast_norm = (contrast - np.min(contrast)) / (np.max(contrast) - np.min(contrast) + 1e-6)
            flux_norm = np.pad(flux, (0, 1)) / np.max(flux)
            flux_norm = (flux_norm - np.min(flux_norm)) / (np.max(flux_norm) - np.min(flux_norm) + 1e-6)
            
            # Combine features with weights (make volume have the biggest impact)
            excitement_score = (0.6 * rms_norm) + (0.2 * contrast_norm) + (0.2 * flux_norm)
            
            # Apply median filter to smooth the scores
            excitement_score = medfilt(excitement_score, kernel_size=5)
            
            # Scale by sensitivity parameter (lower threshold = more segments)
            # The inverse relationship: higher sensitivity = lower threshold
            threshold = 0.75 - (self.sensitivity * 0.5)
            logger.info(f"Using excitement threshold: {threshold}")
            
            # Find segments above threshold
            is_exciting = excitement_score > threshold
            
            # Convert to time segments
            segment_time = hop_length / sr
            exciting_segments = []
            
            in_segment = False
            start_idx = 0
            
            for i, is_exc in enumerate(is_exciting):
                if is_exc and not in_segment:
                    # Start of new exciting segment
                    in_segment = True
                    start_idx = i
                elif not is_exc and in_segment:
                    # End of exciting segment
                    in_segment = False
                    duration = (i - start_idx) * segment_time
                    
                    # Only keep segments of reasonable length
                    if self.min_segment_duration <= duration <= self.max_segment_duration:
                        start_time = start_idx * segment_time
                        end_time = i * segment_time
                        exciting_segments.append((start_time, end_time))
            
            # Handle case where video ends during an exciting segment
            if in_segment:
                duration = (len(is_exciting) - start_idx) * segment_time
                if self.min_segment_duration <= duration <= self.max_segment_duration:
                    start_time = start_idx * segment_time
                    end_time = len(is_exciting) * segment_time
                    exciting_segments.append((start_time, end_time))
            
            # FORCE DETECTION: If no segments were found, just pick the loudest parts
            if not exciting_segments:
                logger.info("No exciting segments found naturally, forcing detection of top segments")
                
                # Find top 2-4 peaks in the RMS (volume)
                num_peaks = max(2, int(len(y) / sr / 15))  # 1 peak per ~15 seconds, at least 2
                num_peaks = min(num_peaks, 4)  # But no more than 4
                
                # Get indices of top N highest RMS values
                if len(rms_norm) > 0:
                    # Find peak indices with minimum distance to avoid overlapping segments
                    peak_indices = []
                    sorted_indices = np.argsort(rms_norm)[::-1]  # Sort in descending order
                    
                    min_distance = int(3 / segment_time)  # Minimum 3 seconds between peaks
                    for idx in sorted_indices:
                        # Check if this peak is far enough from all selected peaks
                        if not any(abs(idx - peak_idx) < min_distance for peak_idx in peak_indices):
                            peak_indices.append(idx)
                            if len(peak_indices) >= num_peaks:
                                break
                    
                    # Create a segment around each peak
                    for peak_idx in peak_indices:
                        segment_duration = 3.0  # Fixed 3-second segments
                        half_duration = segment_duration / 2
                        
                        # Calculate start and end times
                        start_time = max(0, (peak_idx * segment_time) - half_duration)
                        end_time = min((len(rms_norm) * segment_time), (peak_idx * segment_time) + half_duration)
                        
                        # Add the forced segment
                        exciting_segments.append((start_time, end_time))
            
            # Sort segments by start time
            exciting_segments.sort(key=lambda x: x[0])
            
            logger.info(f"Found {len(exciting_segments)} exciting audio segments")
            return exciting_segments
            
        except Exception as e:
            logger.error(f"Error in audio analysis: {str(e)}")
            return []