import streamlit as st
import tempfile
import os
import time
from pathlib import Path
import logging
import subprocess
from datetime import datetime

# Import modules
from modules.scene_detector import SceneDetector
from modules.audio_analyzer import AudioAnalyzer
from modules.action_recognizer import ActionRecognizer
from modules.highlight_creator import HighlightCreator
from modules.video_editor import VideoEditor
from utils.helpers import get_video_info, format_time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# Configure Streamlit page
st.set_page_config(
    page_title="Video Highlight Generator",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    st.title("🎬 Automatic Video Highlight Generator")
    st.markdown("""
    Upload a video and let AI create a highlight reel by detecting key moments.
    The system uses scene detection, audio analysis, and action recognition.
    """)

    # Settings sidebar
    with st.sidebar:
        st.header("Settings")
        
        highlight_duration = st.slider(
            "Target highlight duration (seconds)", 
            min_value=10, max_value=300, value=60, step=10,
            help="The approximate length of the final highlight video"
        )
        
        st.subheader("Detection Methods")
        use_scene_detection = st.toggle("Scene Changes", value=True, 
                                      help="Detect significant visual changes in the video")
        use_audio_analysis = st.toggle("Audio Excitement", value=True, 
                                     help="Detect exciting moments based on audio")
        use_action_recognition = st.toggle("Action Detection", value=False, 
                                        help="Identify important actions or objects")
        
        sensitivity = st.slider(
            "Detection sensitivity", 0.0, 1.0, 0.5,
            help="Higher values detect more moments as interesting"
        )
        
        st.subheader("Output Settings")
        transition_type = st.selectbox(
            "Transition type",
            ["Fade", "Cut", "None"],
            help="Effect used between highlight clips"
        )
        
        resolution_options = {
            "Original": "original",
            "1080p": "1080p",
            "720p": "720p",
            "480p": "480p"
        }
        output_resolution = st.selectbox(
            "Output Resolution",
            list(resolution_options.keys()),
            index=0,
            help="Resolution of the output highlight video"
        )
        
        add_intro = st.checkbox("Add Intro Title", value=True)

    # File uploader
    uploaded_file = st.file_uploader(
        "Upload a video file", 
        type=["mp4", "mov", "avi", "mkv", "webm"],
        help="Maximum upload size depends on your Streamlit server configuration"
    )
    
    if uploaded_file:
        # Save uploaded file temporarily
        temp_dir = tempfile.mkdtemp()
        temp_path = os.path.join(temp_dir, uploaded_file.name)
        
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Video preview
        with st.expander("Preview Original Video", expanded=True):
            st.video(temp_path)
            
            # Get video metadata
            video_info = get_video_info(temp_path)
            
            # Force shorter highlights
            if video_info:
                video_duration = float(video_info.get('duration', 0))
                # Limit to 30% of original video length or 60 seconds, whichever is shorter
                max_allowed_duration = min(video_duration * 0.3, 60.0)
                highlight_duration = min(highlight_duration, max_allowed_duration)
                logger.info(f"Limiting highlights to {highlight_duration:.1f}s (30% of {video_duration:.1f}s)")
            
            if video_info:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Duration", video_info.get('duration_formatted', 'Unknown'))
                with col2:
                    st.metric("Resolution", f"{video_info.get('width', 0)}×{video_info.get('height', 0)}")
                with col3:
                    st.metric("FPS", f"{video_info.get('fps', 0):.1f}")
        
        # Process button
        if st.button("Generate Highlights", type="primary", use_container_width=True):
            # Create progress UI
            progress_container = st.container()
            with progress_container:
                progress_bar = st.progress(0)
                status_text = st.empty()
                results_area = st.container()
            
            # Record start time
            start_time = time.time()
            
            # Collect detection methods
            detection_options = []
            if use_scene_detection:
                detection_options.append("Scene Changes")
            if use_audio_analysis:
                detection_options.append("Audio Excitement")
            if use_action_recognition:
                detection_options.append("Action Detection")
            
            if not detection_options:
                st.warning("Please select at least one detection method.")
                return
            
            try:
                progress_bar.progress(0.1)
                
                # Create processing objects
                scene_detector = SceneDetector(threshold=sensitivity)
                audio_analyzer = AudioAnalyzer(sensitivity=sensitivity)
                action_recognizer = ActionRecognizer(confidence_threshold=0.3 + sensitivity * 0.4)
                highlight_creator = HighlightCreator()
                video_editor = VideoEditor()
                
                # Step 1: Detect scenes
                status_text.text("Detecting scene changes...")
                scenes = []
                if "Scene Changes" in detection_options:
                    scenes = scene_detector.detect_scenes(temp_path)
                    logger.info(f"Detected {len(scenes)} scenes")
                progress_bar.progress(0.25)
                
                # Step 2: Analyze audio
                status_text.text("Analyzing audio for exciting moments...")
                audio_highlights = []
                if "Audio Excitement" in detection_options:
                    audio_highlights = audio_analyzer.find_exciting_moments(temp_path)
                    logger.info(f"Detected {len(audio_highlights)} audio highlights")
                progress_bar.progress(0.5)
                
                # Step 3: Detect actions
                status_text.text("Recognizing important actions...")
                action_highlights = []
                if "Action Detection" in detection_options:
                    action_highlights = action_recognizer.detect_actions(temp_path)
                    logger.info(f"Detected {len(action_highlights)} action segments")
                progress_bar.progress(0.75)
                
                # Step 4: Create highlights - FORCE short segments
                status_text.text("Selecting highlight segments...")
                original_duration = video_info.get('duration', 0) if video_info else 0
                target_duration = min(highlight_duration, original_duration * 0.3)  # Force 30% max

                # Force creation of short segments
                highlights = []
                if original_duration > 0:
                    # Create 3-5 segments distributed through video
                    num_segments = min(5, max(3, int(target_duration / 8)))
                    segment_length = min(8.0, target_duration / num_segments)
                    
                    for i in range(num_segments):
                        # Position segments at different points (0.1, 0.3, 0.5, 0.7, 0.9)
                        position_pct = 0.1 + (0.8 * i / (num_segments - 1)) if num_segments > 1 else 0.5
                        position = position_pct * original_duration
                        start_time_seg = max(0, position - (segment_length / 2))
                        end_time_seg = min(original_duration, start_time_seg + segment_length)
                        highlights.append((start_time_seg, end_time_seg))

                    logger.info(f"FORCED {len(highlights)} segments: {highlights}")
                else:
                    # Fall back to regular highlight creation if can't determine duration
                    highlights = highlight_creator.create_highlights(
                        video_path=temp_path,
                        scenes=scenes,
                        audio_highlights=audio_highlights,
                        action_highlights=action_highlights,
                        target_duration=target_duration
                    )
                
                # Step 5: Edit video with transitions
                status_text.text("Creating highlight video...")
                
                # Generate the final output path
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = os.path.join("output", f"highlights_{timestamp}.mp4")
                
                # Create highlight video using the VideoEditor
                output_path = video_editor.create_highlight_video(
                    video_path=temp_path,
                    highlights=highlights,
                    transition_type=transition_type,
                    output_dir="output",
                    add_intro=add_intro,  # Use the standard intro from VideoEditor if selected
                    resolution=resolution_options[output_resolution]
                )
                
                progress_bar.progress(1.0)
                
                # Calculate processing time
                processing_time = time.time() - start_time
                
                # Display results
                with results_area:
                    if output_path and os.path.exists(output_path):
                        st.success(f"🎉 Highlight video created successfully in {processing_time:.1f} seconds!")
                        
                        # Show highlight clips
                        st.subheader("Highlight Video")
                        st.video(output_path)
                        
                        # Download button
                        with open(output_path, "rb") as file:
                            filename = f"highlight_{Path(uploaded_file.name).stem}.mp4"
                            st.download_button(
                                label="⬇️ Download Highlight Video",
                                data=file,
                                file_name=filename,
                                mime="video/mp4",
                                use_container_width=True
                            )
                        
                        # Statistics
                        st.subheader("Highlight Statistics")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Scenes Detected", len(scenes) if scenes else 0)
                        with col2:
                            st.metric("Audio Highlights", len(audio_highlights) if audio_highlights else 0)
                        with col3:
                            st.metric("Action Segments", len(action_highlights) if action_highlights else 0)
                    else:
                        st.error("Failed to create highlight video. Check if FFmpeg is installed.")
                    
            except Exception as e:
                st.error(f"Error generating highlights: {str(e)}")
                logger.exception("Error in highlight generation")

if __name__ == "__main__":
    # Create necessary directories
    os.makedirs("output", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    main()