import subprocess
import os

def segment_video(video_path, output_dir, segment_duration=5, quality_suffix="1080p-8000k"):
    """
    Segments a video into TS files using FFmpeg.
    Example quality_suffix: "1080p-8000k", "720p-4000k", "480p-1500k"
    """
    if not os.path.exists(video_path):
        print(f"Error: Video file {video_path} not found.")
        return

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    # Output directory for this specific video and quality
    specific_output_dir = os.path.join(output_dir, video_name, quality_suffix)
    os.makedirs(specific_output_dir, exist_ok=True)

    output_pattern = os.path.join(specific_output_dir, f"{video_name}-{quality_suffix}-%03d.ts")
    
    # Example FFmpeg command (adjust parameters as needed)
    # -c:v libx264 -b:v 8000k (example for 8Mbps H.264, if re-encoding)
    # If your source is already in the desired codec and bitrate, you might use -c copy
    # For simplicity, let's assume we are re-encoding or it's a simple copy if source is appropriate.
    # Here, we'll use -c copy assuming the source video is already suitable for segmentation.
    # You might need to transcode to different bitrates for ABR, this example focuses on one quality.
    cmd = [
        'ffmpeg',
        '-i', video_path,
        '-c', 'copy',             # Copy codecs if possible, faster. Or specify e.g. -c:v libx264 -b:v 8000k
        '-map', '0',              # Map all streams
        '-f', 'segment',
        '-segment_time', str(segment_duration),
        '-segment_format', 'mpegts',
        '-reset_timestamps', '1', # Reset timestamps for each segment
        output_pattern
    ]

    print(f"Executing FFmpeg command: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
        print(f"Video segmented successfully into {specific_output_dir}")
    except subprocess.CalledProcessError as e:
        print(f"Error during FFmpeg execution: {e}")
    except FileNotFoundError:
        print("Error: FFmpeg command not found. Make sure FFmpeg is installed and in your PATH.")

if __name__ == '__main__':
    # --- Configuration for Segmentation ---
    SOURCE_VIDEO_FILE = "your_long_video.mp4"  # Replace with your video file
    BASE_SEGMENTS_DIR = "video_segments"      # Segments will be stored here
    
    # Example: Create segments for one quality
    # For ABR, you would run this function multiple times for the same source video
    # but with different FFmpeg parameters to generate different bitrate/resolution versions.
    # e.g., segment_video(SOURCE_VIDEO_FILE, BASE_SEGMENTS_DIR, quality_suffix="1080p-8000k", ffmpeg_encoding_options_for_1080p)
    # e.g., segment_video(SOURCE_VIDEO_FILE, BASE_SEGMENTS_DIR, quality_suffix="480p-1000k", ffmpeg_encoding_options_for_480p)

    if not os.path.exists(SOURCE_VIDEO_FILE):
        print(f"Please create a dummy video file named '{SOURCE_VIDEO_FILE}' or update the path.")
    else:
        segment_video(SOURCE_VIDEO_FILE, BASE_SEGMENTS_DIR, quality_suffix="1080p-8000k")
        # To add another quality for ABR later:
        # segment_video(SOURCE_VIDEO_FILE, BASE_SEGMENTS_DIR, quality_suffix="480p-1500k")