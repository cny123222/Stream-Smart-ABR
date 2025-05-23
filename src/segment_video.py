import subprocess
import os

def segment_video(video_path, output_dir, segment_duration=5, quality_suffix="1080p-8000k", ffmpeg_options=None):
    """
    Segments a video into TS files using FFmpeg.
    Allows for custom ffmpeg options for transcoding.
    ffmpeg_options should be a list of strings, e.g.,
    ['-c:v', 'libx264', '-b:v', '8000k', '-s', '1920x1080', '-c:a', 'aac', '-b:a', '192k']
    """
    if not os.path.exists(video_path):
        print(f"Error: Video file {video_path} not found.")
        return

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    specific_output_dir = os.path.join(output_dir, video_name, quality_suffix)
    os.makedirs(specific_output_dir, exist_ok=True)
    output_pattern = os.path.join(specific_output_dir, f"{video_name}-{quality_suffix}-%03d.ts")

    base_cmd = [
        'ffmpeg',
        '-i', video_path
    ]

    # Add transcoding options if provided
    if ffmpeg_options:
        base_cmd.extend(ffmpeg_options)
    else:
        # Default to copy if no specific transcoding options are given
        # (This might be useful if the source is already in the target format/bitrate)
        print(f"Warning: No ffmpeg_options provided for {quality_suffix}. Defaulting to codec copy. "
              "Ensure source video matches target quality or provide transcoding options.")
        base_cmd.extend(['-c', 'copy']) # Or raise an error if options are mandatory

    # Add stream mapping - be specific if your source has multiple audio/video tracks
    # For bbb_sunflower_1080p_30fps_normal.mp4 which has 1 video (0:0) and 2 audio (0:1 MP3, 0:2 AC3)
    # Let's choose video stream and the first audio stream (MP3)
    base_cmd.extend(['-map', '0:v:0']) # Map the first video stream
    base_cmd.extend(['-map', '0:a:0']) # Map the first audio stream (assuming it's index 0, but it was 0:1 for MP3)
                                      # ffprobe showed video index 0, mp3 audio index 1. So use 0:1 for MP3
                                      # Corrected: base_cmd.extend(['-map', '0:1']) if we want the mp3 stream

    # Let's assume we want video stream 0 and audio stream 1 (MP3 from your ffprobe)
    # If ffmpeg_options already contains -map, this might conflict. Careful design needed.
    # A simpler approach might be to include mapping in ffmpeg_options itself.
    # For now, let's assume ffmpeg_options does NOT include -map.

    # Corrected mapping for BBB video (video stream at index 0, MP3 audio at index 1)
    # This should ideally be more dynamic or part of ffmpeg_options
    current_map_options = ['-map', '0:v:0', '-map', '0:a:1'] # Select video 0, audio 1 (MP3)
    
    # Ensure map options are not duplicated if already in ffmpeg_options
    has_map = any(opt == '-map' for opt in (ffmpeg_options or []))
    if not has_map:
        base_cmd.extend(current_map_options)
    elif ffmpeg_options: # if map is in options, ensure audio/video codecs are also there
        if '-c:v' not in ffmpeg_options: base_cmd.extend(['-c:v', 'copy']) # default video codec if map is set but no codec
        if '-c:a' not in ffmpeg_options: base_cmd.extend(['-c:a', 'copy']) # default audio codec

    segment_cmd_options = [
        '-f', 'segment',
        '-segment_time', str(segment_duration),
        '-segment_format', 'mpegts',
        '-reset_timestamps', '1',
        output_pattern
    ]
    
    final_cmd = base_cmd + segment_cmd_options

    print(f"Executing FFmpeg command for {quality_suffix}: {' '.join(final_cmd)}")
    try:
        subprocess.run(final_cmd, check=True)
        print(f"Video segmented successfully for {quality_suffix} into {specific_output_dir}")
    except subprocess.CalledProcessError as e:
        print(f"Error during FFmpeg execution for {quality_suffix}: {e}")
    except FileNotFoundError:
        print("Error: FFmpeg command not found. Make sure FFmpeg is installed and in your PATH.")

if __name__ == '__main__':
    SOURCE_VIDEO_FILE = "bbb_sunflower.mp4"  # Use your actual downloaded file
    BASE_SEGMENTS_DIR = "video_segments"

    if not os.path.exists(SOURCE_VIDEO_FILE):
        print(f"Please download '{SOURCE_VIDEO_FILE}' or update the path.")
    else:
        # Example 1: Create 1080p @ 8Mbps segments (re-encoding)
        options_1080p_8000k = [
            '-c:v', 'libx264',          # Video codec
            '-b:v', '8000k',            # Video bitrate
            '-s', '1920x1080',          # Resolution (optional if source is already 1080p and not changing)
            '-preset', 'medium',        # x264 preset for encoding speed/quality balance
            '-c:a', 'aac',              # Audio codec (AAC is good for streaming)
            '-b:a', '192k',             # Audio bitrate
            # Map options are handled in the function, or could be passed here if more control is needed
            # e.g. if ffmpeg_options included its own -map, the function's default map should be skipped.
            # For the BBB video with video stream 0 and MP3 audio stream 1:
            # We are using the function's default map: ['-map', '0:v:0', '-map', '0:a:1']
        ]
        segment_video(SOURCE_VIDEO_FILE, BASE_SEGMENTS_DIR, quality_suffix="1080p-8000k", ffmpeg_options=options_1080p_8000k)

        # Example 2: Create 720p @ 4Mbps segments (re-encoding and downscaling)
        options_720p_4000k = [
            '-c:v', 'libx264',
            '-b:v', '4000k',
            '-s', '1280x720',           # Downscale to 720p
            '-preset', 'medium',
            '-c:a', 'aac',
            '-b:a', '128k',
        ]
        segment_video(SOURCE_VIDEO_FILE, BASE_SEGMENTS_DIR, quality_suffix="720p-4000k", ffmpeg_options=options_720p_4000k)

        # Example 3: Create 480p @ 1.5Mbps segments
        options_480p_1500k = [
            '-c:v', 'libx264',
            '-b:v', '1500k',
            '-s', '854x480',            # Example 480p resolution (16:9)
            '-preset', 'medium',
            '-c:a', 'aac',
            '-b:a', '96k',
        ]
        segment_video(SOURCE_VIDEO_FILE, BASE_SEGMENTS_DIR, quality_suffix="480p-1500k", ffmpeg_options=options_480p_1500k)