import subprocess
import os
import logging

# Logger setup
# Use the module's own name for the logger, which is a common Python practice.
# This helps in identifying the origin of log messages in larger projects.
logger = logging.getLogger(__name__)
if not logger.handlers: # Ensure handler is not added multiple times if module is reloaded
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    # Consistent log format, including timestamp, logger name, level, and message.
    log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(log_formatter)
    logger.addHandler(console_handler)

def segment_video(video_path, output_dir, segment_duration=5,
                  quality_suffix="1080p-8000k",
                  ffmpeg_encoder_options=None):
    """
    Segments a video into transport stream (TS) files and generates a corresponding
    media M3U8 playlist for a specific quality level using FFmpeg's HLS muxer.

    This function creates a dedicated directory for each quality level to store
    its TS segments and media M3U8 playlist.

    Args:
        video_path (str): The absolute or relative path to the source video file.
        output_dir (str): The base directory where all video segment processing outputs
                          will be stored (e.g., "video_segments_output"). Subdirectories
                          like "output_dir/video_name/quality_suffix/" will be created.
        segment_duration (int, optional): The target duration of each TS segment
                                          in seconds. Defaults to 5.
        quality_suffix (str, optional): A descriptive suffix for this quality level,
                                        used in directory and file naming conventions
                                        (e.g., "1080p-8000k"). Defaults to "1080p-8000k".
        ffmpeg_encoder_options (list, optional): A list of FFmpeg command-line options
                                                 specifically for encoding this quality
                                                 level. If None, FFmpeg's '-codec copy'
                                                 will be used, meaning no re-encoding.
                                                 Defaults to None.

    Returns:
        str or None: The full path to the generated media M3U8 file if segmentation
                     is successful; otherwise, None.
    """
    if not os.path.exists(video_path):
        logger.error(f"Source video file not found: {video_path}")
        return None

    video_name_no_ext = os.path.splitext(os.path.basename(video_path))[0]

    # Construct the output directory path for this specific quality's segments and playlist.
    # Example: video_segments_output/bbb_sunflower/1080p-8000k
    specific_quality_dir = os.path.join(output_dir, video_name_no_ext, quality_suffix)
    try:
        os.makedirs(specific_quality_dir, exist_ok=True)
    except OSError as e:
        logger.error(f"Failed to create directory {specific_quality_dir}: {e}")
        return None

    # Define media M3U8 filename and its full output path.
    media_m3u8_filename = f"{video_name_no_ext}-{quality_suffix}.m3u8"
    media_m3u8_output_path = os.path.join(specific_quality_dir, media_m3u8_filename)

    # Define the filename pattern for TS segments.
    # FFmpeg's -hls_segment_filename option requires a pattern that can include
    # formatting directives like %05d for sequence numbers.
    ts_segment_filename_template = f"{video_name_no_ext}-{quality_suffix}-%05d.ts"
    # It's generally more robust to provide FFmpeg with a full path pattern for segments,
    # ensuring they are created in the intended 'specific_quality_dir'.
    ts_segment_full_path_pattern = os.path.join(specific_quality_dir, ts_segment_filename_template)

    # Base FFmpeg command. '-y' overwrites output files without asking.
    cmd = ['ffmpeg', '-y', '-i', video_path]

    # Handle FFmpeg encoder options and stream mapping.
    has_map_in_options = False
    if ffmpeg_encoder_options:
        cmd.extend(ffmpeg_encoder_options)
        # Check if user-provided options include a '-map' directive.
        if any(opt == '-map' for opt in ffmpeg_encoder_options):
            has_map_in_options = True
        if not has_map_in_options:
            logger.warning(
                f"ffmpeg_encoder_options for '{quality_suffix}' were provided without a '-map' directive. "
                f"FFmpeg will use its default stream selection behavior, which might not be intended."
            )
    else:
        # If no specific encoding options, use 'codec copy' to avoid re-encoding.
        logger.info(f"No transcoding options provided for '{quality_suffix}'. Using '-codec copy'.")
        cmd.extend(['-codec', 'copy'])
        # For 'codec copy', if no '-map' is given, FFmpeg might not select any streams or might
        # select only video by default. We add a default map to try selecting the first video
        # and first audio stream if they exist. '0:v:0?' and '0:a:0?' are speculative maps;
        # '?' makes them optional, so FFmpeg won't error if a stream type isn't present.
        if not has_map_in_options: # This condition applies if ffmpeg_encoder_options was None
            logger.info(
                "In '-codec copy' mode and no '-map' provided by user. Defaulting to map "
                "first video stream (0:v:0?) and first audio stream (0:a:0?), if available."
            )
            cmd.extend(['-map', '0:v:0?', '-map', '0:a:0?'])

    # Add HLS-specific FFmpeg options.
    cmd.extend([
        '-f', 'hls',                             # Output format is HLS.
        '-hls_time', str(segment_duration),       # Target segment duration.
        '-hls_playlist_type', 'vod',              # Creates a VOD (Video On Demand) type playlist.
                                                  # Use 'event' for live streams.
        '-hls_segment_filename', ts_segment_full_path_pattern, # Full path pattern for TS files.
        '-hls_flags', 'independent_segments',     # Crucial for ABR: ensures each segment can be
                                                  # decoded independently without relying on prior segments
                                                  # beyond the HLS spec for keyframes.
        media_m3u8_output_path                    # Path for the generated media M3U8 file.
    ])

    logger.info(f"Executing FFmpeg for '{quality_suffix}': {' '.join(cmd)}")
    try:
        # Start the FFmpeg process. stdout and stderr are piped to capture output.
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # Wait for the process to complete and get its output.
        # A timeout is set to prevent indefinite blocking (e.g., 1 hour).
        stdout, stderr = process.communicate(timeout=3600)

        # Decode and log FFmpeg's stderr (often contains progress and errors).
        stderr_decoded = stderr.decode(errors='ignore')
        if stderr_decoded.strip(): # Log if not empty
            logger.info(f"FFmpeg STDERR for '{quality_suffix}':\n{stderr_decoded}")
        else:
            logger.info(f"FFmpeg STDERR for '{quality_suffix}': No output or only whitespace.")

        # FFmpeg stdout is usually empty for HLS muxing unless specific options are used.
        stdout_decoded = stdout.decode(errors='ignore')
        if stdout_decoded.strip(): # Log if not empty
            logger.info(f"FFmpeg STDOUT for '{quality_suffix}':\n{stdout_decoded}")

        if process.returncode == 0:
            logger.info(
                f"Video successfully segmented for '{quality_suffix}'. "
                f"Media M3U8 generated: {media_m3u8_output_path}"
            )
            return media_m3u8_output_path
        else:
            logger.error(
                f"FFmpeg HLS execution failed for '{quality_suffix}'. "
                f"Return code: {process.returncode}. Check STDERR above for details."
            )
            return None

    except FileNotFoundError:
        logger.error(
            "FFmpeg command not found. Please ensure FFmpeg is installed "
            "and its executable is in your system's PATH environment variable."
        )
    except subprocess.TimeoutExpired:
        logger.error(
            f"FFmpeg execution timed out (limit: 3600s) for '{quality_suffix}'. "
            "The process will be terminated."
        )
        if process.poll() is None:  # Check if process is still running
            process.kill()
            logger.info(f"FFmpeg process for '{quality_suffix}' killed due to timeout.")
        # Attempt to get any output after killing (may be partial or empty)
        stdout, stderr = process.communicate()
        logger.error(f"FFmpeg STDOUT (on timeout, after kill) for '{quality_suffix}':\n{stdout.decode(errors='ignore')}")
        logger.error(f"FFmpeg STDERR (on timeout, after kill) for '{quality_suffix}':\n{stderr.decode(errors='ignore')}")
    except Exception as e:
        # Catch any other unexpected exceptions during FFmpeg execution.
        logger.error(f"An unexpected error occurred while executing FFmpeg for '{quality_suffix}': {e}", exc_info=True)
    return None


def create_master_playlist(output_video_base_dir, qualities_details_list,
                           master_m3u8_filename="master.m3u8"):
    """
    Creates a master M3U8 playlist file that points to multiple media M3U8 playlists
    for different quality levels (Adaptive Bitrate Streaming).

    Args:
        output_video_base_dir (str): The root output directory for the processed video,
                                     where the master M3U8 file will be saved.
                                     Example: "video_segments_output/video_name".
        qualities_details_list (list): A list of dictionaries. Each dictionary must
                                       contain details for one quality stream/variant,
                                       including 'suffix', 'bandwidth',
                                       'media_m3u8_filename', and optionally
                                       'resolution' and 'codecs'.
                                       Example:
                                       [
                                           {'suffix': '480p-1500k', 'bandwidth': 1596000,
                                            'resolution': '854x480', 'codecs': 'avc1.64001e,mp4a.40.2',
                                            'media_m3u8_filename': 'video_name-480p-1500k.m3u8'},
                                           # ... other quality stream dictionaries ...
                                       ]
        master_m3u8_filename (str, optional): The desired filename for the master M3U8
                                             playlist. Defaults to "master.m3u8".
    """
    master_m3u8_path = os.path.join(output_video_base_dir, master_m3u8_filename)
    try:
        os.makedirs(output_video_base_dir, exist_ok=True)
    except OSError as e:
        logger.error(f"Failed to create directory for master playlist {master_m3u8_path}: {e}")
        return

    try:
        with open(master_m3u8_path, 'w', encoding='utf-8') as f:
            f.write("#EXTM3U\n")            # Standard M3U8 header
            f.write("#EXT-X-VERSION:3\n")   # Specifies HLS protocol version (3 is widely compatible)

            for quality_info in qualities_details_list:
                # Extract details for each quality stream.
                quality_suffix = quality_info['suffix']
                bandwidth = quality_info['bandwidth']
                resolution = quality_info.get('resolution') # Resolution is optional
                # Provide a common default for codecs if not specified for a variant.
                codecs = quality_info.get('codecs', "avc1.42001E,mp4a.40.2") # Generic H.264 + AAC
                media_m3u8_filename = quality_info['media_m3u8_filename']

                # Construct the relative path to the media playlist from the master playlist's location.
                # Example: "480p-1500k/video_name-480p-1500k.m3u8"
                # M3U8 paths should use forward slashes, regardless of the operating system.
                relative_media_m3u8_path = os.path.join(quality_suffix, media_m3u8_filename).replace('\\', '/')

                # Build the #EXT-X-STREAM-INF tag line.
                stream_inf_line = f"#EXT-X-STREAM-INF:BANDWIDTH={bandwidth}"
                if resolution:
                    stream_inf_line += f",RESOLUTION={resolution}"
                if codecs: # The CODECS attribute is important for player compatibility.
                    stream_inf_line += f",CODECS=\"{codecs}\""

                f.write(stream_inf_line + "\n")
                f.write(relative_media_m3u8_path + "\n")

        logger.info(f"Master M3U8 playlist successfully created at: {master_m3u8_path}")
    except IOError as e:
        logger.error(f"Failed to write master M3U8 playlist to {master_m3u8_path}: {e}")
    except KeyError as e:
        logger.error(f"Missing expected key in qualities_details_list for master playlist generation: {e}. "
                     "Each item requires 'suffix', 'bandwidth', and 'media_m3u8_filename'.")


if __name__ == '__main__':
    # --- Configuration for script execution ---
    SOURCE_VIDEO_FILE = "bbb_sunflower.mp4"     # Path to the input video file.
    BASE_OUTPUT_DIR = "video_segments"          # Top-level directory for all outputs.
    SEGMENT_DURATION = 5                        # Target duration for each HLS segment in seconds.
    VIDEO_FRAMERATE = 30 # Assumed framerate for GOP calculation. Adjust if your source differs.
                         # Use 'ffprobe -v error -select_streams v:0 -show_entries stream=r_frame_rate -of default=noprint_wrappers=1:nokey=1 your_video.mp4'
                         # to get the actual framerate (e.g., 30000/1001 for 29.97fps).

    # --- Select an ABR ladder configuration ---
    # Set to True for a 5-level ladder (includes 2160p, 360p).
    # Set to False for a 3-level ladder (1080p, 720p, 480p as previously defined).
    USE_FIVE_LEVELS_LADDER = True # Modify this to switch configurations.

    # --- Basic check for source video ---
    if not os.path.exists(SOURCE_VIDEO_FILE):
        logger.error(f"Source video file '{SOURCE_VIDEO_FILE}' not found. Please check the path. Exiting.")
    else:
        video_name_no_ext = os.path.splitext(os.path.basename(SOURCE_VIDEO_FILE))[0]
        # Define the root output directory for this specific video's processed files.
        # The master M3U8 playlist will be placed here.
        # Example: "video_segments_output/bbb_sunflower/"
        output_dir_for_this_video = os.path.join(BASE_OUTPUT_DIR, video_name_no_ext)

        # --- Comprehensive ABR Ladder Configuration ---
        # This dictionary defines all available quality profiles.
        # The script will select a subset of these based on USE_FIVE_LEVELS_LADDER.
        #
        # IMPORTANT NOTES ON FFmpeg OPTIONS:
        # - '-map 0:v:0 -map 0:a:0': Assumes the primary video and audio are the first of their type.
        #   ALWAYS verify with 'ffprobe' for your specific source video.
        # - '-g <GOP_size>': Group of Pictures size. Typically 2x framerate for HLS.
        # - '-keyint_min <min_keyframe_interval>': Minimum keyframe interval. Usually same as framerate.
        # - '-preset': Controls encoding speed vs. compression efficiency. 'fast' is a common balance.
        #   Other options: 'ultrafast', 'superfast', 'veryfast', 'faster', 'medium', 'slow', 'slower', 'veryslow'.
        # - Codecs string: 'avc1...' for H.264, 'mp4a.40.2' for AAC-LC. These should match your encoding.
        #   The hex part in 'avc1...' (e.g., 64001e) indicates profile and level.
        #   Consult H.264 documentation or use tools to determine correct values if changing profiles/levels.

        all_quality_profiles = {
            "360p-800k": {
                'ffmpeg_opts': [
                    '-map', '0:v:0', '-map', '0:a:0',
                    '-c:v', 'libx264', '-b:v', '800k', '-maxrate', '856k', '-bufsize', '1200k', '-s', '640x360',
                    '-preset', 'fast', '-g', str(VIDEO_FRAMERATE * 2),
                    '-keyint_min', str(VIDEO_FRAMERATE), '-sc_threshold', '0',
                    '-c:a', 'aac', '-b:a', '64k', '-ar', '44100', '-ac', '2'
                ],
                'bandwidth': 800000 + 64000, # Total estimated bandwidth (video + audio)
                'resolution': '640x360',
                'codecs': "avc1.42c01e,mp4a.40.2" # H.264 Baseline@L3.0, AAC-LC
            },
            "480p-1500k": {
                'ffmpeg_opts': [
                    '-map', '0:v:0', '-map', '0:a:0',
                    '-c:v', 'libx264', '-b:v', '1500k', '-maxrate', '1605k', '-bufsize', '2250k', '-s', '854x480',
                    '-preset', 'fast', '-g', str(VIDEO_FRAMERATE * 2),
                    '-keyint_min', str(VIDEO_FRAMERATE), '-sc_threshold', '0',
                    '-c:a', 'aac', '-b:a', '96k', '-ar', '44100', '-ac', '2'
                ],
                'bandwidth': 1500000 + 96000,
                'resolution': '854x480',
                'codecs': "avc1.4d001e,mp4a.40.2" # H.264 Main@L3.0, AAC-LC (example, adjust if needed)
            },
            "720p-4000k": {
                'ffmpeg_opts': [
                    '-map', '0:v:0', '-map', '0:a:0',
                    '-c:v', 'libx264', '-b:v', '4000k', '-maxrate', '4280k', '-bufsize', '6000k', '-s', '1280x720',
                    '-preset', 'fast', '-g', str(VIDEO_FRAMERATE * 2),
                    '-keyint_min', str(VIDEO_FRAMERATE), '-sc_threshold', '0',
                    '-c:a', 'aac', '-b:a', '128k', '-ar', '44100', '-ac', '2'
                ],
                'bandwidth': 4000000 + 128000,
                'resolution': '1280x720',
                'codecs': "avc1.4d001f,mp4a.40.2" # H.264 Main@L3.1, AAC-LC
            },
            "1080p-8000k": {
                'ffmpeg_opts': [
                    '-map', '0:v:0', '-map', '0:a:0',
                    '-c:v', 'libx264', '-b:v', '8000k', '-maxrate', '8560k', '-bufsize', '12000k', '-s', '1920x1080',
                    '-preset', 'fast', '-g', str(VIDEO_FRAMERATE * 2),
                    '-keyint_min', str(VIDEO_FRAMERATE), '-sc_threshold', '0',
                    '-c:a', 'aac', '-b:a', '192k', '-ar', '44100', '-ac', '2'
                ],
                'bandwidth': 8000000 + 192000,
                'resolution': '1920x1080',
                'codecs': "avc1.640028,mp4a.40.2" # H.264 High@L4.0, AAC-LC
            },
            "2160p-16000k": { # 4K UHD
                'ffmpeg_opts': [
                    '-map', '0:v:0', '-map', '0:a:0',
                    '-c:v', 'libx264', '-b:v', '16000k', '-maxrate', '17120k', '-bufsize', '24000k', '-s', '3840x2160',
                    '-preset', 'fast', # Consider 'medium' or 'slow' for 4K if quality is paramount and time allows
                    '-g', str(VIDEO_FRAMERATE * 2),
                    '-keyint_min', str(VIDEO_FRAMERATE), '-sc_threshold', '0',
                    '-c:a', 'aac', '-b:a', '256k', '-ar', '44100', '-ac', '2'
                ],
                'bandwidth': 16000000 + 256000,
                'resolution': '3840x2160',
                'codecs': "avc1.640033,mp4a.40.2" # Example: H.264 High@L5.1 or L5.2. Check FFmpeg output for actual profile/level.
            }
        }

        # Select which quality profiles to process based on the flag.
        if USE_FIVE_LEVELS_LADDER:
            # Define the order of qualities for the 5-level ladder.
            # This order also influences the order in the master M3U8 if not sorted by bandwidth later.
            active_quality_keys = ["360p-800k", "480p-1500k", "720p-4000k", "1080p-8000k", "2160p-16000k"]
            logger.info("Configuration selected: 5 quality levels for segmentation.")
        else:
            active_quality_keys = ["480p-1500k", "720p-4000k", "1080p-8000k"]
            logger.info("Configuration selected: 3 quality levels for segmentation.")

        # Prepare the list of quality configurations to actually process.
        qualities_to_process_list = []
        for key in active_quality_keys:
            if key in all_quality_profiles:
                # Create a copy and add the 'suffix' key for convenience.
                profile_config = all_quality_profiles[key].copy()
                profile_config['suffix'] = key # The suffix is the key itself.
                qualities_to_process_list.append(profile_config)
            else:
                logger.warning(f"Configuration for quality profile key '{key}' not found in 'all_quality_profiles'. Skipping.")

        # This list will store details needed for generating the master M3U8 playlist.
        master_playlist_variant_streams = []

        # Process each selected quality profile.
        for profile_config_item in qualities_to_process_list:
            current_quality_suffix = profile_config_item['suffix']
            logger.info(f"--- Starting processing for quality profile: {current_quality_suffix} ---")

            # Call the segmentation function for the current profile.
            media_m3u8_path_generated = segment_video(
                video_path=SOURCE_VIDEO_FILE,
                output_dir=BASE_OUTPUT_DIR, # segment_video will create subdirs based on video_name and suffix
                segment_duration=SEGMENT_DURATION,
                quality_suffix=current_quality_suffix,
                ffmpeg_encoder_options=profile_config_item['ffmpeg_opts']
            )

            if media_m3u8_path_generated: # If segmentation was successful for this profile
                master_playlist_variant_streams.append({
                    'suffix': current_quality_suffix,
                    'bandwidth': profile_config_item['bandwidth'],
                    'resolution': profile_config_item.get('resolution'),
                    'media_m3u8_filename': os.path.basename(media_m3u8_path_generated),
                    'codecs': profile_config_item.get('codecs')
                })
            else:
                logger.error(f"Segmentation failed for quality profile: {current_quality_suffix}. "
                             "This profile will be excluded from the master playlist.")
            logger.info(f"--- Finished processing for quality profile: {current_quality_suffix} ---")

        # After processing all selected quality profiles, create the master M3U8 playlist.
        if master_playlist_variant_streams:
            logger.info("--- Starting master M3U8 playlist creation ---")
            create_master_playlist(
                output_video_base_dir=output_dir_for_this_video,
                qualities_details_list=master_playlist_variant_streams
                # master_m3u8_filename defaults to "master.m3u8"
            )
        else:
            logger.warning(
                "No media streams were successfully generated after processing all selected profiles. "
                "The master M3U8 playlist will not be created."
            )
        logger.info("All HLS segmentation processing finished.")