import subprocess
import os
import logging

# 日志设置 (保持不变)
logger = logging.getLogger('SegmentVideo')
if not logger.handlers:
    logger.setLevel(logging.INFO)
    ch_seg = logging.StreamHandler()
    formatter_seg = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch_seg.setFormatter(formatter_seg)
    logger.addHandler(ch_seg)

def segment_video(video_path, output_dir, segment_duration=5, 
                  quality_suffix="1080p-8000k", 
                  ffmpeg_encoder_options=None):
    """
    将视频分片为TS文件，并为每个质量等级生成一个M3U8播放列表，使用HLS muxer。
    """
    if not os.path.exists(video_path):
        logger.error(f"错误: 视频文件 {video_path} 未找到。")
        return

    video_name_no_ext = os.path.splitext(os.path.basename(video_path))[0]
    
    specific_quality_dir = os.path.join(output_dir, video_name_no_ext, quality_suffix)
    os.makedirs(specific_quality_dir, exist_ok=True)

    # 对于 -f hls, M3U8 文件名是命令的最后一个参数
    m3u8_filename = f"{video_name_no_ext}-{quality_suffix}.m3u8"
    m3u8_output_path = os.path.join(specific_quality_dir, m3u8_filename)

    # TS分片的文件名模式，相对于M3U8文件的位置
    # FFmpeg的 -hls_segment_filename 需要一个相对于输出M3U8目录的路径，或者一个包含完整路径的模式
    # 如果M3U8和TS在同一目录，可以直接用文件名模式
    ts_segment_filename_pattern = f"{video_name_no_ext}-{quality_suffix}-%05d.ts" # 使用 %05d 以支持更多分片
    # 注意：如果 specific_quality_dir 已经包含在 ts_output_pattern 中，那么 hls_segment_filename 也需要
    # 确保路径正确。通常，hls_segment_filename 是相对于 m3u8_output_path 所在目录的。
    # 这里我们让TS和M3U8在同一个 specific_quality_dir 目录下。
    
    cmd = ['ffmpeg', '-y', '-i', video_path]

    has_map_in_options = False
    if ffmpeg_encoder_options:
        cmd.extend(ffmpeg_encoder_options)
        if any(opt == '-map' for opt in ffmpeg_encoder_options):
            has_map_in_options = True
        if not has_map_in_options:
            logger.warning(f"警告: 为 {quality_suffix} 提供的 ffmpeg_encoder_options 中缺少 '-map' 指令。")
            # 对于HLS，强烈建议明确映射，这里不再添加默认map，促使用户配置
    else: # 默认 codec copy (如果决定支持的话)
        logger.info(f"未提供 {quality_suffix} 的转码选项。将使用 '-codec copy'。")
        cmd.extend(['-codec', 'copy'])
        if not has_map_in_options: # 如果是codec copy且没有map，则提示用户
            logger.warning("Codec copy模式下未提供-map。FFmpeg将使用默认流选择。强烈建议提供map参数。")
            cmd.extend(['-map', '0:v:0?', '-map', '0:a:0?'])

    # 使用HLS muxer的选项
    cmd.extend([
        '-f', 'hls',
        '-hls_time', str(segment_duration),       # 每个分片的时长 (秒)
        '-hls_playlist_type', 'vod',             # 'vod' (Video On Demand) 会在列表末尾添加 #EXT-X-ENDLIST
                                                 # 如果是直播，可以是 'event' 或省略 (默认为live，不加ENDLIST)
        '-hls_segment_filename', os.path.join(specific_quality_dir, ts_segment_filename_pattern), # TS分片的输出模式和路径
        '-hls_flags', 'independent_segments',    # 关键！确保分片可以独立解码
        # '-hls_flags', 'independent_segments+split_by_time', # split_by_time 是默认行为，可以不加
        # '-hls_list_size', '0',                  # 对于 VOD，保留所有分片在列表中 (通常 vod 类型已隐含此行为)
        m3u8_output_path                         # M3U8播放列表的输出路径
    ])

    logger.info(f"执行 FFmpeg ({quality_suffix}): {' '.join(cmd)}")
    try:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate(timeout=3600) 
        if process.returncode == 0:
            logger.info(f"视频 {quality_suffix} 成功分片 (HLS muxer) 并生成M3U8到 {specific_quality_dir}")
        else:
            logger.error(f"FFmpeg HLS 执行错误 ({quality_suffix}): Return code: {process.returncode}")
            logger.error(f"FFmpeg STDOUT: {stdout.decode(errors='ignore')}")
            logger.error(f"FFmpeg STDERR: {stderr.decode(errors='ignore')}")
    except FileNotFoundError:
        logger.error("错误: FFmpeg 命令未找到。请确保已安装 FFmpeg 并将其添加到系统 PATH。")
    except subprocess.TimeoutExpired:
        logger.error(f"FFmpeg 执行超时 ({quality_suffix})。进程将被终止。")
        if process.poll() is None: process.kill() #确保进程被杀死
        stdout, stderr = process.communicate()
        logger.error(f"FFmpeg STDOUT (on timeout): {stdout.decode(errors='ignore')}")
        logger.error(f"FFmpeg STDERR (on timeout): {stderr.decode(errors='ignore')}")
    except Exception as e:
        logger.error(f"执行 FFmpeg 时发生未知错误 ({quality_suffix}): {e}")

if __name__ == '__main__':
    SOURCE_VIDEO_FILE = "bbb_sunflower.mp4"  # Use your actual downloaded file
    BASE_SEGMENTS_DIR = "video_segments"

    if not os.path.exists(SOURCE_VIDEO_FILE):
        print(f"Please download '{SOURCE_VIDEO_FILE}' or update the path.")
    else:
        # Example 1: Create 1080p @ 8Mbps segments (re-encoding)
        options_1080p_8000k = [
            '-map', '0:v:0',  # 选择源文件的第一个视频流
            '-map', '0:1',    # 选择源文件的第二个流（假设这是你想要的MP3音轨）
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
        # segment_video(SOURCE_VIDEO_FILE, BASE_SEGMENTS_DIR, quality_suffix="1080p-8000k", ffmpeg_encoder_options=options_1080p_8000k)

        # Example 2: Create 720p @ 4Mbps segments (re-encoding and downscaling)
        options_720p_4000k = [
            '-map', '0:v:0',  # 选择源文件的第一个视频流
            '-map', '0:1',    # 选择源文件的第二个流（假设这是你想要的MP3音轨）
            '-c:v', 'libx264',
            '-b:v', '4000k',
            '-s', '1280x720',           # Downscale to 720p
            '-preset', 'medium',
            '-c:a', 'aac',
            '-b:a', '128k',
        ]
        # segment_video(SOURCE_VIDEO_FILE, BASE_SEGMENTS_DIR, quality_suffix="720p-4000k", ffmpeg_encoder_options=options_720p_4000k)

        # Example 3: Create 480p @ 1.5Mbps segments
        options_480p_1500k = [
            '-map', '0:v:0',  # 选择源文件的第一个视频流
            '-map', '0:1',    # 选择源文件的第二个流（假设这是你想要的MP3音轨）
            '-c:v', 'libx264',
            '-b:v', '1500k',
            '-s', '854x480',            # Example 480p resolution (16:9)
            '-preset', 'medium',
            '-c:a', 'aac',
            '-b:a', '96k',
        ]
        segment_video(SOURCE_VIDEO_FILE, BASE_SEGMENTS_DIR, quality_suffix="480p-1500k", ffmpeg_encoder_options=options_480p_1500k)