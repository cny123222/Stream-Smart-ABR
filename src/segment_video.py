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
    将视频分片为TS文件，并为每个质量等级生成一个媒体M3U8播放列表，使用HLS muxer。
    返回生成的媒体M3U8文件的完整路径，如果成功的话。
    """
    if not os.path.exists(video_path):
        logger.error(f"错误: 视频文件 {video_path} 未找到。")
        return None

    video_name_no_ext = os.path.splitext(os.path.basename(video_path))[0]
    
    # 特定质量分片的输出目录，例如: video_segments/video_name/1080p-8000k
    specific_quality_dir = os.path.join(output_dir, video_name_no_ext, quality_suffix)
    os.makedirs(specific_quality_dir, exist_ok=True)

    # 媒体M3U8文件名和完整路径
    media_m3u8_filename = f"{video_name_no_ext}-{quality_suffix}.m3u8"
    media_m3u8_output_path = os.path.join(specific_quality_dir, media_m3u8_filename)

    # TS分片的文件名模式。-hls_segment_filename 需要一个路径模式。
    # 如果M3U8和TS在同一目录 (specific_quality_dir)，则可以直接使用文件名模式。
    # FFmpeg 会在 specific_quality_dir 中创建这些TS文件。
    ts_segment_filename_pattern = f"{video_name_no_ext}-{quality_suffix}-%05d.ts"
    ts_segment_filename_pattern = os.path.join(specific_quality_dir, ts_segment_filename_pattern)
    # -hls_segment_filename 参数的值应该是相对于M3U8输出目录的路径，或者是一个绝对路径。
    # 这里我们让TS和M3U8在同一个 specific_quality_dir 目录下，所以直接用模式名。
    # FFmpeg的CWD（当前工作目录）会影响相对路径的解析，为保险起见，给hls_segment_filename一个相对于M3U8的路径，
    # 或者让FFmpeg自己处理。如果m3u8_output_path是最终输出参数，FFmpeg通常会将ts文件放在其旁边。
    # 最简单的方式是让 -hls_segment_filename 只包含文件名模式，FFmpeg会将TS文件输出到与M3U8相同的目录。
    
    cmd = ['ffmpeg', '-y', '-i', video_path]

    has_map_in_options = False
    if ffmpeg_encoder_options:
        cmd.extend(ffmpeg_encoder_options)
        if any(opt == '-map' for opt in ffmpeg_encoder_options):
            has_map_in_options = True
        if not has_map_in_options: # 如果提供了编码选项但没有map，发出警告
            logger.warning(f"警告: 为 {quality_suffix} 提供的 ffmpeg_encoder_options 中缺少 '-map' 指令。FFmpeg将使用默认流选择。")
    else: 
        logger.info(f"未提供 {quality_suffix} 的转码选项。将使用 '-codec copy'。")
        cmd.extend(['-codec', 'copy'])
        # 对于codec copy，如果用户没有提供-map，我们添加一个默认的，尝试选择第一个视频和第一个音频（如果存在）
        # 注意：这里的 '0:a:0?' 如果源视频没有音频，不会报错。
        if not has_map_in_options: # 确保这个检查针对的是 ffmpeg_encoder_options 为 None 的情况
            logger.info("Codec copy模式下未提供-map，默认映射第一个视频流(0:v:0?)和第一个音频流(0:a:0?)")
            cmd.extend(['-map', '0:v:0?', '-map', '0:a:0?'])


    # 使用HLS muxer的选项
    cmd.extend([
        '-f', 'hls',
        '-hls_time', str(segment_duration),
        '-hls_playlist_type', 'vod',
        # '-hls_segment_filename' 的值是相对于M3U8文件所在目录的路径模式
        # 由于M3U8和TS在同一目录 (specific_quality_dir)，这里可以直接使用文件名模式
        '-hls_segment_filename', ts_segment_filename_pattern, 
        '-hls_flags', 'independent_segments',
        media_m3u8_output_path # M3U8播放列表的输出路径 (FFmpeg命令的最后一个参数)
    ])

    logger.info(f"执行 FFmpeg ({quality_suffix}): {' '.join(cmd)}")
    try:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate(timeout=3600) 
        
        stderr_decoded = stderr.decode(errors='ignore')
        if stderr_decoded:
            logger.info(f"FFmpeg STDERR ({quality_suffix}):\n{stderr_decoded}")
        else:
            logger.info(f"FFmpeg STDERR ({quality_suffix}): No output.")
        
        if stdout.decode(errors='ignore'): # stdout 通常为空，除非FFmpeg特殊配置
            logger.info(f"FFmpeg STDOUT ({quality_suffix}):\n{stdout.decode(errors='ignore')}")

        if process.returncode == 0:
            logger.info(f"视频 {quality_suffix} 成功分片 (HLS muxer) 并生成媒体M3U8到 {media_m3u8_output_path}")
            return media_m3u8_output_path
        else:
            logger.error(f"FFmpeg HLS 执行错误 ({quality_suffix}): Return code: {process.returncode}")
            return None
            
    except FileNotFoundError:
        logger.error("错误: FFmpeg 命令未找到。请确保已安装 FFmpeg 并将其添加到系统 PATH。")
    except subprocess.TimeoutExpired:
        logger.error(f"FFmpeg 执行超时 ({quality_suffix})。进程将被终止。")
        if process.poll() is None: process.kill()
        stdout, stderr = process.communicate()
        logger.error(f"FFmpeg STDOUT (on timeout):\n{stdout.decode(errors='ignore')}")
        logger.error(f"FFmpeg STDERR (on timeout):\n{stderr.decode(errors='ignore')}")
    except Exception as e:
        logger.error(f"执行 FFmpeg 时发生未知错误 ({quality_suffix}): {e}")
    return None


def create_master_playlist(output_video_base_dir, qualities_details_list, master_m3u8_filename="master.m3u8"):
    """
    在指定的视频基础目录下创建主 M3U8 播放列表。

    Args:
        output_video_base_dir (str): 特定视频的输出根目录 (例如, "video_segments/video_name")
        qualities_details_list (list): 一个字典列表，每个字典包含一个质量流的信息:
            [
                {'suffix': '480p-1500k', 'bandwidth': 1596000, 'resolution': '854x480', 
                 'codecs': 'avc1.64001e,mp4a.40.2', 'media_m3u8_filename': 'video_name-480p-1500k.m3u8'},
                # ... 其他质量
            ]
        master_m3u8_filename (str): 生成的主M3U8文件名。
    """
    master_m3u8_path = os.path.join(output_video_base_dir, master_m3u8_filename)
    os.makedirs(output_video_base_dir, exist_ok=True) # 确保视频根目录存在

    with open(master_m3u8_path, 'w', encoding='utf-8') as f:
        f.write("#EXTM3U\n")
        f.write("#EXT-X-VERSION:3\n") # HLS 版本号

        for quality_info in qualities_details_list:
            quality_suffix = quality_info['suffix']
            bandwidth = quality_info['bandwidth']
            resolution = quality_info.get('resolution')
            codecs = quality_info.get('codecs', "avc1.42001E,mp4a.40.2") # 提供一个通用默认值
            media_m3u8_filename = quality_info['media_m3u8_filename']

            # 媒体播放列表的路径，相对于主播放列表的路径
            # 例如: 480p-1500k/video_name-480p-1500k.m3u8
            relative_media_m3u8_path = os.path.join(quality_suffix, media_m3u8_filename).replace('\\', '/')

            stream_inf_line = f"#EXT-X-STREAM-INF:BANDWIDTH={bandwidth}"
            if resolution:
                stream_inf_line += f",RESOLUTION={resolution}"
            if codecs: # CODECS 对于某些播放器很重要
                stream_inf_line += f",CODECS=\"{codecs}\""
            
            f.write(stream_inf_line + "\n")
            f.write(relative_media_m3u8_path + "\n")
        
    logger.info(f"主 M3U8 播放列表已创建: {master_m3u8_path}")


if __name__ == '__main__':
    SOURCE_VIDEO_FILE = "bbb_sunflower.mp4" 
    BASE_OUTPUT_DIR = "video_segments" # 顶级输出目录, 例如 "video_segments"
    SEGMENT_DURATION = 5 
    VIDEO_FRAMERATE = 30 # 请根据你的视频实际帧率调整

    if not os.path.exists(SOURCE_VIDEO_FILE):
        logger.error(f"源视频文件 '{SOURCE_VIDEO_FILE}' 未找到。")
    else:
        video_name_no_ext = os.path.splitext(os.path.basename(SOURCE_VIDEO_FILE))[0]
        # 特定视频的输出根目录，主M3U8将存放在这里，例如: "video_segments/bbb_sunflower"
        output_dir_for_this_video = os.path.join(BASE_OUTPUT_DIR, video_name_no_ext)

        # --- 定义不同质量的转码参数和信息 ---
        # !! 再次提醒: 请用 ffprobe 检查你的源视频，确认正确的音频流索引用在 -map 指令中
        # !! 例如，如果想用的音频是第一个音频流，则用 '-map', '0:a:0'
        # !! 如果是全局索引为1的流 (假设视频是0)，则用 '-map', '0:1'
        
        qualities_config = [
            {
                'suffix': "480p-1500k",
                'ffmpeg_opts': [
                    '-map', '0:v:0', '-map', '0:a:0', # !! 假设第一个音频流是你想要的
                    '-c:v', 'libx264', '-b:v', '1500k', '-s', '854x480', 
                    '-preset', 'fast', '-g', str(VIDEO_FRAMERATE * 2), 
                    '-keyint_min', str(VIDEO_FRAMERATE), '-sc_threshold', '0',
                    '-c:a', 'aac', '-b:a', '96k', '-ar', '44100', '-ac', '2'
                ],
                'bandwidth': 1500000 + 96000, 
                'resolution': '854x480',
                'codecs': "avc1.64001e,mp4a.40.2" # 示例: H.264 Main@L3.0, AAC-LC
            },
            {
                'suffix': "720p-4000k",
                'ffmpeg_opts': [
                    '-map', '0:v:0', '-map', '0:a:0', # !! 假设第一个音频流是你想要的
                    '-c:v', 'libx264', '-b:v', '4000k', '-s', '1280x720',
                    '-preset', 'fast', '-g', str(VIDEO_FRAMERATE * 2), 
                    '-keyint_min', str(VIDEO_FRAMERATE), '-sc_threshold', '0',
                    '-c:a', 'aac', '-b:a', '128k', '-ar', '44100', '-ac', '2'
                ],
                'bandwidth': 4000000 + 128000,
                'resolution': '1280x720',
                'codecs': "avc1.64001f,mp4a.40.2" # 示例: H.264 Main@L3.1, AAC-LC
            },
            {
                'suffix': "1080p-8000k",
                'ffmpeg_opts': [
                    '-map', '0:v:0', '-map', '0:a:0', # !! 假设第一个音频流是你想要的
                    '-c:v', 'libx264', '-b:v', '8000k', '-s', '1920x1080',
                    '-preset', 'fast', '-g', str(VIDEO_FRAMERATE * 2), 
                    '-keyint_min', str(VIDEO_FRAMERATE), '-sc_threshold', '0',
                    '-c:a', 'aac', '-b:a', '192k', '-ar', '44100', '-ac', '2'
                ],
                'bandwidth': 8000000 + 192000,
                'resolution': '1920x1080',
                'codecs': "avc1.640028,mp4a.40.2" # 示例: H.264 Main@L4.0, AAC-LC
            }
        ]
        
        # 用于传递给主M3U8生成函数的信息列表
        master_playlist_qualities_info = []

        for config in qualities_config:
            logger.info(f"--- 开始处理质量: {config['suffix']} ---")
            # 调用 segment_video 生成该质量的媒体M3U8和TS分片
            # output_dir 是顶级目录，segment_video内部会创建 video_name/quality_suffix
            media_m3u8_path = segment_video(
                SOURCE_VIDEO_FILE, 
                BASE_OUTPUT_DIR, 
                segment_duration=SEGMENT_DURATION,
                quality_suffix=config['suffix'], 
                ffmpeg_encoder_options=config['ffmpeg_opts']
            )
            
            if media_m3u8_path: # 如果成功生成
                master_playlist_qualities_info.append({
                    'suffix': config['suffix'],
                    'bandwidth': config['bandwidth'],
                    'resolution': config.get('resolution'),
                    # 主播放列表需要的是媒体M3U8的文件名，而不是完整路径
                    'media_m3u8_filename': os.path.basename(media_m3u8_path), 
                    'codecs': config.get('codecs') 
                })
            logger.info(f"--- 完成处理质量: {config['suffix']} ---")

        if master_playlist_qualities_info:
            logger.info("--- 开始创建主 M3U8 播放列表 ---")
            create_master_playlist(
                output_dir_for_this_video, # 主M3U8存放在 "video_segments/video_name/"
                master_playlist_qualities_info
            )
        else:
            logger.info("没有成功生成任何质量的媒体流，因此不创建主M3U8。")