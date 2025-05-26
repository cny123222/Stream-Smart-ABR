import subprocess
import os
import logging

# 日志设置
logger = logging.getLogger('SegmentVideo')
if not logger.handlers: # 防止重复添加 handler
    logger.setLevel(logging.INFO)
    ch_seg = logging.StreamHandler()
    formatter_seg = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch_seg.setFormatter(formatter_seg)
    logger.addHandler(ch_seg)

def segment_video(video_path, output_dir, segment_duration=5, 
                  quality_suffix="1080p-8000k", 
                  ffmpeg_encoder_options=None):
    """
    将视频分片为TS文件，并为每个质量等级生成一个M3U8播放列表。
    ffmpeg_encoder_options: 包含转码参数的列表。
        如果进行转码，强烈建议在此列表中包含明确的 '-map' 指令
        来选择要处理的视频和音频流。
        例如: ['-map', '0:v:0', '-map', '0:a:1', 
               '-c:v', 'libx264', '-b:v', '8000k', 
               '-c:a', 'aac', '-b:a', '192k']
    """
    if not os.path.exists(video_path):
        logger.error(f"错误: 视频文件 {video_path} 未找到。")
        return

    video_name_no_ext = os.path.splitext(os.path.basename(video_path))[0]
    
    specific_quality_dir = os.path.join(output_dir, video_name_no_ext, quality_suffix)
    os.makedirs(specific_quality_dir, exist_ok=True)

    ts_output_pattern = os.path.join(specific_quality_dir, f"{video_name_no_ext}-{quality_suffix}-%03d.ts")
    m3u8_filename = f"{video_name_no_ext}-{quality_suffix}.m3u8" # M3U8文件名与质量等级关联
    m3u8_output_path = os.path.join(specific_quality_dir, m3u8_filename)

    cmd = ['ffmpeg', '-y', '-i', video_path] # -y 覆盖输出文件

    has_map_in_options = False
    if ffmpeg_encoder_options:
        cmd.extend(ffmpeg_encoder_options)
        if any(opt == '-map' for opt in ffmpeg_encoder_options):
            has_map_in_options = True
        if not has_map_in_options:
            logger.warning(f"警告: 为 {quality_suffix} 提供的 ffmpeg_encoder_options 中缺少 '-map' 指令。 "
                           "FFmpeg将使用默认流选择规则，这可能不是期望的结果，特别是对于多音轨视频。 "
                           "建议在转码选项中明确指定 '-map'。")
    else: # 默认 codec copy
        logger.info(f"未提供 {quality_suffix} 的转码选项。将使用 '-codec copy'。")
        cmd.extend(['-codec', 'copy'])
        # 对于 codec copy，如果源文件有多个不必要的流，也应该通过 -map 清理。
        # 为简单起见，如果用户选择 codec copy 且不提供 map，我们默认选择第一个视频和第一个音频。
        # 或者，可以考虑用 -map 0 复制所有流，但这取决于源文件的纯净度。
        # 最佳实践是即使用 codec copy，也通过 -map 指定想要的流。
        if not has_map_in_options: # 再次检查，因为上面是针对 ffmpeg_encoder_options 的
             logger.info("Codec copy模式下未提供-map，默认映射第一个视频和第一个音频流(0:v:0? 0:a:0?)")
             cmd.extend(['-map', '0:v:0?', '-map', '0:a:0?'])


    # 添加分片和M3U8生成的选项
    cmd.extend([
        '-f', 'segment',
        '-segment_time', str(segment_duration),
        '-segment_format', 'mpegts',
        '-segment_list', m3u8_output_path,      # 输出M3U8文件
        '-segment_list_type', 'm3u8',           # M3U8类型
        '-segment_list_entry_prefix', '',       # M3U8中TS文件的相对路径前缀 (同目录则为空)
        '-reset_timestamps', '1',               # 重置每个分片的时间戳
        ts_output_pattern                       # TS分片输出模式
    ])

    logger.info(f"执行 FFmpeg ({quality_suffix}): {' '.join(cmd)}")
    try:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate(timeout=3600) # 设置一个较长的超时，例如1小时
        if process.returncode == 0:
            logger.info(f"视频 {quality_suffix} 成功分片并生成M3U8到 {specific_quality_dir}")
        else:
            logger.error(f"FFmpeg 执行错误 ({quality_suffix}): Return code: {process.returncode}")
            logger.error(f"FFmpeg STDOUT: {stdout.decode(errors='ignore')}")
            logger.error(f"FFmpeg STDERR: {stderr.decode(errors='ignore')}")
    except FileNotFoundError:
        logger.error("错误: FFmpeg 命令未找到。请确保已安装 FFmpeg 并将其添加到系统 PATH。")
    except subprocess.TimeoutExpired:
        logger.error(f"FFmpeg 执行超时 ({quality_suffix})。进程将被终止。")
        process.kill()
        stdout, stderr = process.communicate()
        logger.error(f"FFmpeg STDOUT (on timeout): {stdout.decode(errors='ignore')}")
        logger.error(f"FFmpeg STDERR (on timeout): {stderr.decode(errors='ignore')}")
    except Exception as e:
        logger.error(f"执行 FFmpeg 时发生未知错误 ({quality_suffix}): {e}")

if __name__ == '__main__':
    # --- 配置 ---
    # 确保 SOURCE_VIDEO_FILE 是视频文件名
    SOURCE_VIDEO_FILE = "bbb_sunflower.mp4" 
    BASE_SEGMENTS_DIR = "video_segments" # 与服务器的BASE_SEGMENTS_DIR一致
    SEGMENT_DURATION = 5 
    VIDEO_FRAMERATE = 30 # 视频帧率

    if not os.path.exists(SOURCE_VIDEO_FILE):
        logger.error(f"源视频文件 '{SOURCE_VIDEO_FILE}' 未找到。请下载或更新路径。")
    else:
        # --- 为不同质量等级进行分片 ---
        # 对于 bbb_sunflower_1080p_30fps_normal.mp4:
        # 视频流索引 0 (即 0:v:0 或简写为 0:0)
        # MP3 音频流索引 1 (即 0:a:0 如果是文件中的第一个音轨，或 0:1 如果是全局第二个流)
        # AC3 音频流索引 2 (即 0:a:1 如果是文件中的第二个音轨，或 0:2 如果是全局第三个流)
        # 根据你之前的 ffprobe 输出，MP3 音轨是全局索引 1。所以我们用 '-map', '0:1' 或 '-map', '0:a:X' (X根据实际情况定)
        # ffprobe 输出中的 stream index 是全局的。所以视频是 stream 0，MP3是 stream 1。

        # 目标1: 480p @ 1.5Mbps (示例，你可以按需调整)
        options_480p_1500k = [
            '-map', '0:v:0',  # 选择源文件的第一个视频流
            '-map', '0:1',    # 选择源文件的第二个流（假设这是你想要的MP3音轨）
            '-c:v', 'libx264',
            '-b:v', '1500k',
            '-s', '854x480', 
            '-preset', 'fast', # 使用 'fast' 或 'faster' 以加快测试时的转码速度
            '-g', str(VIDEO_FRAMERATE * 2), # 例如每2秒一个关键帧
            '-keyint_min', str(VIDEO_FRAMERATE),
            '-sc_threshold', '0', # 如果想强制固定GOP间隔，而不是让场景变化影响
            '-c:a', 'aac',      # 音频转码为 AAC (更适合流媒体)
            '-b:a', '96k',
            '-ar', '44100',     # 有时指定标准采样率有帮助
            '-ac', '2'          # 确保输出立体声
        ]
        segment_video(SOURCE_VIDEO_FILE, BASE_SEGMENTS_DIR, 
                      segment_duration=SEGMENT_DURATION,
                      quality_suffix="480p-1500k", 
                      ffmpeg_encoder_options=options_480p_1500k)

        # 你可以取消注释并运行其他质量等级的转码和分片
        # 目标2: 720p @ 4Mbps
        options_720p_4000k = [
            '-map', '0:v:0', '-map', '0:1',
            '-c:v', 'libx264', '-b:v', '4000k', '-s', '1280x720',
            '-preset', 'fast', '-g', str(VIDEO_FRAMERATE * 2), '-keyint_min', str(VIDEO_FRAMERATE), '-sc_threshold', '0',
            '-c:a', 'aac', '-b:a', '128k', '-ar', '44100', '-ac', '2'
        ]
        segment_video(SOURCE_VIDEO_FILE, BASE_SEGMENTS_DIR, 
                      segment_duration=SEGMENT_DURATION,
                      quality_suffix="720p-4000k", 
                      ffmpeg_encoder_options=options_720p_4000k)

        # 目标3: 1080p @ 8Mbps
        options_1080p_8000k = [
            '-map', '0:v:0', '-map', '0:1',
            '-c:v', 'libx264', '-b:v', '8000k', '-s', '1920x1080',
            '-preset', 'fast', '-g', str(VIDEO_FRAMERATE * 2), '-keyint_min', str(VIDEO_FRAMERATE), '-sc_threshold', '0',
            '-c:a', 'aac', '-b:a', '192k', '-ar', '44100', '-ac', '2'
        ]
        segment_video(SOURCE_VIDEO_FILE, BASE_SEGMENTS_DIR, 
                      segment_duration=SEGMENT_DURATION,
                      quality_suffix="1080p-8000k", 
                      ffmpeg_encoder_options=options_1080p_8000k)