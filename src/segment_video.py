import subprocess
import os
import logging

# 日志记录器设置
# 使用模块自己的名称作为日志记录器，这是Python的常见做法。
# 这有助于在大型项目中识别日志消息的来源。
logger = logging.getLogger(__name__)
if not logger.handlers: # 确保在模块重新加载时不会多次添加处理器
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    # 一致的日志格式，包括时间戳、日志记录器名称、级别和消息。
    log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(log_formatter)
    logger.addHandler(console_handler)

def segment_video(video_path, output_dir, segment_duration=5,
                  quality_suffix="1080p-8000k",
                  ffmpeg_encoder_options=None):
    """
    将视频分割为传输流（TS）文件，并为特定质量级别生成相应的
    媒体M3U8播放列表，使用FFmpeg的HLS多路复用器。

    此函数为每个质量级别创建专用目录来存储
    其TS分片和媒体M3U8播放列表。

    参数:
        video_path (str): 源视频文件的绝对或相对路径。
        output_dir (str): 所有视频分片处理输出的基础目录
                          （例如，"video_segments_output"）。子目录
                          如"output_dir/video_name/quality_suffix/"将被创建。
        segment_duration (int, 可选): 每个TS分片的目标持续时间
                                      （秒）。默认为5。
        quality_suffix (str, 可选): 此质量级别的描述性后缀，
                                    用于目录和文件命名约定
                                    （例如，"1080p-8000k"）。默认为"1080p-8000k"。
        ffmpeg_encoder_options (list, 可选): 专门用于编码此质量级别的
                                            FFmpeg命令行选项列表。
                                            如果为None，将使用FFmpeg的'-codec copy'，
                                            意味着不重新编码。默认为None。

    返回:
        str 或 None: 如果分割成功，返回生成的媒体M3U8文件的完整路径；
                     否则返回None。
    """
    if not os.path.exists(video_path):
        logger.error(f"Source video file not found: {video_path}")
        return None

    video_name_no_ext = os.path.splitext(os.path.basename(video_path))[0]

    # 构造此特定质量分片和播放列表的输出目录路径。
    # 示例: video_segments_output/bbb_sunflower/1080p-8000k
    specific_quality_dir = os.path.join(output_dir, video_name_no_ext, quality_suffix)
    try:
        os.makedirs(specific_quality_dir, exist_ok=True)
    except OSError as e:
        logger.error(f"Failed to create directory {specific_quality_dir}: {e}")
        return None

    # 定义媒体M3U8文件名及其完整输出路径。
    media_m3u8_filename = f"{video_name_no_ext}-{quality_suffix}.m3u8"
    media_m3u8_output_path = os.path.join(specific_quality_dir, media_m3u8_filename)

    # 定义TS分片的文件名模式。
    # FFmpeg的-hls_segment_filename选项需要
    # 一个可以包含格式化指令（如%05d用于序列号）的模式。
    ts_segment_filename_template = f"{video_name_no_ext}-{quality_suffix}-%05d.ts"
    # 通常更稳健的做法是为FFmpeg提供分片的完整路径模式，
    # 确保它们在预期的'specific_quality_dir'中创建。
    ts_segment_full_path_pattern = os.path.join(specific_quality_dir, ts_segment_filename_template)

    # 基础FFmpeg命令。'-y'覆盖输出文件而不询问。
    cmd = ['ffmpeg', '-y', '-i', video_path]

    # 处理FFmpeg编码器选项和流映射。
    has_map_in_options = False
    if ffmpeg_encoder_options:
        cmd.extend(ffmpeg_encoder_options)
        # 检查用户提供的选项是否包含'-map'指令。
        if any(opt == '-map' for opt in ffmpeg_encoder_options):
            has_map_in_options = True
        if not has_map_in_options:
            logger.warning(
                f"ffmpeg_encoder_options provided for '{quality_suffix}' do not have '-map' directive. "
                f"FFmpeg will use its default stream selection behavior, which may not be desired."
            )
    else:
        # 如果没有特定的编码选项，使用'codec copy'避免重新编码。
        logger.info(f"No transcoding options provided for '{quality_suffix}'. Using '-codec copy'.")
        cmd.extend(['-codec', 'copy'])
        # 对于'codec copy'，如果没有给出'-map'，FFmpeg可能不会选择任何流，或者可能
        # 默认只选择视频。我们添加默认映射来尝试选择第一个视频
        # 和第一个音频流（如果它们存在）。'0:v:0?'和'0:a:0?'是推测性映射；
        # '?'使它们成为可选的，所以如果某种流类型不存在，FFmpeg不会出错。
        if not has_map_in_options: # 如果ffmpeg_encoder_options为None，此条件适用
            logger.info(
                "In '-codec copy' mode and user provided no '-map'. Defaulting to map "
                "first video stream (0:v:0?) and first audio stream (0:a:0?), if available."
            )
            cmd.extend(['-map', '0:v:0?', '-map', '0:a:0?'])

    # 添加HLS特定的FFmpeg选项。
    cmd.extend([
        '-f', 'hls',                             # 输出格式为HLS。
        '-hls_time', str(segment_duration),       # 目标分片持续时间。
        '-hls_playlist_type', 'vod',              # 创建VOD（视频点播）类型播放列表。
                                                  # 直播流使用'event'。
        '-hls_segment_filename', ts_segment_full_path_pattern, # TS文件的完整路径模式。
        '-hls_flags', 'independent_segments',     # 对ABR至关重要：确保每个分片可以
                                                  # 独立解码，不依赖于HLS规范中关键帧之外的先前分片。
        media_m3u8_output_path                    # 生成的媒体M3U8文件的路径。
    ])

    logger.info(f"Executing FFmpeg command for '{quality_suffix}': {' '.join(cmd)}")
    try:
        # 启动FFmpeg进程。stdout和stderr被管道化以捕获输出。
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # 等待进程完成并获取其输出。
        # 设置超时以防止无限阻塞（例如，1小时）。
        stdout, stderr = process.communicate(timeout=3600)

        # 解码并记录FFmpeg的stderr（通常包含进度和错误）。
        stderr_decoded = stderr.decode(errors='ignore')
        if stderr_decoded.strip(): # 如果不为空则记录
            logger.info(f"FFmpeg STDERR for '{quality_suffix}':\n{stderr_decoded}")
        else:
            logger.info(f"FFmpeg STDERR for '{quality_suffix}': No output or only whitespace.")

        # FFmpeg stdout对于HLS多路复用通常为空，除非使用特定选项。
        stdout_decoded = stdout.decode(errors='ignore')
        if stdout_decoded.strip(): # 如果不为空则记录
            logger.info(f"FFmpeg STDOUT for '{quality_suffix}':\n{stdout_decoded}")

        if process.returncode == 0:
            logger.info(
                f"Video successfully segmented for '{quality_suffix}'. "
                f"Media M3U8 generated: {media_m3u8_output_path}"
            )
            return media_m3u8_output_path
        else:
            logger.error(
                f"FFmpeg HLS execution failed for quality '{quality_suffix}'. "
                f"Return code: {process.returncode}. Check STDERR above for details."
            )
            return None

    except FileNotFoundError:
        logger.error(
            "FFmpeg command not found. Please ensure FFmpeg is installed "
            "and its executable is in the system's PATH environment variable."
        )
    except subprocess.TimeoutExpired:
        logger.error(
            f"FFmpeg execution timed out (limit: 3600 seconds) for quality '{quality_suffix}'. "
            "Process will be terminated."
        )
        if process.poll() is None:  # 检查进程是否仍在运行
            process.kill()
            logger.info(f"FFmpeg process for '{quality_suffix}' has been terminated due to timeout.")
        # 尝试在终止后获取任何输出（可能是部分的或空的）
        stdout, stderr = process.communicate()
        logger.error(f"FFmpeg STDOUT (post-timeout, post-termination) for '{quality_suffix}':\n{stdout.decode(errors='ignore')}")
        logger.error(f"FFmpeg STDERR (post-timeout, post-termination) for '{quality_suffix}':\n{stderr.decode(errors='ignore')}")
    except Exception as e:
        # 捕获在FFmpeg执行期间的任何其他意外异常。
        logger.error(f"Unexpected error occurred during FFmpeg execution for quality '{quality_suffix}': {e}", exc_info=True)
    return None


def create_master_playlist(output_video_base_dir, qualities_details_list,
                           master_m3u8_filename="master.m3u8"):
    """
    创建一个主M3U8播放列表文件，指向多个媒体M3U8播放列表
    用于不同质量级别（自适应比特率流）。

    参数:
        output_video_base_dir (str): 处理过的视频的根输出目录，
                                     主M3U8文件将保存在其中。
                                     示例: "video_segments_output/video_name"。
        qualities_details_list (list): 字典列表。每个字典必须
                                       包含一个质量流/变体的详细信息，
                                       包括'suffix'、'bandwidth'、
                                       'media_m3u8_filename'，以及可选的
                                       'resolution'和'codecs'。
                                       示例:
                                       [
                                           {'suffix': '480p-1500k', 'bandwidth': 1596000,
                                            'resolution': '854x480', 'codecs': 'avc1.64001e,mp4a.40.2',
                                            'media_m3u8_filename': 'video_name-480p-1500k.m3u8'},
                                           # ... 其他质量流字典 ...
                                       ]
        master_m3u8_filename (str, 可选): 主M3U8播放列表的期望文件名。
                                          默认为"master.m3u8"。
    """
    master_m3u8_path = os.path.join(output_video_base_dir, master_m3u8_filename)
    try:
        os.makedirs(output_video_base_dir, exist_ok=True)
    except OSError as e:
        logger.error(f"Failed to create master playlist directory {master_m3u8_path}: {e}")
        return

    try:
        with open(master_m3u8_path, 'w', encoding='utf-8') as f:
            f.write("#EXTM3U\n")            # 标准M3U8头部
            f.write("#EXT-X-VERSION:3\n")   # 指定HLS协议版本（3是广泛兼容的）

            for quality_info in qualities_details_list:
                # 提取每个质量流的详细信息。
                quality_suffix = quality_info['suffix']
                bandwidth = quality_info['bandwidth']
                resolution = quality_info.get('resolution') # 分辨率是可选的
                # 如果未为变体指定编解码器，提供通用默认值。
                codecs = quality_info.get('codecs', "avc1.42001E,mp4a.40.2") # 通用H.264 + AAC
                media_m3u8_filename = quality_info['media_m3u8_filename']

                # 构造从主播放列表位置到媒体播放列表的相对路径。
                # 示例: "480p-1500k/video_name-480p-1500k.m3u8"
                # M3U8路径应使用正斜杠，无论操作系统如何。
                relative_media_m3u8_path = os.path.join(quality_suffix, media_m3u8_filename).replace('\\', '/')

                # 构建#EXT-X-STREAM-INF标签行。
                stream_inf_line = f"#EXT-X-STREAM-INF:BANDWIDTH={bandwidth}"
                if resolution:
                    stream_inf_line += f",RESOLUTION={resolution}"
                if codecs: # CODECS属性对播放器兼容性很重要。
                    stream_inf_line += f",CODECS=\"{codecs}\""

                f.write(stream_inf_line + "\n")
                f.write(relative_media_m3u8_path + "\n")

        logger.info(f"Master M3U8 playlist successfully created at: {master_m3u8_path}")
    except IOError as e:
        logger.error(f"Failed to write master M3U8 playlist {master_m3u8_path}: {e}")
    except KeyError as e:
        logger.error(f"Missing expected key in qualities_details_list for master playlist generation: {e}. "
                     "Each item needs 'suffix', 'bandwidth', and 'media_m3u8_filename'.")


if __name__ == '__main__':
    # --- 脚本执行配置 ---
    SOURCE_VIDEO_FILE = "bbb_sunflower.mp4"     # 输入视频文件路径。
    BASE_OUTPUT_DIR = "video_segments"          # 所有输出的顶级目录。
    SEGMENT_DURATION = 5                        # 每个HLS分片的目标持续时间（秒）。
    VIDEO_FRAMERATE = 30 # 用于GOP计算的假定帧率。如果您的源不同，请调整。
                         # 使用'ffprobe -v error -select_streams v:0 -show_entries stream=r_frame_rate -of default=noprint_wrappers=1:nokey=1 your_video.mp4'
                         # 获取实际帧率（例如，30000/1001表示29.97fps）。

    # --- 选择ABR阶梯配置 ---
    # 设置为True表示5级阶梯（包括2160p、360p）。
    # 设置为False表示3级阶梯（如之前定义的1080p、720p、480p）。
    USE_FIVE_LEVELS_LADDER = True # 修改此项以切换配置。

    # --- 源视频基本检查 ---
    if not os.path.exists(SOURCE_VIDEO_FILE):
        logger.error(f"Source video file '{SOURCE_VIDEO_FILE}' not found. Please check the path. Exiting.")
    else:
        video_name_no_ext = os.path.splitext(os.path.basename(SOURCE_VIDEO_FILE))[0]
        # 定义此特定视频处理文件的根输出目录。
        # 主M3U8播放列表将放置在这里。
        # 示例: "video_segments_output/bbb_sunflower/"
        output_dir_for_this_video = os.path.join(BASE_OUTPUT_DIR, video_name_no_ext)

        # --- 全面的ABR阶梯配置 ---
        # 此字典定义所有可用的质量配置文件。
        # 脚本将根据USE_FIVE_LEVELS_LADDER选择其中的子集。
        #
        # FFmpeg选项的重要说明:
        # - '-map 0:v:0 -map 0:a:0': 假设主要视频和音频是其类型的第一个。
        #   始终使用'ffprobe'验证特定源视频。
        # - '-g <GOP_size>': 图像组大小。对于HLS通常是帧率的2倍。
        # - '-keyint_min <min_keyframe_interval>': 最小关键帧间隔。通常与帧率相同。
        # - '-preset': 控制编码速度与压缩效率。'fast'是常见的平衡。
        #   其他选项: 'ultrafast'、'superfast'、'veryfast'、'faster'、'medium'、'slow'、'slower'、'veryslow'。
        # - 编解码器字符串: 'avc1...'表示H.264，'mp4a.40.2'表示AAC-LC。这些应与您的编码匹配。
        #   'avc1...'中的十六进制部分（例如，64001e）表示配置文件和级别。
        #   如果更改配置文件/级别，请查阅H.264文档或使用工具确定正确值。

        all_quality_profiles = {
            "360p-800k": {
                'ffmpeg_opts': [
                    '-map', '0:v:0', '-map', '0:a:0',
                    '-c:v', 'libx264', '-b:v', '800k', '-maxrate', '856k', '-bufsize', '1200k', '-s', '640x360',
                    '-preset', 'fast', '-g', str(VIDEO_FRAMERATE * 2),
                    '-keyint_min', str(VIDEO_FRAMERATE), '-sc_threshold', '0',
                    '-c:a', 'aac', '-b:a', '64k', '-ar', '44100', '-ac', '2'
                ],
                'bandwidth': 800000 + 64000, # 总估计带宽（视频 + 音频）
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
                'codecs': "avc1.4d001e,mp4a.40.2" # H.264 Main@L3.0, AAC-LC（示例，如需要请调整）
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
                    '-preset', 'fast', # 如果质量至关重要且时间允许，考虑4K使用'medium'或'slow'
                    '-g', str(VIDEO_FRAMERATE * 2),
                    '-keyint_min', str(VIDEO_FRAMERATE), '-sc_threshold', '0',
                    '-c:a', 'aac', '-b:a', '256k', '-ar', '44100', '-ac', '2'
                ],
                'bandwidth': 16000000 + 256000,
                'resolution': '3840x2160',
                'codecs': "avc1.640033,mp4a.40.2" # H.264 High@L5.1或L5.2。检查FFmpeg输出获取实际配置文件/级别。
            }
        }

        # 根据标志选择要处理的质量配置文件。
        if USE_FIVE_LEVELS_LADDER:
            # 定义5级阶梯的质量顺序。
            # 此顺序也影响主M3U8中的顺序，如果稍后不按带宽排序的话。
            active_quality_keys = ["360p-800k", "480p-1500k", "720p-4000k", "1080p-8000k", "2160p-16000k"]
            logger.info("Configuration selected: 5 quality levels for segmentation.")
        else:
            active_quality_keys = ["480p-1500k", "720p-4000k", "1080p-8000k"]
            logger.info("Configuration selected: 3 quality levels for segmentation.")

        # 准备要实际处理的质量配置列表。
        qualities_to_process_list = []
        for key in active_quality_keys:
            if key in all_quality_profiles:
                # 创建副本并添加'suffix'键以便使用。
                profile_config = all_quality_profiles[key].copy()
                profile_config['suffix'] = key # 后缀就是键本身。
                qualities_to_process_list.append(profile_config)
            else:
                logger.warning(f"Configuration for quality profile key '{key}' not found in 'all_quality_profiles'. Skipping.")

        # 此列表将存储生成主M3U8播放列表所需的详细信息。
        master_playlist_variant_streams = []

        # 处理每个选定的质量配置文件。
        for profile_config_item in qualities_to_process_list:
            current_quality_suffix = profile_config_item['suffix']
            logger.info(f"--- Starting processing for quality profile: {current_quality_suffix} ---")

            # 为当前配置文件调用分割函数。
            media_m3u8_path_generated = segment_video(
                video_path=SOURCE_VIDEO_FILE,
                output_dir=BASE_OUTPUT_DIR, # segment_video将根据video_name和suffix创建子目录
                segment_duration=SEGMENT_DURATION,
                quality_suffix=current_quality_suffix,
                ffmpeg_encoder_options=profile_config_item['ffmpeg_opts']
            )

            if media_m3u8_path_generated: # 如果此配置文件的分割成功
                master_playlist_variant_streams.append({
                    'suffix': current_quality_suffix,
                    'bandwidth': profile_config_item['bandwidth'],
                    'resolution': profile_config_item.get('resolution'),
                    'media_m3u8_filename': os.path.basename(media_m3u8_path_generated),
                    'codecs': profile_config_item.get('codecs')
                })
            else:
                logger.error(f"Quality profile segmentation failed: {current_quality_suffix}. "
                             "This profile will be excluded from the master playlist.")
            logger.info(f"--- Finished processing for quality profile: {current_quality_suffix} ---")

        # 处理完所有选定的质量配置文件后，创建主M3U8播放列表。
        if master_playlist_variant_streams:
            logger.info("--- Starting creation of master M3U8 playlist ---")
            create_master_playlist(
                output_video_base_dir=output_dir_for_this_video,
                qualities_details_list=master_playlist_variant_streams
                # master_m3u8_filename默认为"master.m3u8"
            )
        else:
            logger.warning(
                "After processing all selected profiles, no media streams were successfully generated. "
                "Master M3U8 playlist will not be created."
            )
        logger.info("All HLS segmentation processing complete.")