import socket
import os
import threading
import time
import logging
import vlc
from urllib.parse import unquote, urlparse # 用于解码MRL中的路径

# --- ABR and QoE imports ---
from abr_algorithm import ABRAlgorithm, QualityLevel
from qoe_metrics import QoECalculator
from buffer_manager import BufferManager

# --- Configuration ---
SERVER_HOST = '127.0.0.1'
SERVER_PORT = 8081 # 确保与服务器端口一致
DOWNLOAD_DIR = "./TMP"
BUFFER_SIZE = 4096
FETCH_AHEAD_TARGET = 3     # 目标：在播放指针之后，保持这么多分片已下载并加入播放列表
MIN_BUFFER_TO_START_PLAY = 1 # 至少下载并成功加入列表多少个分片后开始播放
SOCKET_TIMEOUT_SECONDS = 10  # 网络操作的通用超时（秒）
RETRY_DOWNLOAD_DELAY = 2     # 下载失败时的重试延迟（秒）
PROGRESS_LOG_INTERVAL = 1.0  # 每隔多少秒打印一次播放进度

# --- Logger Setup ---
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(threadName)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler()])
logger = logging.getLogger('StreamingClientVLC')

# --- VLC Player Setup & State ---
vlc_instance = None
media_list_player = None
media_list = None 
media_player = None 
player_event_manager = None

# --- Streaming State & Control ---
keep_streaming_and_downloading = threading.Event() # 主控制信号：是否继续流式传输和下载
all_segments_processed_by_downloader = threading.Event() # 下载线程是否处理完了所有预期的分片
player_signaled_true_end = threading.Event() # 播放器是否真正播放到所有内容的末尾或出错

download_thread_instance = None 
# 存储已下载分片的信息: {original_segment_idx: {'path': local_abs_path, 'mrl': mrl_str, 'downloaded': bool, 'added_to_playlist': bool}}
downloaded_segments_info = {} 
lock = threading.Lock() # 用于保护 downloaded_segments_info 的并发访问

# 从服务器获取的元数据
total_segments_from_server = 0
avg_segment_duration_from_server = 0.0 # 秒

# 播放跟踪
currently_playing_original_idx = -1 # VLC当前播放媒体对应的原始分片索引 (从0开始)
last_cleaned_original_idx = -1   # 上一个被清理的分片的原始索引，用于避免重复清理已清理过的

# --- ABR & QoE Integration ---
abr_algorithm = None
qoe_calculator = None
buffer_manager = None
abr_enabled = True  # 是否启用ABR
current_quality_suffix = "480p-1500k"  # 当前质量等级

def path_to_mrl(local_file_path):
    """将本地文件绝对路径转换为VLC可以使用的MRL (file:///...)"""
    abs_path = os.path.abspath(local_file_path)
    if os.name == 'nt': # Windows
        return 'file:///' + abs_path.replace('\\', '/')
    else: # macOS, Linux
        return f'file://{abs_path}'

def mrl_to_local_os_path(mrl_string):
    """从MRL（通常由VLC事件返回）转换回本地操作系统路径"""
    if not mrl_string: return None
    try:
        parsed = urlparse(mrl_string)
        if parsed.scheme != 'file': return None #只处理file://协议
        path = unquote(parsed.path) 
        # Windows MRL路径 (如 /C:/Users/...) 需要移除开头的斜杠
        if os.name == 'nt' and len(path) > 1 and path[0] == '/' and path[2] == ':':
            path = path[1:]
        return os.path.normpath(path)
    except Exception as e:
        logger.error(f"解析MRL时出错 '{mrl_string}': {e}")
        return None

def initialize_player():
    """初始化VLC播放器实例、媒体列表播放器、媒体播放器和事件管理器"""
    global vlc_instance, media_list_player, media_list, media_player, player_event_manager
    if vlc_instance is None: # 确保只初始化一次
        instance_args = ['--no-video-title-show', '--quiet']
        # network_caching: libVLC的网络缓存(毫秒)。对于本地文件列表，这个影响不大，
        # 但如果VLC认为它在处理流式内容，可能会有帮助。我们主要依赖自己的应用层缓冲。
        instance_args.append(f'--network-caching=1000') # 例如1秒VLC内部网络缓冲
        
        vlc_instance = vlc.Instance(instance_args)
        
        media_list = vlc_instance.media_list_new([]) # 创建一个空的媒体列表
        media_list_player = vlc_instance.media_list_player_new() # 创建媒体列表播放器
        media_player = vlc_instance.media_player_new() # 创建一个媒体播放器实例
        media_list_player.set_media_player(media_player) # 将MediaPlayer实例设置给MediaListPlayer
        media_list_player.set_media_list(media_list)     # 将MediaList设置给MediaListPlayer
        
        # 绑定VLC事件回调
        player_event_manager = media_player.event_manager()
        player_event_manager.event_attach(vlc.EventType.MediaPlayerMediaChanged, on_media_changed_callback)
        player_event_manager.event_attach(vlc.EventType.MediaPlayerEndReached, on_item_end_reached_callback)
        player_event_manager.event_attach(vlc.EventType.MediaPlayerEncounteredError, on_player_error_callback)
        player_event_manager.event_attach(vlc.EventType.MediaPlayerPositionChanged, on_player_position_changed_callback)
        player_event_manager.event_attach(vlc.EventType.MediaPlayerBuffering, on_player_buffering_callback)
        logger.info("VLC Player initialized.")

# --- VLC Event Callbacks ---
def on_media_changed_callback(event):
    """当VLC播放器开始播放播放列表中的新媒体项时调用"""
    global currently_playing_original_idx, downloaded_segments_info, last_cleaned_original_idx, \
           buffer_manager, qoe_calculator
    
    new_media_instance = media_player.get_media() # 获取当前正在播放的媒体对象
    if not new_media_instance:
        logger.info("MediaPlayerMediaChanged: 新媒体项为 None (播放列表可能为空或已结束).")
        with lock: currently_playing_original_idx = -1
        return

    new_mrl = new_media_instance.get_mrl() # 获取新媒体的MRL
    new_path_playing_abs = mrl_to_local_os_path(new_mrl) # 将MRL转换为本地路径
    
    found_idx = -1 # 用于存储找到的原始分片索引
    if new_path_playing_abs:
        logger.info(f"MediaPlayerMediaChanged: 现在播放 '{os.path.basename(new_path_playing_abs)}'")
        with lock: # 保护对 downloaded_segments_info 的访问
            # 根据当前播放的文件的绝对路径，在我们的跟踪字典中查找其对应的原始分片索引
            for idx, info in downloaded_segments_info.items():
                if info.get('path') and os.path.normpath(info['path']) == new_path_playing_abs:
                    found_idx = idx
                    break
            currently_playing_original_idx = found_idx # 更新全局变量
            
            if found_idx != -1:
                logger.info(f"当前播放分片的原始索引更新为: {found_idx}")
                # --- 文件清理逻辑 ---
                # 清理那些原始索引比 (当前播放索引 - (预设的提前下载数 + 一个小的额外保留数)) 更小的文件
                # 目的是保留一些最近播放过的分片在磁盘上，以备可能的快速回退查找，同时控制磁盘占用。
                cleanup_older_than_this_original_idx = found_idx - (FETCH_AHEAD_TARGET + 1) 
                keys_to_delete_from_map = [] # 记录要从字典中删除的键
                for idx_to_check, info_to_delete in downloaded_segments_info.items():
                    if idx_to_check <= cleanup_older_than_this_original_idx and idx_to_check <= last_cleaned_original_idx:
                        continue # 已经清理过的，跳过
                    elif idx_to_check <= cleanup_older_than_this_original_idx:
                        path_to_delete = info_to_delete.get('path')
                        if path_to_delete and os.path.exists(path_to_delete):
                            try:
                                os.remove(path_to_delete)
                                logger.debug(f"已清理旧分片文件: {os.path.basename(path_to_delete)} (索引 {idx_to_check})")
                            except OSError as e:
                                logger.warning(f"清理文件失败 {path_to_delete}: {e}")
                        keys_to_delete_from_map.append(idx_to_check)
                        
                if keys_to_delete_from_map:
                    for key in keys_to_delete_from_map:
                        del downloaded_segments_info[key]
                    last_cleaned_original_idx = max(keys_to_delete_from_map)
                    logger.debug(f"已从跟踪字典中移除 {len(keys_to_delete_from_map)} 个旧分片条目。")
            else: 
                logger.warning(f"无法将当前播放的MRL {new_mrl} (路径: {new_path_playing_abs}) 映射回原始分片索引。")
    else: 
        logger.warning(f"MediaPlayerMediaChanged: 无法从MRL获取路径: {new_mrl}")
        with lock: currently_playing_original_idx = -1

    # --- Update buffer manager and QoE ---
    # 更新缓冲区管理器
    if buffer_manager and found_idx != -1:
        buffer_manager.update_playing_position(found_idx)
    
    # 记录播放开始（仅首次）
    if qoe_calculator and found_idx == 0:
        qoe_calculator.record_playback_start()


def on_item_end_reached_callback(event):
    """当MediaListPlayer播放完列表中的一个媒体项后触发。注意：这不一定是整个流的结束。"""
    logger.debug("MediaPlayerEndReached (on_item_end_reached_callback): 当前媒体项播放完毕。")
    # MediaListPlayer会自动尝试播放列表中的下一项（如果存在）。
    # 真正的"整个流结束"判断逻辑：
    # 1. 下载线程已确认所有分片都已处理完毕 (all_segments_processed_by_downloader.is_set())
    # 2. 并且播放器状态表明它已停止或结束 (player_state in [vlc.State.Ended, vlc.State.Stopped])
    # 3. 并且播放器当前没有关联的媒体 (media_player.get_media() is None)，这通常表示播放列表已空且播放完毕。
    if all_segments_processed_by_downloader.is_set():
        player_state = media_list_player.get_state()
        current_media_in_player_obj = media_player.get_media() 
        if current_media_in_player_obj is None and \
           player_state in [vlc.State.Ended, vlc.State.Stopped, vlc.State.NothingSpecial]:
            logger.info("所有分片已由下载器处理完毕，且播放器指示所有媒体已结束。发送真正结束信号。")
            player_signaled_true_end.set() # 设置事件，通知主控制循环可以结束了

def on_player_error_callback(event):
    """当VLC播放器遇到错误时调用"""
    logger.error("VLC Player 遇到错误。发送停止信号。")
    player_signaled_true_end.set() # 发生错误，也认为播放会话应结束
    keep_streaming_and_downloading.clear() # 停止下载线程

def on_player_buffering_callback(event):
    """当VLC播放器进入或更新缓冲状态时调用"""
    buffer_percentage = event.u.new_cache # 获取缓冲百分比 (0.0 到 100.0)
    logger.info(f"VLC Player 正在缓冲: {buffer_percentage:.1f}%")
    if buffer_percentage < 100: # 这个条件可能一直成立直到播放开始/结束
        logger.info("播放器正在缓冲... (这通常表现为视频播放过程中的卡顿或加载)")

def on_player_position_changed_callback(event):
    """当播放位置改变时调用，用于更新(控制台)总进度"""
    global total_segments_from_server, avg_segment_duration_from_server, currently_playing_original_idx
    
    if not (media_player and total_segments_from_server > 0 and avg_segment_duration_from_server > 0):
        return # 如果没有元数据或播放器，则不处理

    current_original_idx = -1
    with lock: current_original_idx = currently_playing_original_idx # 获取当前播放分片的原始索引
    
    pos_ratio_current_item = media_player.get_position() # 获取当前播放分片内部的播放进度比例 (0.0 到 1.0)
    
    if current_original_idx != -1 : # 确保我们知道当前在播放哪个原始分片
        # 使用当前分片的实际时长（如果能从VLC获取到）来计算当前分片内的已播放时长，会更精确
        current_item_actual_length_s = avg_segment_duration_from_server # 默认使用平均时长
        length_ms_from_player = media_player.get_length() # 获取当前媒体项的总时长(毫秒)
        if length_ms_from_player > 0: # 如果能获取到，则使用精确值
            current_item_actual_length_s = length_ms_from_player / 1000.0
        
        duration_in_current_segment_s = pos_ratio_current_item * current_item_actual_length_s
        # 已完整播放的那些分片的总时长 (基于平均时长估算)
        duration_of_played_segments = current_original_idx * avg_segment_duration_from_server
        
        current_time_overall_seconds = duration_of_played_segments + duration_in_current_segment_s
        total_time_overall_seconds = total_segments_from_server * avg_segment_duration_from_server
        
        # 为了避免日志刷屏，可以设置一个打印间隔
        if not hasattr(on_player_position_changed_callback, "last_log_time") or \
           time.time() - on_player_position_changed_callback.last_log_time > PROGRESS_LOG_INTERVAL:
            
            progress_msg = (f"进度: {time.strftime('%H:%M:%S', time.gmtime(current_time_overall_seconds))}/"
                            f"{time.strftime('%H:%M:%S', time.gmtime(total_time_overall_seconds))} "
                            f"(播放原始分片: {current_original_idx + 1}/{total_segments_from_server}) " # 显示基于1的索引
                            f"播放器状态: {media_list_player.get_state()}")
            logger.info(progress_msg)
            on_player_position_changed_callback.last_log_time = time.time()


def add_segment_to_vlc_playlist(local_segment_path, segment_idx_original):
    """将下载好的本地分片文件加入VLC的播放列表，并更新跟踪信息"""
    global media_list, media_list_player, vlc_instance, downloaded_segments_info
    if not os.path.exists(local_segment_path):
        logger.error(f"分片 {local_segment_path} 在加入播放列表时未找到。"); return

    abs_path = os.path.abspath(local_segment_path)
    media = vlc_instance.media_new(abs_path) # python-vlc可以直接处理本地绝对路径
    if not media: logger.error(f"为路径创建媒体对象失败: {abs_path}"); return
    
    mrl = media.get_mrl() # 获取VLC内部将使用的MRL

    with lock: # 保护对 media_list 和 downloaded_segments_info 的访问
        media_list.lock() # 锁定媒体列表以进行修改
        added_ok = media_list.add_media(media) # add_media 返回0表示成功, -1表示失败
        media_list.unlock()
        
        if added_ok == 0: # 添加成功
            # 更新或添加分片信息到我们的跟踪字典
            info = downloaded_segments_info.get(segment_idx_original, {'path': abs_path, 'downloaded': True})
            info['mrl'] = mrl
            info['in_playlist'] = True # 标记已加入播放列表
            downloaded_segments_info[segment_idx_original] = info
            logger.info(f"已添加分片 (原始索引:{segment_idx_original}, 文件名:'{os.path.basename(abs_path)}') 到VLC播放列表。"
                        f"播放列表当前项目数: {media_list.count()}")
        else:
            logger.error(f"添加媒体 {abs_path} (MRL: {mrl}) 到 MediaList 失败。错误码: {added_ok}")
            media.release(); return # 如果添加失败，释放我们创建的media对象
            
    media.release() # MediaList 会持有它自己的引用，所以我们可以释放我们创建的这个

    # 检查是否应该开始播放
    current_player_state = media_list_player.get_state()
    # 如果播放器未处于播放相关状态，并且播放列表中的媒体数量达到了最小启动缓冲要求
    if current_player_state not in [vlc.State.Playing, vlc.State.Opening, vlc.State.Buffering, vlc.State.Paused] and \
       media_list.count() >= MIN_BUFFER_TO_START_PLAY:
        logger.info(f"缓冲已达到 ({media_list.count()} 个项目 >= 最低启动要求 {MIN_BUFFER_TO_START_PLAY} 个)。启动播放...")
        # MediaListPlayer 会从MediaList的当前项（通常是第一项，如果列表是新的）开始播放
        # 如果列表之前有内容且已播放一部分，play()会从当前指针继续
        media_list_player.play() 
        time.sleep(0.1) # 给VLC一点时间响应play命令
        if media_list_player.get_state() != vlc.State.Playing:
            logger.warning(f"启动播放失败。当前播放器状态: {media_list_player.get_state()}")


def receive_exact_bytes(sock, num_bytes):
    """从socket精确接收指定数量的字节，包含超时处理。"""
    data = b''
    bytes_to_receive_total = num_bytes
    try:
        while len(data) < bytes_to_receive_total:
            remaining_bytes = bytes_to_receive_total - len(data)
            sock.settimeout(SOCKET_TIMEOUT_SECONDS) 
            chunk = sock.recv(min(BUFFER_SIZE, remaining_bytes))
            # sock.settimeout(None) # 应在循环结束后或finally中重置
            if not chunk: # 套接字关闭
                raise ConnectionError("Socket connection broken while receiving data.")
            data += chunk
    except socket.timeout:
        logger.error(f"Socket接收数据超时：期望 {bytes_to_receive_total} 字节, 已收到 {len(data)} 字节。")
        raise ConnectionError("Socket timeout during data reception.")
    finally:
        sock.settimeout(None) # 完成后重置超时
    return data


def download_segments_task(client_socket_ref, video_name, initial_quality_suffix):
    """后台下载线程的目标函数。"""
    global total_segments_from_server, keep_streaming_and_downloading, \
           all_segments_processed_by_downloader, downloaded_segments_info, \
           currently_playing_original_idx, abr_algorithm, qoe_calculator, \
           buffer_manager, current_quality_suffix

    logger.info("[DL_Thread] 下载线程已启动。")

    # --- ABR相关变量 ---
    current_download_quality = initial_quality_suffix
    last_abr_decision_time = time.time()
    abr_decision_interval = 5.0  # ABR决策间隔（秒）

    # 从原始索引0开始，直到所有分片下载完毕或收到停止信号
    # dbg
    # for segment_idx_to_download in range(total_segments_from_server):
    first = [0]*115
    segment_idx_to_download = -1
    while segment_idx_to_download < total_segments_from_server:
        segment_idx_to_download += 1
        if not first[segment_idx_to_download]:
            print(segment_idx_to_download)
            print(total_segments_from_server)
            first[segment_idx_to_download] = True
        if not keep_streaming_and_downloading.is_set():
            logger.info("[DL_Thread] 收到停止信号。正在终止下载线程。")
            break # 外部要求停止

        # --- 简化的ABR决策逻辑 ---
        current_time = time.time()
        if (abr_enabled and abr_algorithm and 
            current_time - last_abr_decision_time >= abr_decision_interval):
            
            # 获取缓冲区状态
            buffer_level = 0.0
            if buffer_manager:
                try:
                    # 使用buffer_manager的适当方法获取缓冲区水位
                    buffer_level = buffer_manager.get_buffer_level_seconds()
                except:
                    # 如果上面的方法不存在，尝试其他可能的方法名
                    whoknows
                    # try:
                    #     buffer_level = sum(info.segment_duration for info in buffer_manager.buffer_queue)
                    # except:
                    #     buffer_level = 0.0
            
            # 简化的ABR决策
            old_quality = current_download_quality
            try:
                # 尝试调用ABR决策方法
                quality_level = None
                reason = ""
                
                # 尝试使用原始期望的接口
                try:
                    quality_level, reason = abr_algorithm.decide_next_quality()
                    current_download_quality = quality_level.suffix
                except:
                    # 备选方案: 使用更简单的决策逻辑
                    for quality in abr_algorithm.QUALITY_LEVELS:
                        if quality.suffix == old_quality:
                            quality_level = quality
                            break
                    if quality_level:
                        current_download_quality = quality_level.suffix
                        reason = "保持当前质量"
            except Exception as e:
                logger.error(f"[ABR] 决策错误: {e}")
                current_download_quality = initial_quality_suffix  # 出错时使用初始质量
                reason = "决策错误，使用默认质量"
            
            # 记录质量切换
            if qoe_calculator and old_quality != current_download_quality:
                try:
                    # 获取网络吞吐量
                    throughput = 0.0
                    try:
                        throughput = abr_algorithm.network_metrics.get_recent_throughput() / 1000  # kbps
                    except:
                        pass
                    
                    qoe_calculator.record_quality_switch(
                        old_quality, current_download_quality, reason, buffer_level, throughput
                    )
                except Exception as e:
                    logger.error(f"[QoE] 记录质量切换错误: {e}")
            
            last_abr_decision_time = current_time
            if old_quality != current_download_quality:
                logger.info(f"[ABR] 质量切换: {old_quality} -> {current_download_quality} (原因: {reason})")
            else:
                logger.debug(f"[ABR] 质量保持: {current_download_quality} (原因: {reason})")

        # --- 缓冲区管理逻辑简化 ---
        # 不做复杂缓冲区检查，只在每次下载前检查一次当前播放位置和已下载数量
        with lock:
            current_playing_idx = currently_playing_original_idx
            num_segments_ahead = segment_idx_to_download - (current_playing_idx if current_playing_idx != -1 else -1)
        
        player_active = media_list_player.get_state() in [vlc.State.Playing, vlc.State.Buffering, vlc.State.Opening]
        
        # 如果缓冲区已满（已下载比当前播放的超前太多），等待一小段时间
        if player_active and num_segments_ahead > FETCH_AHEAD_TARGET * 2:
            logger.info(f"[DL_Thread] 缓冲区充足，暂停下载... 当前超前: {num_segments_ahead}, 目标: {FETCH_AHEAD_TARGET}")
            time.sleep(0.5)
            segment_idx_to_download -= 1
            continue
            
        # 检查是否已下载过这个分片
        with lock:
            if segment_idx_to_download in downloaded_segments_info and \
               downloaded_segments_info[segment_idx_to_download].get('downloaded', False):
                continue # 已下载，处理下一个索引
        
        # --- 构造文件名和请求路径（使用当前ABR决策的质量）---
        unique_ts_id = f"{segment_idx_to_download:03d}_{int(time.time()*1000000)}" 
        segment_filename_on_server = f"{video_name}-{current_download_quality}-{segment_idx_to_download:03d}.ts"
        request_path_on_server = f"{video_name}/{current_download_quality}/{segment_filename_on_server}"
        local_segment_filename = f"temp_{video_name}_{current_download_quality}_{unique_ts_id}.ts"
        local_segment_path = os.path.join(DOWNLOAD_DIR, local_segment_filename)

        logger.info(f"[DL_Thread] 请求分片 (原始索引 {segment_idx_to_download}, 质量 {current_download_quality}): {request_path_on_server}")
        request_message = f"GET {request_path_on_server}\n"
        download_start_time = time.time()
        
        try:
            if not client_socket_ref or getattr(client_socket_ref, '_closed', False) or client_socket_ref.fileno() == -1:
                logger.error("[DL_Thread] 套接字已关闭或无效。无法下载。正在停止任务。")
                keep_streaming_and_downloading.clear(); break

            client_socket_ref.sendall(request_message.encode('utf-8'))
            header_data = b""
            client_socket_ref.settimeout(SOCKET_TIMEOUT_SECONDS) 
            try:
                while not header_data.endswith(b"\n"):
                    byte = client_socket_ref.recv(1)
                    if not byte: 
                        raise ConnectionError("服务器关闭连接 (分片头部)")
                    header_data += byte
            except socket.timeout:
                logger.error(f"[DL_Thread] 获取分片 {segment_idx_to_download} 头部超时。将在 {RETRY_DOWNLOAD_DELAY}秒后重试。")
                time.sleep(RETRY_DOWNLOAD_DELAY); continue # 重试当前分片的下载
            finally: client_socket_ref.settimeout(None)

            header_str = header_data.decode('utf-8').strip()
            if header_str.startswith("OK "):
                try: 
                    expected_size = int(header_str.split(" ", 1)[1])
                except (IndexError, ValueError): 
                    logger.error(f"[DL_Thread] 分片 {segment_idx_to_download} 的OK响应格式无效: {header_str}")
                    continue
                
                segment_data = receive_exact_bytes(client_socket_ref, expected_size)
                download_end_time = time.time()
                download_duration = download_end_time - download_start_time
                
                if len(segment_data) != expected_size:
                    logger.error(f"[DL_Thread] 分片 {segment_idx_to_download} 大小不匹配：期望 {expected_size}，实际 {len(segment_data)}")
                    continue
                
                with open(local_segment_path, 'wb') as f: 
                    f.write(segment_data)
                
                # --- 更新ABR网络指标 ---
                if abr_algorithm:
                    try:
                        buffer_level = 0.0
                        if buffer_manager:
                            try:
                                buffer_level = buffer_manager.get_buffer_level_seconds() 
                            except:
                                try:
                                    buffer_level = sum(info.segment_duration for info in buffer_manager.buffer_queue)
                                except:
                                    pass
                                    
                        # 尝试用不同的方法名调用
                        try:
                            abr_algorithm.update_network_metrics(expected_size, download_duration, buffer_level)
                        except:
                            # 如果上面的接口不存在，尝试其他可能的接口
                            try:
                                abr_algorithm.network_metrics.add_download_sample(expected_size, download_duration)
                                abr_algorithm.network_metrics.add_buffer_level(buffer_level)
                            except:
                                logger.error("[ABR] 无法更新网络指标")
                    except Exception as e:
                        logger.error(f"[ABR] 更新网络指标错误: {e}")
                
                # --- 更新QoE统计 ---
                if qoe_calculator:
                    try:
                        qoe_calculator.record_segment_download(expected_size, download_duration, current_download_quality)
                    except Exception as e:
                        logger.error(f"[QoE] 记录分片下载错误: {e}")
                
                # --- 更新缓冲区管理器 ---
                if buffer_manager:
                    try:
                        buffer_manager.add_segment_to_buffer(segment_idx_to_download, current_download_quality, avg_segment_duration_from_server)
                    except Exception as e:
                        logger.error(f"[BM] 添加分片到缓冲区错误: {e}")
                
                # --- 在加入播放列表前，在map中标记为已下载（添加质量信息） ---
                with lock:
                     downloaded_segments_info[segment_idx_to_download] = {
                         'path': os.path.abspath(local_segment_path), 
                         'mrl': None, 
                         'downloaded': True, 
                         'in_playlist': False,
                         'quality': current_download_quality  # 添加质量信息
                     }
                
                add_segment_to_vlc_playlist(local_segment_path, segment_idx_to_download)
                
                speed_kbps = (expected_size * 8) / (download_duration * 1000) if download_duration > 0 else 0
                logger.info(f"[DL_Thread] 分片 {segment_idx_to_download} 下载完成 "
                          f"(质量: {current_download_quality}, 大小: {expected_size}字节, "
                          f"时间: {download_duration:.3f}s, 速度: {speed_kbps:.1f}kbps)")
                
            elif header_str.startswith("ERROR 404"):
                logger.warning(f"[DL_Thread] 分片 {segment_idx_to_download} (总共: {total_segments_from_server}) 服务器返回404。判定为流结束。")
                all_segments_processed_by_downloader.set(); keep_streaming_and_downloading.clear(); break 
            else: 
                logger.error(f"[DL_Thread] 服务器返回错误 (分片 {segment_idx_to_download}): {header_str}")
                if "ERROR" in header_str:
                    # 如果服务器返回错误但不是404，尝试重试使用不同的质量
                    if current_download_quality != initial_quality_suffix:
                        logger.info(f"[DL_Thread] 尝试使用初始质量 {initial_quality_suffix} 重新请求")
                        current_download_quality = initial_quality_suffix
                        continue  # 重试同一分片，但使用初始质量
                all_segments_processed_by_downloader.set(); keep_streaming_and_downloading.clear(); break
        except ConnectionError as e: 
            logger.error(f"[DL_Thread] 下载分片 {segment_idx_to_download} 时连接错误: {e}. 停止下载。")
            all_segments_processed_by_downloader.set(); keep_streaming_and_downloading.clear(); break
        except Exception as e:
            logger.error(f"[DL_Thread] 下载分片 {segment_idx_to_download} 时发生意外错误: {e}", exc_info=True)
            all_segments_processed_by_downloader.set(); keep_streaming_and_downloading.clear(); break
            
    # for 循环结束后（无论是正常完成还是break）
    if segment_idx_to_download + 1 >= total_segments_from_server and keep_streaming_and_downloading.is_set():
        all_segments_processed_by_downloader.set() # 确保如果循环正常结束，也设置此标志
        logger.info(f"[DL_Thread] 所有 {total_segments_from_server} 个分片已由下载线程处理完毕。")
    
    logger.info("[DL_Thread] 下载任务结束或被信号停止。")


def start_streaming_session_control(client_sock, video_name, initial_quality_suffix):
    """主控制函数：获取元数据，启动下载线程，并保持主线程运行以处理VLC事件和用户输入。"""
    global total_segments_from_server, avg_segment_duration_from_server, \
           keep_streaming_and_downloading, download_thread_instance, \
           all_segments_processed_by_downloader, downloaded_segments_info, \
           currently_playing_original_idx, last_cleaned_original_idx, media_list,\
           player_signaled_true_end, abr_algorithm, qoe_calculator, buffer_manager, \
           current_quality_suffix, abr_enabled  

    # 确保下载目录存在
    if not os.path.exists(DOWNLOAD_DIR): os.makedirs(DOWNLOAD_DIR)

    # --- 初始化ABR和QoE组件 ---
    if abr_enabled:
        try:
            abr_algorithm = ABRAlgorithm(initial_quality_suffix)
            qoe_calculator = QoECalculator()
            buffer_manager = BufferManager()
            current_quality_suffix = initial_quality_suffix
            logger.info("ABR算法和QoE统计已启动")
        except Exception as e:
            logger.error(f"初始化ABR组件失败: {e}")
            abr_enabled = False
    
    # 初始化或重置播放器和状态变量
    initialize_player() 
    with lock: 
        if media_list_player and media_list_player.get_state() != vlc.State.NothingSpecial:
            media_list_player.stop()
        
        if media_list: # 清空并释放旧的 MediaList 中的媒体项
            media_list.lock()
            items_to_release = [media_list.item_at_index(i) for i in range(media_list.count())]
            media_list.release() # 释放旧的 media_list 对象本身
            for item in items_to_release:
                if item: 
                    item.release()
        media_list = vlc_instance.media_list_new([]) # 创建一个新的空列表
        media_list_player.set_media_list(media_list) # 将新列表设置给播放器
        logger.info("MediaList 已为新会话重新初始化并设置给播放器。")
        
        downloaded_segments_info.clear()
        currently_playing_original_idx = -1
        last_cleaned_original_idx = -1
        total_segments_from_server = 0
        avg_segment_duration_from_server = 0.0

    keep_streaming_and_downloading.set() # 设置事件，允许下载线程运行
    all_segments_processed_by_downloader.clear() # 清除下载完成标志
    player_signaled_true_end.clear() # 清除播放器结束标志

    logger.info(f"尝试获取元数据: {video_name}/{initial_quality_suffix}")
    request_message = f"METADATA {video_name}/{initial_quality_suffix}\n"
    try:
        client_sock.sendall(request_message.encode('utf-8'))
        header_data = b""
        client_sock.settimeout(SOCKET_TIMEOUT_SECONDS) # 为元数据响应设置超时
        try:
            while not header_data.endswith(b"\n"):
                byte = client_sock.recv(1)
                if not byte: 
                    raise ConnectionError("服务器在元数据响应中关闭连接")
                header_data += byte
        except socket.timeout: 
            logger.error("等待元数据响应超时。")
            return # 获取元数据失败，无法继续
        finally: 
            client_sock.settimeout(None) # 重置socket超时
        
        header_str = header_data.decode('utf-8').strip()
        logger.info(f"服务器元数据响应: {header_str}")

        if header_str.startswith("METADATA_OK "):
            parts = header_str.split(" ")
            try:
                total_segments_from_server = int(parts[1])
                avg_segment_duration_from_server = float(parts[2])
            except (IndexError, ValueError) as e:
                logger.error(f"解析元数据响应失败: '{header_str}' - {e}"); return

            if avg_segment_duration_from_server <= 0: # 防御性编程
                avg_segment_duration_from_server = 5.0 # 回退到默认
                logger.warning(f"服务器返回的平均分片时长无效或为零，使用默认值 {avg_segment_duration_from_server}秒")
            logger.info(f"元数据获取成功: 总分片数={total_segments_from_server}, 平均分片时长={avg_segment_duration_from_server:.2f}秒")

            if total_segments_from_server == 0: 
                logger.warning("服务器报告总分片数为0，没有可播放的内容。"); return

            # 启动下载线程
            download_thread_instance = threading.Thread(target=download_segments_task, 
                                                        args=(client_sock, video_name, initial_quality_suffix),
                                                        daemon=True, name="SegmentDownloader")
            download_thread_instance.start()
            
            logger.info("下载线程已启动。主线程将监控播放状态 (按 Ctrl+C 停止)...")
            
            # --- 主线程保持运行，处理用户输入和进度报告 ---
            last_progress_log_time = time.time()
            
            try:
                # 主循环保持运行，直到播放器真正结束或出错，或者用户中断
                while not player_signaled_true_end.is_set():
                    try:
                        # 进度报告
                        current_time = time.time()
                        if current_time - last_progress_log_time >= PROGRESS_LOG_INTERVAL:
                            with lock:
                                playing_idx_copy = currently_playing_original_idx
                                downloaded_count = sum(1 for info in downloaded_segments_info.values() if info.get('downloaded', False))
                                playlist_count = sum(1 for info in downloaded_segments_info.values() if info.get('in_playlist', False))
                            
                            if total_segments_from_server > 0:
                                progress_percent = (playing_idx_copy + 1) / total_segments_from_server * 100 if playing_idx_copy >= 0 else 0
                                logger.info(f"播放进度: {playing_idx_copy + 1}/{total_segments_from_server} "
                                          f"({progress_percent:.1f}%), 已下载: {downloaded_count}, 播放列表: {playlist_count}")
                                
                                # --- 显示ABR状态 ---
                                if abr_algorithm:
                                    try:
                                        # 尝试获取当前质量
                                        current_quality = None
                                        try:
                                            current_quality = abr_algorithm.get_current_quality()
                                        except:
                                            # 如果上面的方法不存在，匹配一个质量对象
                                            for q in abr_algorithm.QUALITY_LEVELS:
                                                if q.suffix == current_quality_suffix:
                                                    current_quality = q
                                                    break
                                        
                                        buffer_level = 0.0
                                        if buffer_manager:
                                            try:
                                                buffer_level = buffer_manager.get_buffer_level_seconds()
                                            except:
                                                try:
                                                    buffer_level = sum(info.segment_duration for info in buffer_manager.buffer_queue)
                                                except:
                                                    pass
                                                    
                                        avg_throughput = 0.0
                                        try:
                                            avg_throughput = abr_algorithm.network_metrics.get_average_throughput() / 1000  # kbps
                                        except:
                                            pass
                                            
                                        quality_desc = current_quality.suffix if current_quality else current_quality_suffix
                                        logger.info(f"ABR状态: 质量={quality_desc}, "
                                              f"缓冲区={buffer_level:.1f}s, 平均带宽={avg_throughput:.1f}kbps")
                                    except Exception as e:
                                        logger.error(f"显示ABR状态时出错: {e}")
                            
                            last_progress_log_time = current_time

                        # 检查下载线程是否完成
                        if all_segments_processed_by_downloader.is_set():
                            player_state = media_list_player.get_state()
                            current_media_in_player_obj = media_player.get_media()
                            
                            if (player_state in [vlc.State.Ended, vlc.State.Stopped] and 
                                current_media_in_player_obj is None):
                                logger.info("所有分片已下载且播放完毕。流式传输会话结束。")
                                player_signaled_true_end.set()
                                break

                        # 检查播放器是否信号结束
                        if player_signaled_true_end.is_set():
                            logger.info("播放器信号结束。停止流式传输。")
                            break

                        time.sleep(0.5)  # 短暂休眠减少CPU使用

                    except KeyboardInterrupt:
                        logger.info("用户中断 (Ctrl+C)。正在停止流式传输...")
                        break
            except KeyboardInterrupt: 
                logger.info("主控制循环收到键盘中断。发送停止信号...")
                keep_streaming_and_downloading.clear() 
                player_signaled_true_end.set() # 确保主循环退出
            
            logger.info("主控制循环结束或被中断。")

        else: # 元数据获取失败
            logger.error(f"从服务器获取元数据失败: {header_str}"); return

    except ConnectionError as e: 
        logger.error(f"元数据获取或会话控制中连接错误: {e}")
    except Exception as e: 
        logger.error(f"会话控制中发生意外错误: {e}", exc_info=True)
    finally: # 这个finally属于start_streaming_session_control的try
        logger.info("会话控制结束。确保下载线程停止...")
        keep_streaming_and_downloading.clear() # 确保下载线程会看到停止信号
        if download_thread_instance and download_thread_instance.is_alive():
            logger.info("等待下载线程汇合 (从会话控制)...")
            download_thread_instance.join(timeout=3.0) # 给下载线程一点时间结束
            if download_thread_instance.is_alive():
                 logger.warning("下载线程在会话控制结束时未能干净退出。")


def main():
    global vlc_instance, media_list_player, keep_streaming_and_downloading, download_thread_instance, \
           qoe_calculator

    # --- Windows DLL 加载辅助 ---
    if os.name == 'nt':
        vlc_install_dir = None
        common_paths = [
            os.path.join(os.environ.get("ProgramFiles", "C:\\Program Files"), "VideoLAN", "VLC"),
            os.path.join(os.environ.get("ProgramFiles(x86)", "C:\\Program Files (x86)"), "VideoLAN", "VLC")
        ]
        for p in common_paths:
            if os.path.isdir(p) and os.path.exists(os.path.join(p, "libvlc.dll")):
                vlc_install_dir = p; break
        if vlc_install_dir and hasattr(os, 'add_dll_directory'):
            try:
                os.add_dll_directory(vlc_install_dir)
                logger.info(f"已添加VLC DLL路径: {vlc_install_dir}")
            except Exception as e_dll: 
                logger.warning(f"添加VLC DLL路径失败: {e_dll}")
        elif not vlc_install_dir : logger.warning("未在常见位置找到VLC安装目录。请确保VLC在系统PATH中。")

    # --- 视频和质量配置 ---
    VIDEO_TO_STREAM = "bbb_sunflower"
    QUALITY_TO_STREAM = "480p-1500k" # 选择一个你已为其生成分片和M3U8的有效质量

    client_s = None # 初始化客户端套接字
    try:
        # ---- 初始化下载目录 ----
        if not os.path.exists(DOWNLOAD_DIR):
            os.makedirs(DOWNLOAD_DIR)
        else: # 清理之前运行可能留下的临时文件
            logger.info(f"清理旧的临时文件于目录: {DOWNLOAD_DIR}...")
            for f_name in os.listdir(DOWNLOAD_DIR):
                f_path = os.path.join(DOWNLOAD_DIR, f_name)
                if os.path.isfile(f_path):
                    try:
                        os.remove(f_path)
                        logger.debug(f"已删除旧文件: {f_name}")
                    except OSError as e:
                        logger.warning(f"删除旧文件失败 {f_name}: {e}")
        
        # ---- 连接服务器 ----
        client_s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        logger.info(f"正在连接服务器 {SERVER_HOST}:{SERVER_PORT}...")
        client_s.connect((SERVER_HOST, SERVER_PORT))
        logger.info("已连接到服务器。")

        # ---- 开始流式会话控制 ----
        start_streaming_session_control(client_s, VIDEO_TO_STREAM, QUALITY_TO_STREAM)

    except KeyboardInterrupt: 
        logger.info("应用程序被用户中断 (Ctrl+C)。")
    except socket.timeout: 
        logger.error(f"连接服务器 {SERVER_HOST}:{SERVER_PORT} 超时。")
    except ConnectionRefusedError: 
        logger.error(f"连接被拒绝。服务器 {SERVER_HOST}:{SERVER_PORT} 可能未运行。")
    except Exception as e: 
        logger.error(f"主函数发生错误: {e}", exc_info=True) # exc_info=True 会记录完整堆栈
    finally:
        logger.info("主函数: 正在关闭客户端...")
        keep_streaming_and_downloading.clear() # 确保所有循环都会停止

        if download_thread_instance and download_thread_instance.is_alive():
            logger.info("主函数: 等待下载线程结束...")
            download_thread_instance.join(timeout=3.0) # 给下载线程一点时间退出
        
        if media_list_player: 
            media_list_player.stop() # 停止VLC播放器
            logger.info("主函数: VLC MediaListPlayer 已停止。")
        
        if client_s: # 如果套接字对象存在
            if not getattr(client_s, '_closed', False):                
                try:
                    client_s.sendall(b"QUIT\n")
                    logger.info("已向服务器发送QUIT命令。")
                except: 
                    pass # 忽略发送QUIT时的错误
            try:
                client_s.close()
                logger.info("客户端套接字已关闭。")
            except OSError: 
                pass # 忽略关闭套接字时的错误
            finally:
                client_s = None
        
        # --- QoE统计导出 ---
        if qoe_calculator:
            try:
                qoe_calculator.finalize_session()
                
                # 确保logs目录存在
                logs_dir = "logs"
                if not os.path.exists(logs_dir):
                    os.makedirs(logs_dir)
                
                timestamp = int(time.time())
                qoe_report_path = os.path.join(logs_dir, f"qoe_report_{timestamp}.json")
                qoe_calculator.export_to_json(qoe_report_path)
                
                # 打印QoE摘要
                session = qoe_calculator.get_session_summary()
                logger.info(f"=== QoE会话摘要 ===")
                logger.info(f"播放时长: {(session.session_end_time or time.time()) - session.session_start_time:.1f}秒")
                logger.info(f"下载分片数: {session.total_segments_downloaded}")
                logger.info(f"总下载量: {session.total_bytes_downloaded / (1024*1024):.1f}MB")
                logger.info(f"平均下载速度: {session.average_download_speed_kbps:.1f}kbps")
                logger.info(f"质量切换次数: {len(session.quality_switches)}")
                logger.info(f"缓冲事件数: {len(session.buffering_events)}")
                logger.info(f"总缓冲时间: {session.total_buffering_time:.1f}秒")
                logger.info(f"启动延迟: {session.startup_delay_seconds:.1f}秒")
                logger.info(f"平均主观评分(MOS): {session.mean_opinion_score:.2f}/5.0")
                logger.info(f"质量分布: {session.quality_time_distribution}")
            except Exception as e:
                logger.error(f"导出QoE报告时出错: {e}")

        logger.info("主函数: 执行下载分片的最终清理...")
        with lock: 
            # 从 downloaded_segments_info 中获取所有记录的已下载文件路径
            paths_to_clean_final = [info['path'] for info in downloaded_segments_info.values() if isinstance(info, dict) and 'path' in info]
            downloaded_segments_info.clear() 
        
        for path in list(set(paths_to_clean_final)): # 去重后迭代
            # 确保路径是字符串，是绝对路径，在我们期望的下载目录中，并且文件存在
            if isinstance(path, str) and os.path.isabs(path) and \
               path.startswith(os.path.abspath(DOWNLOAD_DIR)) and os.path.exists(path):
                try:
                    os.remove(path)
                    logger.debug(f"最终清理: 已删除 {os.path.basename(path)}")
                except OSError as e:
                    logger.warning(f"最终清理删除文件失败 {path}: {e}")
        
        if vlc_instance: 
            vlc_instance.release() # 释放VLC实例
            logger.info("主函数: VLC实例已释放。")
        
        logger.info("客户端应用程序结束。")

if __name__ == "__main__":
    main()