import socket
import os
import time
import logging
import vlc 
import threading
from urllib.parse import unquote, urlparse # 用于解码MRL中的路径

# --- Configuration ---
SERVER_HOST = '127.0.0.1'
SERVER_PORT = 8081 # 确保与服务器端口一致
DOWNLOAD_DIR = "download"
BUFFER_SIZE = 4096
FETCH_AHEAD_TARGET = 3     # 目标：在播放指针之后，保持这么多分片已下载并加入播放列表
MIN_BUFFER_TO_START_PLAY = 1 # 至少下载多少个分片到播放列表后开始播放
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
# Events
keep_streaming_and_downloading = threading.Event() 
all_segments_processed_by_downloader = threading.Event() 
player_signaled_true_end = threading.Event() 

# {original_segment_idx: {'path': local_abs_path, 'mrl': mrl_str, 
#                         'downloaded': bool, 'added_to_playlist': bool, 
#                         'size': int, 'duration': float (optional from m3u8 if parsed)}}
downloaded_segments_info = {} 
lock = threading.Lock() 

total_segments_from_server = 0
avg_segment_duration_from_server = 0.0 

currently_playing_original_idx = -1 
last_cleaned_original_idx = -1   

def path_to_mrl(local_file_path):
    """将本地文件绝对路径转换为VLC可以使用的MRL (file:///...)"""
    abs_path = os.path.abspath(local_file_path)
    # vlc.py的 media_new() 通常可以直接处理Python的绝对路径
    # 为了明确，我们手动转为MRL格式
    if os.name == 'nt': # Windows
        return 'file:///' + abs_path.replace('\\', '/')
    else: # macOS, Linux
        return f'file://{abs_path}'

def mrl_to_local_os_path(mrl_string):
    """从MRL（通常由VLC事件返回）转换回本地操作系统路径"""
    if not mrl_string: return None
    try:
        parsed = urlparse(mrl_string)
        if parsed.scheme != 'file': return None
        path = unquote(parsed.path) 
        if os.name == 'nt' and len(path) > 1 and path[0] == '/' and path[2] == ':':
            path = path[1:] # 去除Windows MRL中驱动器号前的额外斜杠
        return os.path.normpath(path)
    except Exception as e:
        logger.error(f"解析MRL时出错 '{mrl_string}': {e}")
        return None

def initialize_player():
    """初始化VLC播放器实例和相关对象"""
    global vlc_instance, media_list_player, media_list, media_player, player_event_manager
    if vlc_instance is None: # 确保只初始化一次
        instance_args = ['--no-video-title-show', '--quiet']
        # network_caching: libVLC的网络缓存(毫秒)。对于本地文件列表，这个影响不大，
        # 但如果VLC认为它在处理流式内容，可能会有帮助。
        # 我们主要依赖自己的应用层缓冲。
        instance_args.append(f'--network-caching=1000') 
        
        vlc_instance = vlc.Instance(instance_args)
        media_list = vlc_instance.media_list_new([])
        media_list_player = vlc_instance.media_list_player_new()
        media_player = vlc_instance.media_player_new() 
        media_list_player.set_media_player(media_player) # 将MediaPlayer实例设置给MediaListPlayer
        media_list_player.set_media_list(media_list)    # 将MediaList设置给MediaListPlayer
        
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
    global currently_playing_original_idx, downloaded_segments_info, last_cleaned_original_idx
    
    new_media_instance = media_player.get_media()
    if not new_media_instance:
        logger.info("MediaPlayerMediaChanged: 新媒体项为 None (播放列表可能为空或已结束).")
        with lock: currently_playing_original_idx = -1
        return

    new_mrl = new_media_instance.get_mrl()
    new_path_playing_abs = mrl_to_local_os_path(new_mrl) # 将MRL转为本地路径
    
    found_idx = -1
    if new_path_playing_abs:
        logger.info(f"MediaPlayerMediaChanged: 现在播放 '{os.path.basename(new_path_playing_abs)}'")
        with lock:
            # 根据路径从我们的跟踪字典中找到对应的原始分片索引
            for idx, info in downloaded_segments_info.items():
                if info.get('path') and os.path.normpath(info['path']) == new_path_playing_abs:
                    found_idx = idx
                    break
            currently_playing_original_idx = found_idx # 更新当前播放的原始索引
            if found_idx != -1:
                logger.info(f"当前播放分片的原始索引更新为: {found_idx}")
                # --- 清理逻辑 ---
                # 清理比 (当前播放索引 - (预缓冲数 + 一个额外缓冲)) 更早的分片
                # 目的是在磁盘上保留一定数量的已播放分片，用于可能的快速回看（虽然本版无UI）
                # 同时控制磁盘空间占用
                cleanup_older_than_this_original_idx = found_idx - (FETCH_AHEAD_TARGET + 1) 
                keys_to_delete_from_map = []
                for idx_to_check, info_to_delete in downloaded_segments_info.items():
                    # 只清理那些索引小于阈值，并且之前没有清理过的
                    if idx_to_check < cleanup_older_than_this_original_idx and idx_to_check > last_cleaned_original_idx :
                        path_to_delete = info_to_delete.get('path')
                        # 确保路径有效且在我们的下载目录中
                        if path_to_delete and path_to_delete.startswith(os.path.abspath(DOWNLOAD_DIR)) and os.path.exists(path_to_delete):
                            try:
                                os.remove(path_to_delete)
                                logger.info(f"清理旧分片 (原始索引 {idx_to_check}): {path_to_delete}")
                                keys_to_delete_from_map.append(idx_to_check)
                            except OSError as e: 
                                logger.error(f"删除旧分片 {path_to_delete} 失败: {e}")
                if keys_to_delete_from_map: # 从跟踪字典中移除已删除的条目
                    for key in keys_to_delete_from_map:
                        if key in downloaded_segments_info:
                            del downloaded_segments_info[key]
                    last_cleaned_original_idx = max(keys_to_delete_from_map) # 更新最后一个被清理的索引
            else: 
                logger.warning(f"无法将当前播放的MRL {new_mrl} (路径: {new_path_playing_abs}) 映射回原始分片索引。")
    else: 
        logger.warning(f"MediaPlayerMediaChanged: 无法从MRL获取路径: {new_mrl}")
        with lock: currently_playing_original_idx = -1


def on_item_end_reached_callback(event):
    """当MediaListPlayer播放完列表中的一个媒体项后触发"""
    logger.debug("MediaPlayerEndReached (on_item_end_reached_callback): 当前媒体项播放完毕。")
    # MediaListPlayer 会自动尝试播放列表中的下一项。
    # 我们需要更可靠的信号来判断整个流是否结束。
    # player_signaled_true_end 事件会在下载线程确认所有分片已处理完毕，
    # 并且此回调确认播放器也到达了所有内容的末尾时被设置。
    if all_segments_processed_by_downloader.is_set(): # 下载线程已处理完所有预期分片
        player_state = media_list_player.get_state()
        current_media_in_player_obj = media_player.get_media() # 获取播放器当前关联的媒体

        # 如果当前播放器中没有媒体了 (通常意味着列表播放完毕或被清空)
        # 并且播放器状态是 Ended, Stopped, 或 NothingSpecial (空闲)
        if current_media_in_player_obj is None and \
           player_state in [vlc.State.Ended, vlc.State.Stopped, vlc.State.NothingSpecial]:
            logger.info("所有分片已由下载器处理完毕，且播放器指示所有媒体已结束。发送真正结束信号。")
            player_signaled_true_end.set() # 通知主控制循环可以结束了

def on_player_error_callback(event):
    logger.error("VLC Player 遇到错误。发送停止信号。")
    player_signaled_true_end.set() # 认为播放已结束或无法继续
    keep_streaming_and_downloading.clear() # 停止下载

def on_player_buffering_callback(event):
    buffer_percentage = event.u.new_cache # 获取缓冲百分比 (0.0 到 100.0)
    logger.info(f"VLC Player 正在缓冲: {buffer_percentage:.1f}%")
    if buffer_percentage < 100:
        logger.info("播放器正在缓冲... (表现为卡顿)")
        # 在GUI版本中，可以在此处显示缓冲指示器

def on_player_position_changed_callback(event):
    """当播放位置改变时，用于更新(控制台)总进度"""
    global total_segments_from_server, avg_segment_duration_from_server, currently_playing_original_idx
    
    # 确保元数据已加载，且播放器和媒体有效
    if not (media_player and total_segments_from_server > 0 and avg_segment_duration_from_server > 0):
        return

    # 获取当前播放分片在整个视频流中的原始索引 (由 on_media_changed_callback 更新)
    current_original_idx = -1
    with lock: current_original_idx = currently_playing_original_idx
    
    # 获取当前播放分片内部的播放进度比例 (0.0 到 1.0)
    pos_ratio_current_item = media_player.get_position() 
    
    if current_original_idx != -1 : # 确保我们知道当前在播放哪个原始分片
        # 使用当前分片的实际时长（如果能从VLC获取到）来计算当前分片内的已播放时长，会更精确
        # 否则回退到使用平均分片时长
        current_item_actual_length_s = avg_segment_duration_from_server 
        length_ms_from_player = media_player.get_length() # 当前媒体项的总时长(毫秒)
        if length_ms_from_player > 0: 
            current_item_actual_length_s = length_ms_from_player / 1000.0
        
        duration_in_current_segment_s = pos_ratio_current_item * current_item_actual_length_s
        # 已完整播放的那些分片的总时长 (基于平均时长估算)
        duration_of_played_segments = current_original_idx * avg_segment_duration_from_server
        
        current_time_overall_seconds = duration_of_played_segments + duration_in_current_segment_s
        total_time_overall_seconds = total_segments_from_server * avg_segment_duration_from_server
        
        # 为了避免日志刷屏，可以设置一个打印间隔
        if not hasattr(on_player_position_changed_callback, "last_log_time") or \
           time.time() - on_player_position_changed_callback.last_log_time > PROGRESS_LOG_INTERVAL:
            
            # 构造进度日志消息
            progress_msg = (f"进度: {time.strftime('%H:%M:%S', time.gmtime(current_time_overall_seconds))}/"
                            f"{time.strftime('%H:%M:%S', time.gmtime(total_time_overall_seconds))} "
                            f"(播放原始分片: {current_original_idx + 1}/{total_segments_from_server}) " # 显示基于1的索引
                            f"播放器状态: {media_list_player.get_state()}")
            logger.info(progress_msg)
            on_player_position_changed_callback.last_log_time = time.time()


def add_segment_to_vlc_playlist(local_segment_path, segment_idx_original):
    """将下载好的本地分片文件加入VLC的播放列表"""
    global media_list, media_list_player, vlc_instance, downloaded_segments_info
    if not os.path.exists(local_segment_path):
        logger.error(f"分片 {local_segment_path} 在加入播放列表时未找到。"); return

    abs_path = os.path.abspath(local_segment_path)
    media = vlc_instance.media_new(abs_path) # python-vlc可以直接处理本地绝对路径
    if not media: logger.error(f"为路径创建媒体对象失败: {abs_path}"); return
    
    mrl = media.get_mrl() # 获取VLC内部将使用的MRL

    with lock:
        media_list.lock() # 锁定媒体列表以进行修改
        added_ok = media_list.add_media(media) # add_media 返回0表示成功, -1表示失败
        media_list.unlock()
        
        if added_ok == 0: # 添加成功
            # 更新或添加分片信息到我们的跟踪字典
            info = downloaded_segments_info.get(segment_idx_original, 
                                               {'path': abs_path, 'downloaded': True}) # 保留已有信息
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
    # 如果播放器未处于播放相关状态 (Playing, Opening, Buffering, Paused)
    # 并且播放列表中的媒体数量达到了最小启动缓冲要求
    if current_player_state not in [vlc.State.Playing, vlc.State.Opening, vlc.State.Buffering, vlc.State.Paused] and \
       media_list.count() >= MIN_BUFFER_TO_START_PLAY:
        logger.info(f"缓冲已达到 ({media_list.count()} 个项目 >= 最低启动要求 {MIN_BUFFER_TO_START_PLAY} 个)。启动播放...")
        media_list_player.play() # MediaListPlayer 会从MediaList的当前项（通常是第一项，如果列表是新的）开始播放
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
            if not chunk: # 套接字关闭
                raise ConnectionError("Socket connection broken while receiving data.")
            data += chunk
    except socket.timeout:
        logger.error(f"Socket接收数据超时：期望 {bytes_to_receive_total} 字节, 已收到 {len(data)} 字节。")
        raise ConnectionError("Socket timeout during data reception.")
    finally:
        sock.settimeout(None) # 完成后重置超时
    return data


def download_segments_task(client_socket_ref, video_name, quality_suffix):
    """后台下载线程的目标函数。"""
    global total_segments_from_server, keep_streaming_and_downloading, \
           all_segments_processed_by_downloader, downloaded_segments_info, \
           currently_playing_original_idx

    logger.info("[DL_Thread] 下载线程已启动。")
    # 从原始索引0开始，直到所有分片下载完毕或收到停止信号
    for segment_idx_to_download in range(total_segments_from_server):
        if not keep_streaming_and_downloading.is_set():
            logger.info("[DL_Thread] 收到停止信号。正在终止下载线程。")
            break # 外部要求停止

        # --- 缓冲逻辑 Start ---
        # 这个循环用于在缓冲足够时暂停下载，直到播放器消耗了一些缓冲或下载线程被要求停止
        while keep_streaming_and_downloading.is_set(): 
            with lock:
                current_playing_idx_snapshot = currently_playing_original_idx
                # 计算已下载并标记为在播放列表中的，且索引 >= 当前播放的（或0如果未开始）的分片数量
                num_segments_in_active_pipeline = 0
                for idx, info in downloaded_segments_info.items():
                    # 考虑那些已下载，已加入播放列表，并且其索引大于等于当前播放的第一个分片
                    # (如果还未开始播放，currently_playing_idx_snapshot 为 -1, 则所有已下载的都算)
                    if info.get('downloaded') and info.get('in_playlist') and \
                       idx >= (current_playing_idx_snapshot if current_playing_idx_snapshot != -1 else 0) :
                        num_segments_in_active_pipeline +=1
            
            player_is_active = media_list_player.get_state() in [vlc.State.Playing, vlc.State.Buffering, vlc.State.Opening]

            # 如果“管道中”的分片数量小于我们的预缓冲目标，或者播放器当前不活跃（例如刚开始或已暂停，需要填充初始缓冲）
            # 则跳出这个等待循环，继续下载当前 segment_idx_to_download
            if num_segments_in_active_pipeline < FETCH_AHEAD_TARGET or \
               (not player_is_active and num_segments_in_active_pipeline < MIN_BUFFER_TO_START_PLAY) :
                break 
            else:
                # logger.debug(f"[DL_Thread] 缓冲目标已达到 (管道中 {num_segments_in_active_pipeline} 个 >= 目标 {FETCH_AHEAD_TARGET} 个). "
                #             f"暂停下载分片 {segment_idx_to_download}.")
                time.sleep(0.1) # 缓冲充足，下载线程短暂休眠，避免不必要的CPU占用
        
        if not keep_streaming_and_downloading.is_set(): # 在缓冲等待后再次检查停止信号
             logger.info("[DL_Thread] 在缓冲等待后收到停止信号。终止下载。")
             break
        # --- 缓冲逻辑 End ---
        
        with lock: # 检查是否已下载过（虽然for循环的索引是递增的，但以防万一）
            if segment_idx_to_download in downloaded_segments_info and \
               downloaded_segments_info[segment_idx_to_download].get('downloaded', False):
                # logger.info(f"[DL_Thread] 分片 {segment_idx_to_download} 已下载。跳过。")
                continue
        
        # 构造文件名和请求路径
        unique_ts_id = f"{segment_idx_to_download:03d}_{int(time.time()*1000000)}" # 确保临时文件名唯一
        segment_filename_on_server = f"{video_name}-{quality_suffix}-{segment_idx_to_download:03d}.ts"
        request_path_on_server = f"{video_name}/{quality_suffix}/{segment_filename_on_server}"
        local_segment_filename = f"temp_{video_name}_{quality_suffix}_{unique_ts_id}.ts"
        local_segment_path = os.path.join(DOWNLOAD_DIR, local_segment_filename)

        logger.info(f"[DL_Thread] 请求分片 (原始索引 {segment_idx_to_download}): {request_path_on_server}")
        request_message = f"GET {request_path_on_server}\n"
        try:
            # 确保套接字仍然有效
            if not client_socket_ref or getattr(client_socket_ref, '_closed', False) or client_socket_ref.fileno() == -1:
                logger.error("[DL_Thread] 套接字已关闭或无效。无法下载。正在停止任务。")
                keep_streaming_and_downloading.clear(); break

            client_socket_ref.sendall(request_message.encode('utf-8')) # 发送GET请求
            
            # 接收响应头 (OK <size> 或 ERROR)
            header_data = b""
            client_socket_ref.settimeout(SOCKET_TIMEOUT_SECONDS) 
            try:
                while not header_data.endswith(b"\n"):
                    byte = client_socket_ref.recv(1)
                    if not byte: raise ConnectionError("服务器关闭连接 (分片头部)")
                    header_data += byte
            except socket.timeout:
                logger.error(f"[DL_Thread] 获取分片 {segment_idx_to_download} 头部超时。将在 {RETRY_DOWNLOAD_DELAY}秒后重试。")
                time.sleep(RETRY_DOWNLOAD_DELAY); continue # 重试当前分片的下载
            finally: client_socket_ref.settimeout(None) # 清除超时

            header_str = header_data.decode('utf-8').strip()
            if header_str.startswith("OK "):
                try: 
                    expected_size = int(header_str.split(" ", 1)[1])
                except (IndexError, ValueError): 
                    logger.error(f"[DL_Thread] 分片 {segment_idx_to_download} 的OK响应格式无效: {header_str}"); continue
                
                # 接收并保存分片数据
                segment_data = receive_exact_bytes(client_socket_ref, expected_size) # 此函数内部有超时
                with open(local_segment_path, 'wb') as f: f.write(segment_data)
                
                if len(segment_data) == expected_size:
                    # 先在map中标记为已下载，再加入播放列表
                    with lock:
                         downloaded_segments_info[segment_idx_to_download] = {
                             'path': os.path.abspath(local_segment_path), 
                             'mrl': None, # MRL会在add_segment_to_vlc_playlist中生成并更新
                             'downloaded': True, 
                             'in_playlist': False # 初始为False，加入列表后更新
                         }
                    add_segment_to_vlc_playlist(local_segment_path, segment_idx_to_download)
                else: # 文件大小不匹配
                    logger.error(f"[DL_Thread] 分片 {local_segment_path} 大小不匹配。")
                    if os.path.exists(local_segment_path): os.remove(local_segment_path) # 清理部分下载的文件
            
            elif header_str.startswith("ERROR 404"):
                logger.warning(f"[DL_Thread] 分片 {segment_idx_to_download} (总共: {total_segments_from_server}) 服务器返回404。判定为流结束。")
                all_segments_processed_by_downloader.set() # 标记所有预期的分片都已“处理”完毕
                keep_streaming_and_downloading.clear(); break # 停止下载循环
            else: # 其他服务器错误
                logger.error(f"[DL_Thread] 服务器返回错误 (分片 {segment_idx_to_download}): {header_str}")
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


def start_streaming_session_control(client_sock, video_name, quality_suffix):
    """主控制函数：获取元数据，启动下载线程，并保持主线程运行以处理VLC事件和用户输入。"""
    global total_segments_from_server, avg_segment_duration_from_server, \
           keep_streaming_and_downloading, download_thread_instance, \
           all_segments_processed_by_downloader, downloaded_segments_info, \
           currently_playing_original_idx, last_cleaned_original_idx, media_list,\
           player_signaled_true_end

    if not os.path.exists(DOWNLOAD_DIR): os.makedirs(DOWNLOAD_DIR)
    initialize_player() # 初始化VLC播放器实例

    # --- 清理上一次会话的状态 ---
    with lock: 
        if media_list_player and media_list_player.get_state() != vlc.State.NothingSpecial:
            media_list_player.stop() # 停止当前可能在播放的内容
        
        if media_list: # 清空并释放旧的 MediaList 中的媒体项
            media_list.lock()
            # 从后往前移除并释放，避免索引变化导致问题
            for i in range(media_list.count() - 1, -1, -1):
                media_item_to_release = media_list.item_at_index(i)
                media_list.remove_index(i) 
                if media_item_to_release:
                    media_item_to_release.release() # 释放每个媒体对象
            media_list.unlock()
            logger.info("已清空并释放旧播放列表中的媒体项。")
        # 注意：media_list 对象本身在 initialize_player 中如果已存在则不会重新创建，
        # 所以我们只是清空其内容。如果想完全重新开始，可以在 initialize_player 中加入释放和重建逻辑。
        # 为了简单，这里仅清空内容。如果需要完全重置播放器，可以考虑在 initialize_player 中释放并重建 media_list_player 和 media_player。

        downloaded_segments_info.clear()
        currently_playing_original_idx = -1
        last_cleaned_original_idx = -1
        total_segments_from_server = 0
        avg_segment_duration_from_server = 0.0
    # --- 状态重置完毕 ---

    keep_streaming_and_downloading.set() # 设置事件，允许下载线程运行
    all_segments_processed_by_downloader.clear() # 清除下载完成标志
    player_signaled_true_end.clear() # 清除播放器结束标志

    logger.info(f"尝试获取元数据: {video_name}/{quality_suffix}")
    request_message = f"METADATA {video_name}/{quality_suffix}\n"
    try:
        client_sock.sendall(request_message.encode('utf-8'))
        header_data = b""
        client_sock.settimeout(SOCKET_TIMEOUT_SECONDS) # 为元数据响应设置超时
        try:
            while not header_data.endswith(b"\n"):
                byte = client_sock.recv(1)
                if not byte: raise ConnectionError("服务器在元数据头部读取时关闭连接。")
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
            total_segments_from_server = int(parts[1])
            avg_segment_duration_from_server = float(parts[2])
            if avg_segment_duration_from_server <= 0: # 防御性编程
                avg_segment_duration_from_server = 5.0 
                logger.warning(f"服务器返回的平均分片时长无效，使用默认值 {avg_segment_duration_from_server}秒")
            logger.info(f"元数据获取成功: 总分片数={total_segments_from_server}, 平均分片时长={avg_segment_duration_from_server:.2f}秒")

            if total_segments_from_server == 0: 
                logger.warning("服务器报告总分片数为0，没有可播放的内容。"); return

            # 启动下载线程
            download_thread_instance = threading.Thread(target=download_segments_task, 
                                                        args=(client_sock, video_name, quality_suffix),
                                                        daemon=True, name="SegmentDownloader")
            download_thread_instance.start()
            
            logger.info("下载线程已启动。主线程将监控播放状态 (按 Ctrl+C 停止)...")
            try:
                # 主循环保持运行，直到播放器真正结束或出错，或者用户中断
                while not player_signaled_true_end.is_set():
                    # 如果下载线程意外停止（例如网络错误），但播放器还在尝试播放剩余缓冲
                    if not keep_streaming_and_downloading.is_set() and \
                       media_list_player.get_state() not in [vlc.State.Playing, vlc.State.Buffering, vlc.State.Opening, vlc.State.Paused]:
                        logger.info("下载已停止且播放器不活跃。可能即将结束。")
                        # 此时可以再检查一下 all_segments_processed_by_downloader 和 media_list.count()
                        if all_segments_processed_by_downloader.is_set() and media_list.count() == 0:
                            logger.info("确认流结束。")
                            player_signaled_true_end.set() # 主动标记结束
                        # break # 可以考虑跳出，或者让 on_item_end_reached_callback 来最终设置 player_signaled_true_end
                    time.sleep(0.5) # 主线程的轮询间隔，处理事件等
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
    global vlc_instance, media_list_player, keep_streaming_and_downloading, download_thread_instance

    # --- Windows DLL 加载辅助 ---
    # 最佳实践仍然是确保VLC的安装目录在系统的PATH环境变量中。
    # 这里的代码尝试在 `import vlc` 之前添加VLC的DLL目录到搜索路径，这才是正确的时机。
    # 但由于 `import vlc` 在脚本顶部，这个辅助逻辑如果需要，应该在更早的地方执行，
    # 或者在脚本启动前通过外部方式设置环境变量。
    # 既然你的Windows上已能正常显示，说明python-vlc已能找到库，这里保持简单。
    if os.name == 'nt':
        # 你可以根据需要保留或调整之前版本中的DLL查找和os.add_dll_directory逻辑
        # 如果没有它也能工作，说明PATH或注册表已正确配置
        logger.info("Windows系统，请确保VLC安装目录在PATH中，或python-vlc能通过注册表找到它。")

    # --- 视频和质量配置 ---
    # VIDEO_TO_STREAM 应该是服务器上由原始视频文件名生成的文件夹名
    VIDEO_TO_STREAM = "bbb_sunflower" 
    QUALITY_TO_STREAM = "480p-1500k" # 选择一个你已分片的有效质量

    client_s = None # 初始化客户端套接字
    try:
        # ---- 初始化下载目录 ----
        if not os.path.exists(DOWNLOAD_DIR):
            os.makedirs(DOWNLOAD_DIR)
        else: # 清理之前运行可能留下的临时文件
            logger.info(f"清理旧的临时文件于目录: {DOWNLOAD_DIR}...")
            for f_name in os.listdir(DOWNLOAD_DIR):
                if f_name.startswith("temp_") and f_name.endswith(".ts"):
                    try:
                        os.remove(os.path.join(DOWNLOAD_DIR, f_name))
                    except OSError as e_rem:
                        logger.warning(f"无法移除旧文件 {f_name}: {e_rem}")
        
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
        logger.error(f"主函数发生错误: {e}", exc_info=True)
    finally:
        logger.info("主函数: 正在优雅地关闭客户端...")
        keep_streaming_and_downloading.clear() # 确保所有循环都会停止

        if download_thread_instance and download_thread_instance.is_alive():
            logger.info("主函数: 等待下载线程结束...")
            download_thread_instance.join(timeout=3.0) # 给下载线程一点时间退出
        
        if media_list_player: 
            media_list_player.stop() # 停止VLC播放器
            logger.info("主函数: VLC MediaListPlayer 已停止。")
        
        if client_s: # 如果套接字对象存在
            if not getattr(client_s, '_closed', False): # 检查套接字是否已逻辑关闭
                logger.info("主函数: 向服务器发送QUIT命令并关闭连接。")
                try: 
                    client_s.sendall(b"QUIT\n")
                except OSError: 
                    logger.warning("主函数: 发送QUIT失败；套接字可能已关闭。")
            try:
                client_s.shutdown(socket.SHUT_RDWR) # 尝试更优雅地关闭双向连接
            except OSError: pass # 忽略错误，可能已经关闭
            finally:
                client_s.close() # 最终关闭套接字
                logger.info("主函数: 客户端套接字已关闭。")
        
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
                    logger.info(f"主函数: 最终清理文件 {path}")
                except OSError as e: 
                    logger.error(f"主函数: 最终清理文件 {path} 时发生错误: {e}")
        
        if vlc_instance: 
            vlc_instance.release() # 释放VLC实例
            logger.info("主函数: VLC实例已释放。")
        
        logger.info("客户端应用程序结束。")

if __name__ == "__main__":
    main()