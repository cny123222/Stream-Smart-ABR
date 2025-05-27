import socket
import os
import time
import logging
import vlc 
import threading
from urllib.parse import unquote, urlparse # 用于解码MRL中的路径
import AES
# --- Configuration ---
SERVER_HOST = '127.0.0.1'
SERVER_PORT = 8081 # 确保与服务器端口一致
DOWNLOAD_DIR = "download"
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
    global currently_playing_original_idx, downloaded_segments_info, last_cleaned_original_idx
    
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
                    # 清理条件：索引小于阈值，并且大于上一次清理的最后一个索引（避免重复检查已清理的）
                    if idx_to_check < cleanup_older_than_this_original_idx and idx_to_check > last_cleaned_original_idx :
                        path_to_delete = info_to_delete.get('path')
                        # 确保路径是下载目录中的临时文件且真实存在
                        if path_to_delete and path_to_delete.startswith(os.path.abspath(DOWNLOAD_DIR)) and os.path.exists(path_to_delete):
                            try:
                                os.remove(path_to_delete) # 删除文件
                                logger.info(f"清理旧分片 (原始索引 {idx_to_check}): {path_to_delete}")
                                keys_to_delete_from_map.append(idx_to_check)
                            except OSError as e: 
                                logger.error(f"删除旧分片 {path_to_delete} 失败: {e}")
                if keys_to_delete_from_map: # 如果有文件被标记为删除
                    for key in keys_to_delete_from_map:
                        if key in downloaded_segments_info: # 从跟踪字典中移除
                            del downloaded_segments_info[key]
                    last_cleaned_original_idx = max(keys_to_delete_from_map) # 更新最后清理的索引
            else: 
                logger.warning(f"无法将当前播放的MRL {new_mrl} (路径: {new_path_playing_abs}) 映射回原始分片索引。")
    else: 
        logger.warning(f"MediaPlayerMediaChanged: 无法从MRL获取路径: {new_mrl}")
        with lock: currently_playing_original_idx = -1


def on_item_end_reached_callback(event):
    """当MediaListPlayer播放完列表中的一个媒体项后触发。注意：这不一定是整个流的结束。"""
    logger.debug("MediaPlayerEndReached (on_item_end_reached_callback): 当前媒体项播放完毕。")
    # MediaListPlayer会自动尝试播放列表中的下一项（如果存在）。
    # 真正的“整个流结束”判断逻辑：
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
                # 计算已下载但尚未播放的（或正在播放的）分片数量
                # 即 segment_idx_to_download (下一个要下载的) 与 current_playing_idx_snapshot 的差距
                # 如果 current_playing_idx_snapshot 是 -1 (还没开始播)，则我们看已下载的是否达到 MIN_BUFFER_TO_START_PLAY
                
                # num_downloaded_and_not_yet_played_or_playing：
                # 指的是，在 downloaded_segments_info 中，索引 >= (current_playing_idx_snapshot if != -1 else 0) 
                # 并且 <= (segment_idx_to_download - 1) 的分片数量
                # 这个逻辑可以更精细，例如直接检查 MediaList 中未播放的数量
                
                # 简化逻辑：如果下一个要下载的分片索引，比当前播放的原始分片索引，
                # 超出的数量已经达到了 FETCH_AHEAD_TARGET，并且播放器正在活跃播放，则等待。
                # 如果播放器未激活，则只要下载数量未达到 MIN_BUFFER_TO_START_PLAY 就继续下载。
                
                num_segments_ahead_of_playing = segment_idx_to_download - \
                                                (current_playing_idx_snapshot if current_playing_idx_snapshot != -1 else -1)
                                                # 如果还没开始播，就看已下载多少个 (segment_idx_to_download 就是下一个)

            player_is_active = media_list_player.get_state() in [vlc.State.Playing, vlc.State.Buffering, vlc.State.Opening]

            if (player_is_active and num_segments_ahead_of_playing < FETCH_AHEAD_TARGET) or \
               (not player_is_active and segment_idx_to_download < MIN_BUFFER_TO_START_PLAY):
                # logger.debug(f"[DL_Thread] Buffer not full or player not active. "
                #             f"Ahead: {num_segments_ahead_of_playing}, Target: {FETCH_AHEAD_TARGET}. Continue DL {segment_idx_to_download}")
                break # 缓冲不足，或播放器未激活但初始缓冲未满，继续下载当前 segment_idx_to_download
            elif not player_is_active and segment_idx_to_download >= MIN_BUFFER_TO_START_PLAY:
                # logger.debug(f"[DL_Thread] Player not active but min buffer met. Waiting for player to start.")
                time.sleep(0.1) # 播放器未激活，但初始缓冲已满，等待播放器启动
            else: # 播放器激活，且缓冲已满
                # logger.debug(f"[DL_Thread] Buffer target met (ahead {num_segments_ahead_of_playing}). Pausing download of seg {segment_idx_to_download}.")
                time.sleep(0.1) # 缓冲充足，下载线程短暂休眠
        
        if not keep_streaming_and_downloading.is_set(): # 在缓冲等待后再次检查停止信号
             logger.info("[DL_Thread] 在缓冲等待后收到停止信号。终止下载。")
             break
        # --- 缓冲逻辑 End ---
        
        with lock: # 再次检查是否已下载过，防止重复（理论上for循环不会，但多层控制下可能）
            if segment_idx_to_download in downloaded_segments_info and \
               downloaded_segments_info[segment_idx_to_download].get('downloaded', False):
                # logger.info(f"[DL_Thread] 分片 {segment_idx_to_download} 已下载。跳过。")
                continue # 已下载，处理下一个索引
        
        # 构造文件名和请求路径
        # 为本地临时文件名加入更唯一的标识，以防旧文件干扰（尽管启动时会清理download目录）
        unique_ts_id = f"{segment_idx_to_download:03d}_{int(time.time()*1000000)}" 
        segment_filename_on_server = f"{video_name}-{quality_suffix}-{segment_idx_to_download:03d}.ts"
        request_path_on_server = f"{video_name}/{quality_suffix}/{segment_filename_on_server}"
        local_segment_filename = f"temp_{video_name}_{quality_suffix}_{unique_ts_id}.ts"
        local_segment_path = os.path.join(DOWNLOAD_DIR, local_segment_filename)

        logger.info(f"[DL_Thread] 请求分片 (原始索引 {segment_idx_to_download}): {request_path_on_server}")
        request_message = f"GET {request_path_on_server}\n"
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
                    if not byte: raise ConnectionError("服务器关闭连接 (分片头部)")
                    header_data += byte
            except socket.timeout:
                logger.error(f"[DL_Thread] 获取分片 {segment_idx_to_download} 头部超时。将在 {RETRY_DOWNLOAD_DELAY}秒后重试。")
                time.sleep(RETRY_DOWNLOAD_DELAY); continue # 重试当前分片的下载
            finally: client_socket_ref.settimeout(None)

            header_str = header_data.decode('utf-8').strip()
            if header_str.startswith("OK "):
                try: expected_size = int(header_str.split(" ", 1)[1])
                except (IndexError, ValueError): 
                    logger.error(f"[DL_Thread] 分片 {segment_idx_to_download} 的OK响应格式无效: {header_str}"); continue
                
                segment_data = receive_exact_bytes(client_socket_ref, expected_size)
                len_segment_data_encrypt = len(segment_data)
                segment_data = AES.aes_decrypt_cbc(segment_data, AES.AES_KEY) # 解密分片数据
                logger.info(f"[DL_Thread] 正在写入解密数据到本地分片文件: {local_segment_path}，明文长度: {len(segment_data)} 字节")
                with open(local_segment_path, 'wb') as f: f.write(segment_data)
                
                if len_segment_data_encrypt == expected_size:
                    # 在加入播放列表前，就在map中标记为已下载
                    with lock:
                         downloaded_segments_info[segment_idx_to_download] = {
                             'path': os.path.abspath(local_segment_path), 
                             'mrl': None, 
                             'downloaded': True, 
                             'in_playlist': False 
                         }
                    add_segment_to_vlc_playlist(local_segment_path, segment_idx_to_download)
                else:
                    logger.error(f"[DL_Thread] 分片 {local_segment_path} 大小不匹配。")
                    if os.path.exists(local_segment_path): os.remove(local_segment_path)
            elif header_str.startswith("ERROR 404"):
                logger.warning(f"[DL_Thread] 分片 {segment_idx_to_download} (总共: {total_segments_from_server}) 服务器返回404。判定为流结束。")
                all_segments_processed_by_downloader.set(); keep_streaming_and_downloading.clear(); break 
            else: 
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

    # 确保下载目录存在
    if not os.path.exists(DOWNLOAD_DIR): os.makedirs(DOWNLOAD_DIR)
    
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
                if item: item.release() # 释放每个 media 对象
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
                # 注意: os.add_dll_directory 应该在 import vlc 之前调用才最有效。
                # 由于 import vlc 在脚本顶部，这里的调用可能更多是形式上的，
                # 除非 python-vlc 的 find_lib 实现能后续利用这个信息。
                # 如果 import vlc 已经成功，说明 python-vlc 已能找到库。
                # os.add_dll_directory(vlc_install_dir) 
                logger.info(f"VLC DLL 目录提示: {vlc_install_dir}. 确保它在PATH中或python-vlc能找到。")
            except Exception as e_dll: logger.warning(f"无法添加VLC DLL目录 {vlc_install_dir}: {e_dll}")
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