import sys
import socket
import os
import time
import logging
import vlc 
import threading
from urllib.parse import unquote, urlparse

from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QPushButton, 
                             QSlider, QLabel, QFrame, QHBoxLayout, QSizePolicy, 
                             QStyle, QLineEdit, QComboBox, QMessageBox)
from PyQt5.QtCore import QTimer, pyqtSignal, QObject, Qt, pyqtSlot


# --- Configuration ---
SERVER_HOST = '127.0.0.1'
SERVER_PORT = 8081 
DOWNLOAD_DIR = "download" 
BUFFER_SIZE = 4096
FETCH_AHEAD_TARGET = 3     
MIN_BUFFER_TO_START_PLAY = 1 
SOCKET_TIMEOUT_SECONDS = 10  
RETRY_DOWNLOAD_DELAY = 3     
PROGRESS_UPDATE_INTERVAL = 1.0  

# --- Logger Setup ---
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(threadName)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler()])
logger = logging.getLogger('StreamingClientPyQt_VLC')

# --- 辅助函数 ---
def path_to_mrl(local_file_path):
    abs_path = os.path.abspath(local_file_path)
    if os.name == 'nt': return 'file:///' + abs_path.replace('\\', '/')
    else: return f'file://{abs_path}'

def mrl_to_local_os_path(mrl_string):
    if not mrl_string: return None
    try:
        parsed = urlparse(mrl_string)
        if parsed.scheme != 'file': return None
        path = unquote(parsed.path) 
        if os.name == 'nt' and len(path) > 1 and path[0] == '/' and path[2] == ':':
            path = path[1:]
        return os.path.normpath(path)
    except Exception as e: logger.error(f"解析MRL时出错 '{mrl_string}': {e}"); return None

def receive_exact_bytes(sock, num_bytes):
    data = b''
    bytes_to_receive_total = num_bytes
    try:
        while len(data) < bytes_to_receive_total:
            remaining_bytes = bytes_to_receive_total - len(data)
            sock.settimeout(SOCKET_TIMEOUT_SECONDS) 
            chunk = sock.recv(min(BUFFER_SIZE, remaining_bytes))
            if not chunk: raise ConnectionError("Socket broken while receiving data.")
            data += chunk
    except socket.timeout:
        logger.error(f"Socket接收数据超时：期望 {bytes_to_receive_total} 字节, 已收到 {len(data)} 字节。")
        raise ConnectionError("Socket timeout during data reception.")
    finally:
        sock.settimeout(None) 
    return data

# --- PyQt5 GUI 主应用类 ---
class VideoPlayerWindow(QWidget):
    status_update_signal = pyqtSignal(str)
    metadata_ready_signal = pyqtSignal(int, float)
    segment_downloaded_signal = pyqtSignal(str, int)
    playback_control_signal = pyqtSignal(str) 
    overall_progress_signal = pyqtSignal(int, int) 
    playback_truly_ended_signal = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setWindowTitle("流媒体播放器 (PyQt5 + VLC)")
        self.setGeometry(100, 100, 800, 650) 
        if not os.path.exists(DOWNLOAD_DIR): # 在构造函数中创建下载目录
            os.makedirs(DOWNLOAD_DIR)

        self.client_socket = None
        self.download_thread = None
        
        self.vlc_instance = None
        self.media_list_player = None
        self.media_list = None
        self.media_player = None
        
        self.keep_streaming_event = threading.Event()
        self.all_segments_processed_event = threading.Event() 
        self.player_stopped_or_error_event = threading.Event()

        self.downloaded_segments_info = {} 
        self.lock = threading.Lock() 
        
        self.currently_playing_original_idx = -1 
        self.last_cleaned_original_idx = -1    
        self.total_segments_from_server = 0    
        self.avg_segment_duration_from_server = 0.0 

        self._is_slider_pressed = False 
        self._last_progress_log_time = 0 
        self._last_progress_log_msg = ""   
        
        self.init_ui() 
        self.connect_signals_slots() 
        self.initialize_vlc_player() 
        self.show() 

    def init_ui(self):
        main_layout = QVBoxLayout()
        input_layout = QHBoxLayout()
        self.video_name_input = QLineEdit("bbb_sunflower") 
        self.quality_combo = QComboBox()
        self.quality_combo.addItems(["480p-1500k", "720p-4000k", "1080p-8000k"]) 
        self.quality_combo.setCurrentText("480p-1500k") 
        self.load_button = QPushButton("加载视频")
        
        input_layout.addWidget(QLabel("视频名:"))
        input_layout.addWidget(self.video_name_input)
        input_layout.addWidget(QLabel("质量:"))
        input_layout.addWidget(self.quality_combo)
        input_layout.addWidget(self.load_button)
        main_layout.addLayout(input_layout)

        self.video_frame = QFrame() 
        self.video_frame.setStyleSheet("background-color: black;")
        self.video_frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        main_layout.addWidget(self.video_frame, 1) 

        control_layout = QHBoxLayout()
        self.play_pause_button = QPushButton() 
        self.play_pause_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.play_pause_button.setEnabled(False) 
        self.stop_button = QPushButton() 
        self.stop_button.setIcon(self.style().standardIcon(QStyle.SP_MediaStop))
        self.stop_button.setEnabled(False) 
        
        control_layout.addWidget(self.play_pause_button)
        control_layout.addWidget(self.stop_button)
        main_layout.addLayout(control_layout)

        progress_layout = QHBoxLayout()
        self.progress_slider = QSlider(Qt.Horizontal) 
        self.progress_slider.setRange(0,0)
        self.progress_slider.setEnabled(False) 
        self.time_label = QLabel("00:00:00 / 00:00:00") 
        
        progress_layout.addWidget(self.progress_slider)
        progress_layout.addWidget(self.time_label)
        main_layout.addLayout(progress_layout)

        self.status_label = QLabel("请点击“加载视频”开始") 
        main_layout.addWidget(self.status_label)
        
        self.setLayout(main_layout)

    def connect_signals_slots(self):
        self.load_button.clicked.connect(self.handle_load_video_button_clicked)
        self.play_pause_button.clicked.connect(self.toggle_play_pause)
        self.stop_button.clicked.connect(self.handle_stop_button_clicked)
        
        self.progress_slider.sliderMoved.connect(self.handle_slider_moved_passthrough) # 改为只传递值
        self.progress_slider.sliderPressed.connect(self.handle_slider_pressed)
        self.progress_slider.sliderReleased.connect(self.handle_slider_released)

        self.status_update_signal.connect(self.status_label.setText)
        self.metadata_ready_signal.connect(self.on_gui_metadata_ready)
        self.segment_downloaded_signal.connect(self.on_gui_segment_downloaded)
        self.playback_control_signal.connect(self.on_gui_playback_control)
        self.overall_progress_signal.connect(self.on_gui_update_overall_progress)
        self.playback_truly_ended_signal.connect(self.on_gui_playback_truly_ended)

    def initialize_vlc_player(self): 
        if self.vlc_instance: return 
        instance_args = ['--no-video-title-show', '--quiet', f'--network-caching=1000']
        self.vlc_instance = vlc.Instance(instance_args)
        self.media_list = self.vlc_instance.media_list_new([])
        self.media_list_player = self.vlc_instance.media_list_player_new()
        self.media_player = self.vlc_instance.media_player_new()
        self.media_list_player.set_media_player(self.media_player)
        self.media_list_player.set_media_list(self.media_list)
        
        if os.name == 'nt': self.media_player.set_hwnd(int(self.video_frame.winId()))
        else: self.media_player.set_nsobject(int(self.video_frame.winId()))
            
        event_manager = self.media_player.event_manager()
        event_manager.event_attach(vlc.EventType.MediaPlayerMediaChanged, self._vlc_on_media_changed)
        event_manager.event_attach(vlc.EventType.MediaPlayerEndReached, self._vlc_on_item_end_reached)
        event_manager.event_attach(vlc.EventType.MediaPlayerEncounteredError, self._vlc_on_player_error)
        event_manager.event_attach(vlc.EventType.MediaPlayerPositionChanged, self._vlc_on_player_position_changed)
        event_manager.event_attach(vlc.EventType.MediaPlayerBuffering, self._vlc_on_player_buffering)
        logger.info("VLC Player initialized and linked to GUI frame.")

    # --- VLC 事件回调 (实例方法) ---
    def _vlc_on_media_changed(self, event):
        new_media_instance = self.media_player.get_media()
        if not new_media_instance:
            self.status_update_signal.emit("播放器：媒体项为空")
            with self.lock: self.currently_playing_original_idx = -1
            return

        new_mrl = new_media_instance.get_mrl()
        new_path_playing_abs = mrl_to_local_os_path(new_mrl)
        
        found_idx = -1
        if new_path_playing_abs:
            self.status_update_signal.emit(f"正在播放: {os.path.basename(new_path_playing_abs)}")
            with self.lock:
                for idx, info in self.downloaded_segments_info.items():
                    if info.get('path') and os.path.normpath(info['path']) == new_path_playing_abs:
                        found_idx = idx; break
                self.currently_playing_original_idx = found_idx
                if found_idx != -1:
                    logger.info(f"VLC CB: 当前播放原始索引: {found_idx}")
                    cleanup_older_than = found_idx - (FETCH_AHEAD_TARGET + 1)
                    keys_to_delete = []
                    for idx_chk, info_del in list(self.downloaded_segments_info.items()): # 迭代副本以允许删除
                        if idx_chk < cleanup_older_than and idx_chk > self.last_cleaned_original_idx:
                            path_del = info_del.get('path')
                            if path_del and path_del.startswith(os.path.abspath(DOWNLOAD_DIR)) and os.path.exists(path_del):
                                try: 
                                    os.remove(path_del)
                                    logger.info(f"VLC CB: 清理旧分片 {idx_chk} ({os.path.basename(path_del)})")
                                    keys_to_delete.append(idx_chk)
                                except OSError as e: logger.error(f"VLC CB: 删除 {path_del} 失败: {e}")
                    if keys_to_delete: 
                        for k_del in keys_to_delete:
                            if k_del in self.downloaded_segments_info:
                                del self.downloaded_segments_info[k_del]
                        self.last_cleaned_original_idx = max(keys_to_delete)
                else: logger.warning(f"VLC CB: 无法映射MRL {new_mrl} 到原始索引")
        else: 
            logger.warning(f"VLC CB: 无法从MRL获取路径: {new_mrl}")
            with self.lock: self.currently_playing_original_idx = -1

    def _vlc_on_item_end_reached(self, event):
        logger.debug("VLC CB: 当前媒体项播放完毕")
        if self.all_segments_processed_event.is_set():
            player_state = self.media_list_player.get_state()
            current_media = self.media_player.get_media()
            # 检查是否是列表中的最后一个媒体项播放完毕
            # (MediaListPlayer在播放完最后一项后，get_media()可能仍然返回最后一项，但状态会是Ended)
            # 并且下载线程已确认所有分片都已处理
            is_last_item_in_list = False
            if self.media_list and self.media_list.count() > 0 and current_media:
                 self.media_list.lock()
                 last_item_in_list_mrl = self.media_list.item_at_index(self.media_list.count() - 1).get_mrl()
                 self.media_list.unlock()
                 if current_media.get_mrl() == last_item_in_list_mrl and player_state == vlc.State.Ended:
                     is_last_item_in_list = True
            
            if is_last_item_in_list or \
               (current_media is None and player_state in [vlc.State.Ended, vlc.State.Stopped, vlc.State.NothingSpecial]):
                logger.info("VLC CB: 所有分片已处理且播放器已结束所有媒体")
                self.player_stopped_or_error_event.set()
                self.playback_truly_ended_signal.emit()

    def _vlc_on_player_error(self, event):
        logger.error("VLC CB: 播放器遇到错误")
        self.player_stopped_or_error_event.set()
        self.keep_streaming_event.clear()

    def _vlc_on_player_buffering(self, event):
        buffer_val = event.u.new_cache
        self.status_update_signal.emit(f"播放器缓冲中: {buffer_val:.1f}%")
        if buffer_val < 100: logger.info("播放器正在缓冲... (卡顿)")

    def _vlc_on_player_position_changed(self, event):
        with self.lock: # 使用锁保护共享变量的读取
            total_segments = self.total_segments_from_server
            avg_duration = self.avg_segment_duration_from_server
            current_idx = self.currently_playing_original_idx
        
        if not (self.media_player and total_segments > 0 and avg_duration > 0): return

        pos_ratio = self.media_player.get_position()
        item_len_ms = self.media_player.get_length()
        
        if current_idx != -1 and item_len_ms > 0:
            current_item_played_ms = int(pos_ratio * item_len_ms)
            overall_played_ms = int(current_idx * avg_duration * 1000 + current_item_played_ms)
            overall_total_ms = int(total_segments * avg_duration * 1000)
            self.overall_progress_signal.emit(overall_played_ms, overall_total_ms)

    # --- PyQt Slots (GUI Thread Actions) ---
    @pyqtSlot()
    def handle_load_video_button_clicked(self):
        logger.info("UI: '加载视频' 按钮点击.")
        self.stop_current_session_async(clear_ui_fields=False) 
        QTimer.singleShot(200, self._actually_start_new_session) 
        
    def _actually_start_new_session(self):
        if not self.establish_socket_connection():
            self.status_update_signal.emit("连接服务器失败.")
            return

        video_name = self.video_name_input.text().strip()
        quality_suffix = self.quality_combo.currentText()
        if not video_name or not quality_suffix:
            self.status_update_signal.emit("请输入视频名并选择质量.")
            return

        self.status_update_signal.emit(f"正在加载 {video_name}/{quality_suffix}...")
        
        with self.lock: 
            self.downloaded_segments_info.clear()
            self.currently_playing_original_idx = -1
            self.last_cleaned_original_idx = -1
            self.total_segments_from_server = 0 
            self.avg_segment_duration_from_server = 0.0
            if self.media_list: 
                self.media_list_player.stop() # 确保先停止播放器再操作列表
                self.media_list.lock()
                items_to_release = [self.media_list.item_at_index(i) for i in range(self.media_list.count())]
                for i in range(len(items_to_release) - 1, -1, -1): self.media_list.remove_index(i)
                for item in items_to_release: 
                    if item: item.release()
                self.media_list.unlock()
                logger.info("GUI: MediaList已为新会话清空。")
        
        self.keep_streaming_event.set()
        self.all_segments_processed_event.clear()
        self.player_stopped_or_error_event.clear()

        self.download_thread = threading.Thread(
            target=self._download_and_manage_segments_task,
            args=(video_name, quality_suffix),
            daemon=True, name="SegmentDownloaderGUI"
        )
        self.download_thread.start()
        self.play_pause_button.setEnabled(True)
        self.stop_button.setEnabled(True)


    @pyqtSlot(int, float)
    def on_gui_metadata_ready(self, total_segments, avg_duration):
        with self.lock:
            self.total_segments_from_server = total_segments
            self.avg_segment_duration_from_server = avg_duration
        
        if total_segments > 0 and avg_duration > 0:
            total_duration_ms = int(total_segments * avg_duration * 1000)
            self.progress_slider.setRange(0, total_duration_ms)
            self.progress_slider.setEnabled(True) 
            self.time_label.setText(f"00:00:00 / {time.strftime('%H:%M:%S', time.gmtime(total_duration_ms / 1000))}")
            self.status_update_signal.emit(f"元数据: {total_segments} 分片, 总长约 {self.time_label.text().split(' / ')[1]}")
        else:
            self.status_update_signal.emit("元数据无效或无分片.")
            self.stop_current_session_async(clear_ui_fields=True)


    @pyqtSlot(str, int)
    def on_gui_segment_downloaded(self, local_path, original_idx):
        abs_path = os.path.abspath(local_path)
        media = self.vlc_instance.media_new(abs_path)
        if not media: logger.error(f"GUI: 创建媒体对象失败 {abs_path}"); return
        
        mrl = media.get_mrl()
        with self.lock:
            self.media_list.lock()
            added_ok = self.media_list.add_media(media)
            self.media_list.unlock()
            
            if added_ok == 0:
                info = self.downloaded_segments_info.get(original_idx, {'path': abs_path, 'downloaded': True})
                info['mrl'] = mrl; info['in_playlist'] = True
                self.downloaded_segments_info[original_idx] = info
                logger.info(f"GUI: 添加分片 orig_idx:{original_idx} 到VLC。列表项数: {self.media_list.count()}")
            else:
                logger.error(f"GUI: 添加媒体 {abs_path} 到MediaList失败. Code: {added_ok}")
        media.release()

        current_player_state = self.media_list_player.get_state()
        if current_player_state not in [vlc.State.Playing, vlc.State.Opening, vlc.State.Buffering, vlc.State.Paused] and \
           self.media_list.count() >= MIN_BUFFER_TO_START_PLAY:
            self.playback_control_signal.emit("play")

    @pyqtSlot(str)
    def on_gui_playback_control(self, command):
        if command == "play":
            if self.media_list_player and self.media_list_player.get_state() != vlc.State.Playing:
                logger.info("GUI: 触发播放")
                self.media_list_player.play()
                self.play_pause_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))
        # 可以扩展 "pause"
        
    @pyqtSlot(int, int)
    def on_gui_update_overall_progress(self, current_ms, total_ms):
        if not self._is_slider_pressed: 
            self.progress_slider.setValue(current_ms)
        self.time_label.setText(f"{time.strftime('%H:%M:%S', time.gmtime(current_ms / 1000))} / "
                                f"{time.strftime('%H:%M:%S', time.gmtime(total_ms / 1000))}")
    
    @pyqtSlot()
    def on_gui_playback_truly_ended(self):
        self.status_update_signal.emit("所有分片播放完毕。")
        self.play_pause_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.play_pause_button.setEnabled(False)
        self.stop_button.setEnabled(False)
        self.progress_slider.setEnabled(False)
        if self.progress_slider.maximum() > 0 : 
            self.progress_slider.setValue(self.progress_slider.maximum())
        else:
            self.progress_slider.setValue(0)

    def toggle_play_pause(self):
        if not self.media_list_player: return
        state = self.media_list_player.get_state()
        if state == vlc.State.Playing:
            self.media_list_player.pause()
            # 图标和状态文本由 PlayerStateChanged 事件处理
        else: 
            if self.media_list.count() > 0:
                self.media_list_player.play()
            else:
                self.status_update_signal.emit("播放列表为空，请先加载。")
    
    @pyqtSlot()
    def handle_stop_button_clicked(self):
        self.stop_current_session_async(clear_ui_fields=True)

    def stop_current_session_async(self, clear_ui_fields=True):
        logger.info("UI: 请求停止当前流媒体会话...")
        self.keep_streaming_event.clear() 
        self.player_stopped_or_error_event.set() # 通知主循环也应该结束等待

        # 不需要立即join下载线程，让它自行结束或在closeEvent中处理。
        # UI层面的stop主要是停止新的下载和播放。
        if self.media_list_player:
            self.media_list_player.stop()
        
        if clear_ui_fields:
            self.play_pause_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
            self.play_pause_button.setEnabled(False)
            self.stop_button.setEnabled(False)
            self.progress_slider.setEnabled(False)
            self.progress_slider.setValue(0)
            self.progress_slider.setRange(0,0) 
            self.time_label.setText("00:00:00 / 00:00:00")
            self.status_update_signal.emit("已停止。请加载视频。")


    def handle_slider_pressed(self):
        if self.media_list_player and self.media_list_player.is_playing():
            self._is_slider_pressed = True 
            self.media_list_player.pause() 
            logger.debug("UI: 进度条按下，播放暂停。")

    def handle_slider_released(self):
        if self.media_list_player and self._is_slider_pressed:
            self.perform_seek(self.progress_slider.value()) 
            # perform_seek 内部会处理播放的恢复
        self._is_slider_pressed = False
        logger.debug("UI: 进度条释放。")

    def handle_slider_moved_passthrough(self, position_ms):
        """当用户拖动滑块时，只更新时间标签，实际跳转在释放时处理"""
        if not self._is_slider_pressed: return
        with self.lock:
            total_s = self.total_segments_from_server
            avg_dur = self.avg_segment_duration_from_server
        if total_s > 0 and avg_dur > 0:
            total_stream_duration_ms = int(total_s * avg_dur * 1000)
            self.time_label.setText(f"{time.strftime('%H:%M:%S', time.gmtime(position_ms / 1000))} / "
                                    f"{time.strftime('%H:%M:%S', time.gmtime(total_stream_duration_ms / 1000))}")


    def perform_seek(self, target_overall_ms):
        """根据整个流的毫秒时间点执行跳转"""
        with self.lock: # 获取当前元数据快照
            total_s = self.total_segments_from_server
            avg_dur = self.avg_segment_duration_from_server
        
        if not self.media_player or not (total_s > 0 and avg_dur > 0):
            logger.warning("Seek: 元数据不完整，无法执行跳转。")
            if self.media_list_player and self.media_list_player.can_pause() and not self._is_slider_pressed : self.media_list_player.play()
            return

        target_overall_seconds = target_overall_ms / 1000.0
        target_segment_original_idx = int(target_overall_seconds / avg_dur)
        
        if target_segment_original_idx >= total_s: target_segment_original_idx = total_s - 1
        if target_segment_original_idx < 0: target_segment_original_idx = 0
        
        position_within_segment_ms = int((target_overall_seconds - (target_segment_original_idx * avg_dur)) * 1000)
        if position_within_segment_ms < 0: position_within_segment_ms = 0

        logger.info(f"UI: 用户请求跳转到总时间 {target_overall_seconds:.2f}s (目标原始分片索引 {target_segment_original_idx}, 段内偏移 {position_within_segment_ms}ms)")

        with self.lock:
            target_segment_info = self.downloaded_segments_info.get(target_segment_original_idx)
            target_mrl = target_segment_info.get('mrl') if target_segment_info and target_segment_info.get('in_playlist') else None
            
            if target_mrl:
                idx_in_playlist = -1
                self.media_list.lock()
                for i in range(self.media_list.count()):
                    item = self.media_list.item_at_index(i)
                    if item and item.get_mrl() == target_mrl:
                        idx_in_playlist = i; break
                self.media_list.unlock()

                if idx_in_playlist != -1:
                    logger.info(f"UI: 跳转到播放列表索引 {idx_in_playlist} (对应原始索引 {target_segment_original_idx})")
                    # 在播放之前先设置好要seek到的媒体项的目标时间
                    # (VLC的MediaListPlayer行为：play_item_at_index后，新项会立即开始播放)
                    # 我们需要让它播放后，再立即设置时间。
                    self.media_list_player.play_item_at_index(idx_in_playlist)
                    
                    # 延迟设置时间，给VLC足够的时间切换媒体并准备好接受setPosition/setTime
                    def delayed_set_time_for_seek():
                        player_state = self.media_list_player.get_state()
                        if player_state in [vlc.State.Playing, vlc.State.Paused, vlc.State.Buffering]:
                            logger.info(f"Seek: 尝试为新媒体项设置时间 {position_within_segment_ms}ms")
                            self.media_player.set_time(position_within_segment_ms)
                            # 如果拖动时暂停了，并且现在滑块没有被按下，则恢复播放
                            if not self._is_slider_pressed and player_state == vlc.State.Paused and self.media_list_player.can_pause():
                                self.media_list_player.play()
                        else: # 如果状态不对，稍后重试一次 (非常简单的重试)
                            logger.warning(f"Seek: 播放器状态为 {player_state}, 稍后重试设置时间")
                            QTimer.singleShot(150, delayed_set_time_for_seek)

                    QTimer.singleShot(100, delayed_set_time_for_seek) # 初始延迟
                else: 
                    logger.warning(f"UI: 无法在播放列表中找到 MRL {target_mrl} 进行跳转。")
                    if not self._is_slider_pressed and self.media_list_player.can_pause(): self.media_list_player.play()
            else: 
                logger.warning(f"UI: 无法找到目标分片 {target_segment_original_idx} 的信息进行跳转（可能未下载或未加入列表）。")
                if not self._is_slider_pressed and self.media_list_player.can_pause(): self.media_list_player.play()


    def _download_and_manage_segments_task(self, video_name, quality_suffix):
        # ... (与上一版本基本相同的下载线程逻辑，确保使用 self.total_segments_from_server 等实例变量)
        # ... (并通过 self.status_update_signal, self.metadata_ready_signal, self.segment_downloaded_signal 与GUI通信)
        logger.info("[DL_Thread_GUI] 下载与播放列表管理线程启动。")
        
        # 1. 获取元数据
        if not self.client_socket or getattr(self.client_socket, '_closed', True) or self.client_socket.fileno() == -1:
            self.status_update_signal.emit("错误：套接字无效，无法获取元数据") 
            self.keep_streaming_event.clear(); return

        request_msg = f"METADATA {video_name}/{quality_suffix}\n"
        try:
            self.client_socket.sendall(request_msg.encode('utf-8'))
            header_data = b""
            self.client_socket.settimeout(SOCKET_TIMEOUT_SECONDS)
            try:
                while not header_data.endswith(b"\n"):
                    byte = self.client_socket.recv(1)
                    if not byte: raise ConnectionError("元数据头：服务器关闭连接")
                    header_data += byte
            except socket.timeout:
                logger.error("[DL_Thread_GUI] 获取元数据超时"); self.status_update_signal.emit("错误：获取元数据超时"); self.keep_streaming_event.clear(); return
            finally: self.client_socket.settimeout(None)

            header_str = header_data.decode('utf-8').strip()
            if header_str.startswith("METADATA_OK "):
                parts = header_str.split(" ")
                total_segs_srv = int(parts[1])
                avg_dur_srv = float(parts[2])
                if avg_dur_srv <= 0: avg_dur_srv = 5.0 
                self.metadata_ready_signal.emit(total_segs_srv, avg_dur_srv) 
            else:
                logger.error(f"[DL_Thread_GUI] 获取元数据失败: {header_str}"); self.status_update_signal.emit(f"错误：元数据 {header_str}"); self.keep_streaming_event.clear(); return
        except Exception as e:
            logger.error(f"[DL_Thread_GUI] 获取元数据时出错: {e}"); self.status_update_signal.emit(f"错误：元数据异常 {e}"); self.keep_streaming_event.clear(); return

        # 等待GUI线程通过 self.total_segments_from_server 更新总分片数
        # 一个简单的等待，确保元数据已在主线程中被设置
        for _ in range(50): # 最多等5秒
            with self.lock:
                if self.total_segments_from_server > 0: break
            time.sleep(0.1)
        
        with self.lock:
            if self.total_segments_from_server == 0: 
                logger.warning("[DL_Thread_GUI] 总分片数为0，下载线程退出。")
                self.all_segments_processed_event.set() 
                self.status_update_signal.emit("没有可下载的分片。") # 使用信号
                return

        # 2. 下载分片
        for seg_idx in range(self.total_segments_from_server): # 使用从元数据获取的总数
            if not self.keep_streaming_event.is_set(): logger.info("[DL_Thread_GUI] 收到停止信号，终止下载。"); break
            
            # --- 缓冲逻辑 ---
            while self.keep_streaming_event.is_set():
                with self.lock:
                    current_playing_snapshot = self.currently_playing_original_idx
                    segments_in_pipe = 0 
                    for s_i, s_info in self.downloaded_segments_info.items():
                        if s_info.get('in_playlist',False) and s_i >= (current_playing_snapshot if current_playing_snapshot !=-1 else 0):
                            segments_in_pipe+=1
                player_active = self.media_list_player.get_state() in [vlc.State.Playing, vlc.State.Buffering, vlc.State.Opening]
                if (player_active and segments_in_pipe < FETCH_AHEAD_TARGET) or \
                   (not player_active and segments_in_pipe < MIN_BUFFER_TO_START_PLAY):
                    break
                else: time.sleep(0.1)
            if not self.keep_streaming_event.is_set(): break
            # --- 缓冲逻辑结束 ---

            with self.lock:
                if seg_idx in self.downloaded_segments_info and self.downloaded_segments_info[seg_idx].get('downloaded'):
                    if not self.downloaded_segments_info[seg_idx].get('in_playlist'): # 如果已下载但未加入列表
                         self.segment_downloaded_signal.emit(self.downloaded_segments_info[seg_idx]['path'], seg_idx)
                    continue
            
            s_filename_srv = f"{video_name}-{quality_suffix}-{seg_idx:03d}.ts"
            s_req_path_srv = f"{video_name}/{quality_suffix}/{s_filename_srv}"
            unique_id = f"{video_name}_{quality_suffix}_{seg_idx:03d}_{int(time.time()*1000000)}"
            local_s_filename = f"temp_{unique_id}.ts"
            local_s_path = os.path.join(DOWNLOAD_DIR, local_s_filename)

            logger.info(f"[DL_Thread_GUI] 请求分片 {seg_idx}: {s_req_path_srv}")
            try:
                if not self.client_socket or getattr(self.client_socket, '_closed', True) or self.client_socket.fileno() == -1:
                    self.status_update_signal.emit("错误：套接字在下载前关闭"); self.keep_streaming_event.clear(); break
                
                self.client_socket.sendall(f"GET {s_req_path_srv}\n".encode('utf-8'))
                s_header_data = b""
                self.client_socket.settimeout(SOCKET_TIMEOUT_SECONDS)
                try:
                    while not s_header_data.endswith(b"\n"):
                        s_byte = self.client_socket.recv(1)
                        if not s_byte: raise ConnectionError("服务器关闭连接 (分片头)")
                        s_header_data += s_byte
                except socket.timeout:
                    logger.error(f"[DL_Thread_GUI] 分片 {seg_idx} 头超时，重试"); time.sleep(RETRY_DOWNLOAD_DELAY); continue
                finally: self.client_socket.settimeout(None)

                s_header_str = s_header_data.decode('utf-8').strip()
                if s_header_str.startswith("OK "):
                    s_expected_size = int(s_header_str.split(" ",1)[1])
                    s_data = receive_exact_bytes(self.client_socket, s_expected_size)
                    with open(local_s_path, 'wb') as f_s: f_s.write(s_data)
                    if len(s_data) == s_expected_size:
                        self.segment_downloaded_signal.emit(local_s_path, seg_idx) # 通知GUI线程
                    else:
                        logger.error(f"[DL_Thread_GUI] 分片 {seg_idx} 大小不匹配")
                        if os.path.exists(local_s_path): os.remove(local_s_path)
                elif s_header_str.startswith("ERROR 404"):
                    logger.warning(f"[DL_Thread_GUI] 分片 {seg_idx} 404，流结束"); self.all_segments_processed_event.set(); self.keep_streaming_event.clear(); break
                else:
                    logger.error(f"[DL_Thread_GUI] 分片 {seg_idx} 服务器错误: {s_header_str}"); self.all_segments_processed_event.set(); self.keep_streaming_event.clear(); break
            except Exception as e_dl:
                logger.error(f"[DL_Thread_GUI] 下载分片 {seg_idx} 异常: {e_dl}"); self.status_update_signal.emit(f"错误：下载分片 {seg_idx} 异常"); self.keep_streaming_event.clear(); break
        
        # for 循环结束后
        if seg_idx + 1 >= self.total_segments_from_server and self.keep_streaming_event.is_set(): # 检查是否是正常完成所有下载
            self.all_segments_processed_event.set()
        
        self.status_update_signal.emit("所有分片已下载或处理完毕。") 
        logger.info("[DL_Thread_GUI] 下载任务结束。")


    def establish_socket_connection(self):
        """建立或重新建立到服务器的socket连接"""
        if self.client_socket and not getattr(self.client_socket, '_closed', True) and self.client_socket.fileno() != -1:
            return True 
        if self.client_socket:
            try: self.client_socket.close()
            except: pass
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            self.client_socket.settimeout(SOCKET_TIMEOUT_SECONDS / 2) # 连接超时可以短一些
            self.client_socket.connect((SERVER_HOST, SERVER_PORT))
            self.client_socket.settimeout(None) # 清除连接超时，后续操作使用各自的超时
            return True
        except Exception as e:
            logger.error(f"建立Socket连接失败: {e}"); self.client_socket = None; return False

    def closeEvent(self, event):
        """当用户关闭PyQt窗口时触发，确保所有资源被释放"""
        logger.info("窗口关闭事件。正在进行最终清理...")
        self.handle_stop_button_clicked() # 确保下载线程和播放器停止
        
        # 等待下载线程（如果它还在join）
        if self.download_thread and self.download_thread.is_alive():
            logger.info("CloseEvent: 等待下载线程结束...")
            self.download_thread.join(timeout=2.0)

        if self.client_socket:
             if not getattr(self.client_socket, '_closed', True) and self.client_socket.fileno() != -1:
                try:
                    logger.info("CloseEvent: 发送QUIT到服务器...")
                    self.client_socket.sendall(b"QUIT\n")
                except OSError: pass 
             try: self.client_socket.shutdown(socket.SHUT_RDWR)
             except OSError: pass
             finally: self.client_socket.close(); logger.info("CloseEvent: Socket已关闭")
        
        # 释放VLC资源
        if self.media_list_player: self.media_list_player.release(); self.media_list_player = None
        if self.media_player: self.media_player.release(); self.media_player = None
        if self.media_list: self.media_list.release(); self.media_list = None
        if self.vlc_instance: self.vlc_instance.release(); self.vlc_instance = None
        
        logger.info("客户端应用已关闭。")
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv) # 创建QApplication实例
    
    if os.name == 'nt':
        # Windows DLL 路径辅助 (如果VLC在PATH中，通常不需要)
        vlc_install_dir = None
        common_paths = [
            os.path.join(os.environ.get("ProgramFiles", "C:\\Program Files"), "VideoLAN", "VLC"),
            os.path.join(os.environ.get("ProgramFiles(x86)", "C:\\Program Files (x86)"), "VideoLAN", "VLC")
        ]
        for p_path in common_paths:
            if os.path.isdir(p_path) and os.path.exists(os.path.join(p_path, "libvlc.dll")):
                vlc_install_dir = p_path; break
        if vlc_install_dir and hasattr(os, 'add_dll_directory'): # Python 3.8+
            try: 
                # 注意：为了让 python-vlc 正确找到DLL，此调用理论上应在 `import vlc` 之前。
                # 但由于 `import vlc` 在脚本顶部，这里主要起提示作用。
                # 如果 `import vlc` 已经成功，说明 python-vlc 已能找到库。
                # os.add_dll_directory(vlc_install_dir) 
                logger.info(f"找到VLC安装目录: {vlc_install_dir}。确保它在系统PATH中或被python-vlc自动检测。")
            except Exception as e_dll: logger.warning(f"添加VLC DLL目录失败: {e_dll}")
        elif not vlc_install_dir : logger.warning("未在常见位置找到VLC。请确保VLC安装目录在系统PATH中。")

    
    if not os.path.exists(DOWNLOAD_DIR):
        os.makedirs(DOWNLOAD_DIR)
    else: # 清理旧的临时文件
        logger.info(f"启动时清理旧的临时文件于目录: {DOWNLOAD_DIR}...")
        for f_name in os.listdir(DOWNLOAD_DIR):
            if f_name.startswith("temp_") and f_name.endswith(".ts"):
                try: os.remove(os.path.join(DOWNLOAD_DIR, f_name))
                except OSError as e_rem: logger.warning(f"无法移除旧文件 {f_name}: {e_rem}")
    
    player_window = VideoPlayerWindow() # 创建主窗口实例
    sys.exit(app.exec_()) # 进入Qt事件循环