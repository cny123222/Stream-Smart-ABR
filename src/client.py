import os
import time
import logging
import threading
import requests
from urllib.parse import urlparse, urljoin, quote, unquote, parse_qs
import http.server
import socketserver
import socket
import webbrowser
import json # 用于WebSocket消息处理

# --- WebSocket和AsyncIO ---
import asyncio
import websockets

import AES # AES解密模块
from ABR import ABRManager, fetch_master_m3u8_for_abr_init
from QoE import QoEMetricsManager # QoE指标管理器
import network_simulator # 网络模拟模块
import argparse

# --- 配置 ---
REBUFFERING = False
SERVER_HOST = '127.0.0.1'
SERVER_PORT = 8081
LOCAL_PROXY_HOST = '127.0.0.1'
LOCAL_PROXY_PORT = 8082
WEBSOCKET_PORT = 8083 # WebSocket服务器端口
DOWNLOAD_DIR = "download" # 在HLS.js直接播放时不常用
SOCKET_TIMEOUT_SECONDS = 10
VIDEO_TO_STREAM_NAME = "bbb_sunflower"
HTML_TEMPLATE_PATH = os.path.join(os.path.dirname(__file__), "player.html")

# --- 日志记录器设置 ---
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(threadName)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler()])
logger = logging.getLogger('HLSJS_Client_With_QoE')

# --- 本地代理服务器全局变量 ---
g_local_proxy_server_instance = None
g_proxy_runner_thread = None

# --- WebSocket服务器全局变量 ---
g_connected_websocket_clients = set()
g_websocket_server_thread = None
g_asyncio_loop_for_websocket = None
g_websocket_stop_event = None # 将是一个asyncio.Event

# 全局实例
qoe_manager = QoEMetricsManager()

# --- WebSocket服务器函数（handle_websocket_client需要路由QoE消息） ---
async def handle_websocket_client(websocket):
    global g_connected_websocket_clients, REBUFFERING
    client_identifier = getattr(websocket, 'path', None)
    if client_identifier is None:
        try: 
            client_identifier = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}"
        except Exception: 
            client_identifier = "UnknownClient" # 如果remote_address也失败的备用方案
            
    logger.info(f"WebSocket client connected: {client_identifier}")
    g_connected_websocket_clients.add(websocket)
    try:
        async for message_str in websocket:
            logger.debug(f"WebSocket received from {client_identifier}: {message_str}")
            try:
                message = json.loads(message_str)
                if message.get("type") == "QOE_EVENT":
                    event_data = message.get("data", {})
                    print('--------------  ', event_data, ' -----------awawa')
                    event_name = event_data.get("event")
                    if event_data.get("value") is not None and event_data.get("value") < 1.0 and not REBUFFERING:
                        REBUFFERING = True
                        event_name = "REBUFFERING_START"
                    elif event_data.get("value") is not None and event_data.get("value") >= 1.0 and REBUFFERING:
                        REBUFFERING = False
                        event_name = "REBUFFERING_END"
                    timestamp = event_data.get("timestamp", time.time() * 1000)

                    if event_name == "STARTUP_LATENCY":
                        qoe_manager.record_startup_latency(event_data.get("value"), timestamp)
                    elif event_name == "REBUFFERING_START":
                        REBUFFERING = True
                        qoe_manager.record_rebuffering_start(timestamp)
                    elif event_name == "REBUFFERING_END":
                        REBUFFERING = False
                        qoe_manager.record_rebuffering_end(timestamp)
                    elif event_name == "QUALITY_SWITCH":
                        qoe_manager.record_quality_switch(
                            event_data.get("fromLevel"),
                            event_data.get("toLevel"),
                            event_data.get("toBitrate"),
                            timestamp
                        )
                    elif event_name == "PLAYBACK_ENDED":
                        qoe_manager.log_playback_session_end(timestamp)
                    elif event_name == "BUFFER_UPDATE":
                        buffer_value_seconds = event_data.get("value")
                        logger.debug(f"QoE Event from {client_identifier}: Buffer Update = {buffer_value_seconds:.2f}s at {timestamp}")
                        # 将缓冲区信息传递给 ABRManager
                        if ABRManager.instance:
                            # 需要在ABRManager中添加一个方法来接收这个值
                            ABRManager.instance.update_player_buffer_level(buffer_value_seconds) 
                        else:
                            logger.warning("ABRManager instance not available to update buffer level.")
                    else:
                        logger.warning(f"Unknown QoE event name from {client_identifier}: {event_name}")
                # else: # 如果有其他双向通信，可以处理非QoE消息
                #     logger.debug(f"Received non-QoE message from {client_identifier}: {message_str}")

            except json.JSONDecodeError:
                logger.error(f"Failed to decode JSON from WebSocket ({client_identifier}): {message_str}")
            except Exception as e:
                logger.error(f"Error processing message from {client_identifier}: {e}", exc_info=True)
    except websockets.exceptions.ConnectionClosedOK:
        logger.info(f"WebSocket client {client_identifier} disconnected gracefully.")
    except websockets.exceptions.ConnectionClosedError as e:
        logger.warning(f"WebSocket client {client_identifier} connection closed with error: {e}")
    except Exception as e:
        logger.error(f"Error in WebSocket client handler for {client_identifier}: {e}", exc_info=True)
    finally:
        g_connected_websocket_clients.remove(websocket)
        logger.info(f"WebSocket client {client_identifier} removed from connected set.")

async def run_websocket_server_async():
    global g_asyncio_loop_for_websocket, g_websocket_stop_event # g_asyncio_loop_for_websocket由调用线程设置

    # 健全性检查：确保我们认为使用的循环确实是当前运行的循环
    # 这部分主要用于调试；websockets.serve应该使用get_running_loop()
    try:
        current_loop = asyncio.get_running_loop()
        if g_asyncio_loop_for_websocket is not current_loop:
            logger.warning(
                f"run_websocket_server_async: g_asyncio_loop_for_websocket (id: {id(g_asyncio_loop_for_websocket)}) "
                f"is not the current running loop (id: {id(current_loop)}). This might be an issue if not intended."
            )
            # 如果start_websocket_server_in_thread正确设置了线程的循环，
            # 然后在该循环上调用run_until_complete来执行此协程，
            # 那么在此协程内get_running_loop()应该返回同一个循环。
    except RuntimeError:
        logger.error("run_websocket_server_async: No current asyncio loop running when expected!")
        return # 没有循环无法继续

    if g_websocket_stop_event is None:
        # 在此协程运行的循环上下文中创建事件
        g_websocket_stop_event = asyncio.Event() 

    logger.info(f"Starting WebSocket server on ws://{LOCAL_PROXY_HOST}:{WEBSOCKET_PORT}")
    
    server_instance = None
    try:
        # 从websockets.serve调用中移除loop=g_asyncio_loop_for_websocket
        async with websockets.serve(handle_websocket_client, LOCAL_PROXY_HOST, WEBSOCKET_PORT) as server:
            server_instance = server 
            server_address = "N/A"
            if server.sockets:
                try:
                    server_address = server.sockets[0].getsockname()
                except Exception: # 处理getsockname可能无法立即获得或在所有套接字类型上不可用的情况
                    pass
            logger.info(f"WebSocket server '{server_address}' now serving.")
            await g_websocket_stop_event.wait() 
    except asyncio.CancelledError:
        logger.info("WebSocket server task (run_websocket_server_async) was cancelled.")
    except Exception as e: # 捕获serve()期间的其他潜在错误
        logger.error(f"Error during websockets.serve: {e}", exc_info=True)
    finally:
        if server_instance:
            logger.info("Closing WebSocket server instance...")
            server_instance.close()
            try:
                await server_instance.wait_closed() 
            except Exception as e_close:
                logger.error(f"Error during server_instance.wait_closed: {e_close}")
        logger.info("WebSocket server (run_websocket_server_async) has shut down.")

def start_websocket_server_in_thread():
    global g_websocket_server_thread, g_asyncio_loop_for_websocket
    
    thread_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(thread_loop)
    # 这个全局变量现在正确保存了由此WebSocket服务器线程独占管理的循环对象。
    g_asyncio_loop_for_websocket = thread_loop 

    try:
        thread_loop.run_until_complete(run_websocket_server_async())
    except KeyboardInterrupt: 
        logger.info("WebSocket server thread (run_until_complete) received KeyboardInterrupt.")
    except SystemExit: 
        logger.info("WebSocket server asyncio loop stopping (SystemExit).")
    finally:
        logger.info("WebSocket server thread: Cleaning up asyncio loop.")
        try:
            # 尝试取消此特定循环中的所有剩余任务
            all_tasks = asyncio.all_tasks(loop=thread_loop)
            if all_tasks:
                logger.info(f"Cancelling {len(all_tasks)} outstanding asyncio tasks in WebSocket thread loop...")
                for task in all_tasks:
                    if not task.done() and not task.cancelled(): # 检查是否需要取消
                        task.cancel()
                # 等待任务处理取消
                # 这个gather应该由循环本身运行。
                # 但是，run_until_complete已经退出。
                # 如果任务需要清理，我们可能需要在短暂的最终run_until_complete中运行gather。
                # 为了简单起见，我们假设取消已设置。
                # thread_loop.run_until_complete(asyncio.gather(*all_tasks, return_exceptions=True))
                # 如果循环已经停止，这可能很棘手。
            
            # 如果循环由于g_websocket_stop_event.set() -> run_until_complete完成而停止，
            # 那么我们继续关闭。
            if thread_loop.is_running(): # 如果run_until_complete干净退出，应该不为真
                logger.warning("WebSocket thread loop still running unexpectedly during cleanup; attempting stop.")
                thread_loop.stop()
            if not thread_loop.is_closed():
                thread_loop.close()
                logger.info("WebSocket server asyncio loop closed by thread.")
            else:
                logger.info("WebSocket server asyncio loop was already closed.")
        except RuntimeError as e:
             logger.error(f"RuntimeError during WebSocket thread loop cleanup: {e}")


async def broadcast_message_async(message_str):
    if g_connected_websocket_clients:
        # 创建发送消息的任务列表，避免在迭代期间集合发生变化的问题
        clients_to_send_to = list(g_connected_websocket_clients)
        tasks = [client.send(message_str) for client in clients_to_send_to]
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    client_repr = str(clients_to_send_to[i].remote_address) if hasattr(clients_to_send_to[i], 'remote_address') else "Unknown Client"
                    logger.error(f"Error sending message to WebSocket client {client_repr}: {result}")

def schedule_abr_broadcast(level_index):
    if g_asyncio_loop_for_websocket and g_asyncio_loop_for_websocket.is_running():
        message = json.dumps({"type": "SET_LEVEL", "levelIndex": level_index})
        # 安排异步广播函数在WebSocket的asyncio循环中运行
        asyncio.run_coroutine_threadsafe(broadcast_message_async(message), g_asyncio_loop_for_websocket)
    else:
        logger.warning("Cannot schedule ABR broadcast: WebSocket asyncio loop not available or not running.")
        
def schedule_abr_bw_estimate_broadcast(estimated_mbps):
    if g_asyncio_loop_for_websocket and g_asyncio_loop_for_websocket.is_running():
        message = json.dumps({"type": "ABR_BW_ESTIMATE_UPDATE", "data": {"estimated_Mbps": estimated_mbps}})
        asyncio.run_coroutine_threadsafe(broadcast_message_async(message), g_asyncio_loop_for_websocket)
    else:
        logger.warning("Cannot schedule ABR BW Estimate broadcast: WebSocket asyncio loop not available.")
        
def schedule_network_sim_status_broadcast(status_data):
    if g_asyncio_loop_for_websocket and g_asyncio_loop_for_websocket.is_running():
        message = json.dumps({"type": "NETWORK_SIM_UPDATE", "data": status_data})
        asyncio.run_coroutine_threadsafe(broadcast_message_async(message), g_asyncio_loop_for_websocket)
    else:
        logger.warning("Cannot schedule Network Sim Status broadcast: WebSocket asyncio loop not available or not running.")

class DecryptionProxyHandler(http.server.BaseHTTPRequestHandler):
    # 确保_rewrite_master_playlist向HLS.js提供所有变体
    # 这样HLS.js知道所有可用级别及其原始顺序。

    def do_GET(self):
        log_adapter = logging.LoggerAdapter(logger, {'path': self.path})
        request_log_tag = f"[ProxyRequest URI: {self.path}]"
        parsed_url = urlparse(self.path)
        path_components = parsed_url.path.strip('/').split('/')

        try:
            if parsed_url.path == '/' or parsed_url.path == '/player.html':
                log_adapter.info(f"{request_log_tag} Serving player.html")
                try:
                    with open(HTML_TEMPLATE_PATH, 'r', encoding='utf-8') as f:
                        html_content_template = f.read()

                    # 执行替换
                    html_content = html_content_template.replace("{LOCAL_PROXY_HOST}", LOCAL_PROXY_HOST) \
                                                    .replace("{LOCAL_PROXY_PORT}", str(LOCAL_PROXY_PORT)) \
                                                    .replace("{WEBSOCKET_PORT}", str(WEBSOCKET_PORT)) \
                                                    .replace("{VIDEO_TO_STREAM_NAME}", VIDEO_TO_STREAM_NAME)

                    self.send_response(200)
                    self.send_header('Content-type', 'text/html; charset=utf-8')
                    self.end_headers()
                    self.wfile.write(html_content.encode('utf-8'))
                except FileNotFoundError:
                    logger.error(f"HTML template file not found at {HTML_TEMPLATE_PATH}")
                    self.send_error(404, "HTML Player File Not Found")
                except Exception as e_html:
                    logger.error(f"Error serving HTML player: {e_html}", exc_info=True)
                    self.send_error(500, "Error serving HTML player")
                return

            if len(path_components) == 2 and path_components[0] == VIDEO_TO_STREAM_NAME and path_components[1] == "master.m3u8":
                video_name_from_url = path_components[0]
                original_master_m3u8_url = f"http://{SERVER_HOST}:{SERVER_PORT}/{video_name_from_url}/master.m3u8"
                log_adapter.info(f"{request_log_tag} Request for master M3U8. Fetching from: {original_master_m3u8_url}")
                try:
                    response = requests.get(original_master_m3u8_url, timeout=SOCKET_TIMEOUT_SECONDS)
                    response.raise_for_status()
                    master_content = response.text
                    # 此重写必须确保所有原始变体都存在，
                    # 这样HLS.js知道所有级别索引。
                    # 媒体播放列表的URL被代理。
                    modified_master_content = self._rewrite_master_playlist(master_content, original_master_m3u8_url)
                    
                    self.send_response(200)
                    self.send_header('Content-type', 'application/vnd.apple.mpegurl')
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    self.wfile.write(modified_master_content.encode('utf-8'))
                except requests.exceptions.RequestException as e:
                    self.send_error(502, f"Bad Gateway: Could not fetch master M3U8: {e}")
                return

            if len(path_components) == 3 and path_components[0] == VIDEO_TO_STREAM_NAME and path_components[2].endswith(".m3u8"):
                video_name_from_url = path_components[0]
                quality_dir_from_url = path_components[1]
                playlist_filename_from_url = path_components[2]
                original_media_m3u8_url = f"http://{SERVER_HOST}:{SERVER_PORT}/{video_name_from_url}/{quality_dir_from_url}/{playlist_filename_from_url}"
                try:
                    response = requests.get(original_media_m3u8_url, timeout=SOCKET_TIMEOUT_SECONDS)
                    response.raise_for_status()
                    media_content = response.text
                    modified_media_content = self._rewrite_media_playlist(media_content, original_media_m3u8_url)
                    self.send_response(200)
                    self.send_header('Content-type', 'application/vnd.apple.mpegurl')
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    self.wfile.write(modified_media_content.encode('utf-8'))
                except requests.exceptions.RequestException as e:
                    self.send_error(502, f"Bad Gateway: Could not fetch media M3U8: {e}")
                return

            if parsed_url.path == '/decrypt_segment':
                query_params = parse_qs(parsed_url.query)
                original_ts_url_on_server_encoded = query_params.get('url', [None])[0]
                if not original_ts_url_on_server_encoded:
                    self.send_error(400, "Bad Request: Missing 'url' parameter")
                    return
                original_ts_url_on_server = unquote(original_ts_url_on_server_encoded)
                segment_filename_for_log = original_ts_url_on_server.split('/')[-1]
                
                effective_segment_download_duration_for_abr = -1.0 # 初始化
                data_size_bytes = 0 # 初始化

                try:
                    # 步骤1：从源获取分片
                    # fetch_start_time = time.time() # 如果ABR依赖代理->客户端时间则不严格需要
                    response_ts = requests.get(original_ts_url_on_server, timeout=SOCKET_TIMEOUT_SECONDS, stream=False)
                    response_ts.raise_for_status()
                    encrypted_data = response_ts.content
                    # fetch_end_time = time.time() # 如果ABR依赖代理->客户端时间则不严格需要

                    if not encrypted_data:
                        self.send_error(502, "Bad Gateway: Empty TS content from origin")
                        return
                    
                    decrypted_data = AES.aes_decrypt_cbc(encrypted_data, AES.AES_KEY)
                    data_size_bytes = len(decrypted_data)

                    # 在开始数据传输之前发送HTTP头（无论是否限流）
                    self.send_response(200)
                    self.send_header('Content-type', 'video/MP2T')
                    self.send_header('Content-Length', str(data_size_bytes))
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()

                    # 步骤2：使用network_simulator模拟发送到hls.js的带宽
                    current_sim_bps = network_simulator.get_current_simulated_bandwidth()

                    if current_sim_bps is not None and current_sim_bps > 0:
                        # 限流传输
                        effective_segment_download_duration_for_abr = network_simulator.throttle_data_transfer(
                            decrypted_data,
                            current_sim_bps,
                            self.wfile, # 传递输出流（套接字）
                            segment_filename_for_log
                        )
                    else:
                        # 无模拟，尽可能快地发送
                        log_adapter.info(f"{request_log_tag} - Sending segment {segment_filename_for_log} ({data_size_bytes / 1024:.1f} KB) at full speed.")
                        _t_actual_send_start = time.time()
                        self.wfile.write(decrypted_data)
                        self.wfile.flush()
                        _t_actual_send_end = time.time()
                        effective_segment_download_duration_for_abr = _t_actual_send_end - _t_actual_send_start
                        if effective_segment_download_duration_for_abr < 0.001: # 避免零或太小的值
                            effective_segment_download_duration_for_abr = 0.001
                
                except socket.error as e_sock: # 在self.wfile.write期间捕获套接字错误（无论是否限流）
                    logger.warning(f"{request_log_tag} Socket error sending segment {segment_filename_for_log} to hls.js: {e_sock}")
                    if ABRManager.instance: ABRManager.instance.report_download_error(original_ts_url_on_server)
                    # 如果套接字损坏，不要尝试发送进一步的错误响应
                    return 
                except requests.exceptions.RequestException as e_req: # 从源获取时出错
                    if ABRManager.instance: ABRManager.instance.report_download_error(original_ts_url_on_server)
                    self.send_error(502, f"Bad Gateway: Could not fetch TS: {e_req}")
                    return
                except Exception as e_proc: # 其他错误如解密等
                    logger.error(f"{request_log_tag} Error processing segment {segment_filename_for_log}: {e_proc}", exc_info=True)
                    if ABRManager.instance: ABRManager.instance.report_download_error(original_ts_url_on_server)
                    # 如果头部尚未发送或套接字正常，尝试发送错误
                    try:
                        if not self.headers_sent: # 有点启发式，可能需要更好的检查
                           self.send_error(500, f"Internal Server Error: Segment processing failed: {e_proc}")
                    except Exception as e_send_err:
                        logger.error(f"{request_log_tag} Further error sending 500: {e_send_err}")
                    return
                
                # 步骤3：报告给ABRManager（仅在上述没有阻止发送数据的致命错误时）
                if ABRManager.instance and effective_segment_download_duration_for_abr > 0 and data_size_bytes > 0:
                    ABRManager.instance.add_segment_download_stat(
                        original_ts_url_on_server,
                        data_size_bytes,
                        effective_segment_download_duration_for_abr
                    )
                return # /decrypt_segment结束
            
            self.send_error(404, "Not Found")
        except requests.exceptions.RequestException as e: 
            logger.error(f"{request_log_tag} Proxy HTTP fetch error: {e}"); 
            self.send_error(502, f"Bad Gateway: {e}")
        except Exception as e: 
            logger.error(f"{request_log_tag} Proxy error: {e}", exc_info=True); 
            self.send_error(500, "Internal Server Error")
            
    def _rewrite_master_playlist(self, master_content, original_master_url):
        lines = master_content.splitlines(); modified_lines = []
        for i in range(len(lines)):
            line = lines[i].strip()
            if line.startswith("#EXT-X-STREAM-INF:"):
                modified_lines.append(line) 
                if i + 1 < len(lines):
                    media_playlist_relative_or_absolute_url_in_master = lines[i+1].strip()
                    original_media_playlist_absolute_url = urljoin(original_master_url, media_playlist_relative_or_absolute_url_in_master)
                    parsed_original_media_playlist_url = urlparse(original_media_playlist_absolute_url)
                    proxied_media_playlist_url = f"http://{LOCAL_PROXY_HOST}:{LOCAL_PROXY_PORT}{parsed_original_media_playlist_url.path}"
                    modified_lines.append(proxied_media_playlist_url)
                    i += 1 
            else:
                modified_lines.append(line)
        return "\n".join(modified_lines)
    
    def _rewrite_media_playlist(self, media_content, original_media_url):
        lines = media_content.splitlines(); modified_lines = []
        base_url = urljoin(original_media_url, '.')
        for line in lines:
            s_line = line.strip()
            if s_line and not s_line.startswith("#") and (s_line.endswith(".ts") or ".ts?" in s_line):
                abs_seg_url = urljoin(base_url, s_line); enc_seg_url = quote(abs_seg_url, safe='')
                modified_lines.append(f"http://{LOCAL_PROXY_HOST}:{LOCAL_PROXY_PORT}/decrypt_segment?url={enc_seg_url}")
            else: modified_lines.append(line)
        return "\n".join(modified_lines)
    def log_message(self, format, *args): logger.debug(f"ProxyHTTP: {self.address_string()} - {args[0]} {args[1]}")

class ThreadingLocalProxyServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True

def _run_proxy_server_target():
    global g_local_proxy_server_instance
    try:
        g_local_proxy_server_instance = ThreadingLocalProxyServer(
            (LOCAL_PROXY_HOST, LOCAL_PROXY_PORT), DecryptionProxyHandler)
        logger.info(f"PROXY_THREAD: Local proxy server starting on http://{LOCAL_PROXY_HOST}:{LOCAL_PROXY_PORT}")
        g_local_proxy_server_instance.serve_forever() 
    except Exception as e: logger.error(f"PROXY_THREAD: Error: {e}", exc_info=True); g_local_proxy_server_instance = None
    finally: logger.info("PROXY_THREAD: Proxy server loop finished.")
    

def start_proxy_server():
    global g_proxy_runner_thread, g_local_proxy_server_instance
    if g_proxy_runner_thread and g_proxy_runner_thread.is_alive(): 
        return bool(g_local_proxy_server_instance) 
    g_local_proxy_server_instance = None 
    g_proxy_runner_thread = threading.Thread(target=_run_proxy_server_target, daemon=True, name="ProxyServerThread")
    g_proxy_runner_thread.start()
    time.sleep(0.5) 
    return bool(g_proxy_runner_thread.is_alive() and g_local_proxy_server_instance)


def stop_proxy_server():
    global g_local_proxy_server_instance, g_proxy_runner_thread
    if g_local_proxy_server_instance:
        logger.info("PROXY_MAIN: Stopping proxy server instance...")
        g_local_proxy_server_instance.shutdown()
        if g_proxy_runner_thread: g_proxy_runner_thread.join(timeout=2.0)
        g_local_proxy_server_instance.server_close()
        g_local_proxy_server_instance = None
        g_proxy_runner_thread = None
        logger.info("PROXY_MAIN: Proxy server stopped.")

def main():
    global g_websocket_server_thread, g_asyncio_loop_for_websocket, qoe_manager

    # 创建解析器对象
    parser = argparse.ArgumentParser(description='Client application with two integer arguments')
    # 添加第一个整数参数
    parser.add_argument('arg1', type=int, help='abr_decision')
    # 添加第二个整数参数
    parser.add_argument('arg2', type=int, help='network_condition')
    # 添加第三个字符串参数
    parser.add_argument('arg3', type=str, help='write_path')
    # 解析命令行参数
    args = parser.parse_args()

    abr_decision = args.arg1
    # [1, 3]
    net_decision = args.arg2
    # [1, 8]
    write_path = args.arg3

    qoe_manager.set_write_path(write_path)
    abr_manager_instance = None
    scenario_player = None # 在这里定义scenario_player以便在finally中访问

    # AES密钥检查，DOWNLOAD_DIR创建
    if not hasattr(AES, 'AES_KEY') or not AES.AES_KEY: logger.error("AES_KEY missing!"); return
    if not callable(getattr(AES, 'aes_decrypt_cbc', None)): logger.error("aes_decrypt_cbc missing!"); return
    if not os.path.exists(DOWNLOAD_DIR): os.makedirs(DOWNLOAD_DIR)


    if not start_proxy_server(): 
        logger.error("HTTP Proxy server failed to start.")
        return

    logger.info("Starting WebSocket server thread...")
    g_websocket_server_thread = threading.Thread(target=start_websocket_server_in_thread, daemon=True, name="WebSocketServerThread")
    g_websocket_server_thread.start()
    time.sleep(1)
    
    if not (g_asyncio_loop_for_websocket and g_asyncio_loop_for_websocket.is_running()):
        logger.warning("WebSocket asyncio loop doesn't seem to be running. ABR/QoE control might fail.")

    master_m3u8_url = f"http://{SERVER_HOST}:{SERVER_PORT}/{VIDEO_TO_STREAM_NAME}/master.m3u8"
    available_streams = fetch_master_m3u8_for_abr_init(master_m3u8_url)
    if available_streams:
        # --- 选择想要的决策逻辑 ---
        # selected_logic = ABRManager.LOGIC_TYPE_BANDWIDTH_ONLY
        selected_logic = ABRManager.LOGIC_TYPE_BANDWIDTH_BUFFER
        # selected_logic = ABRManager.LOGIC_TYPE_ENHANCED_BUFFER_RESPONSE 
        if abr_decision == 1:
            selected_logic = ABRManager.LOGIC_TYPE_BANDWIDTH_ONLY
        elif abr_decision == 2:
            selected_logic = ABRManager.LOGIC_TYPE_BANDWIDTH_BUFFER
        elif abr_decision == 3:
            selected_logic = ABRManager.LOGIC_TYPE_BUFFER_ONLY
        else:
            logger.error("Invalid ABR decision. Using default logic.")
            selected_logic = ABRManager.LOGIC_TYPE_BANDWIDTH_BUFFER
            
        abr_manager_instance = ABRManager(
            available_streams,
            broadcast_abr_decision_callback=schedule_abr_broadcast,
            broadcast_bw_estimate_callback=schedule_abr_bw_estimate_broadcast,
            logic_type=selected_logic
        )
        abr_manager_instance.start()
    else:
        logger.error("Could not fetch streams for ABR init. ABR will not function.")

    # --- 初始化并启动网络场景播放器 ---
    logger.info("MAIN: Initializing and starting network scenario player...")
    network_simulator.set_bandwidth_update_callback(schedule_network_sim_status_broadcast)
    scenario_player = network_simulator.create_default_simulation_scenario(mode = net_decision)
    scenario_player = network_simulator.create_default_simulation_scenario()
    # scenario_player = network_simulator.NetworkScenarioPlayer()
    # scenario_player.add_step(duration_seconds=10, bandwidth_bps=None) # 10秒全速
    # scenario_player.add_step(duration_seconds=20, bandwidth_bps=1_000_000) # 20秒1Mbps
    # ...
    scenario_player.start()


    player_url = f"http://{LOCAL_PROXY_HOST}:{LOCAL_PROXY_PORT}/player.html"
    logger.info(f"Opening player at: {player_url}")
    webbrowser.open(player_url)
    logger.info("Client setup complete. Press Ctrl+C to stop.")
    
    try:
        while True: 
            time.sleep(60)
            # qoe_manager.print_summary() # 可选的定期摘要
    except KeyboardInterrupt:
        logger.info("Ctrl+C pressed. Shutting down...")
    finally:
        logger.info("Initiating cleanup...")
        
        abr_streams_info = None
        if abr_manager_instance and hasattr(abr_manager_instance, 'available_streams'):
            abr_streams_info = abr_manager_instance.available_streams
        qoe_manager.log_playback_session_end(available_abr_streams=abr_streams_info) # 记录QoE会话结束
        
        if abr_manager_instance:
            logger.info("Stopping ABR Manager...")
            abr_manager_instance.stop()

        if scenario_player: # 停止场景播放器
            logger.info("Stopping Network Scenario Player...")
            scenario_player.stop()

        if g_asyncio_loop_for_websocket and g_websocket_stop_event:
            if not g_asyncio_loop_for_websocket.is_closed():
                logger.info("Signaling WebSocket server to stop...")
                g_asyncio_loop_for_websocket.call_soon_threadsafe(g_websocket_stop_event.set)
            else:
                logger.info("WebSocket asyncio loop already closed.")

        if g_websocket_server_thread and g_websocket_server_thread.is_alive():
            logger.info("Waiting for WebSocket server thread to join...")
            g_websocket_server_thread.join(timeout=5.0)
            if g_websocket_server_thread.is_alive(): 
                logger.warning("WebSocket server thread did not join cleanly.")

        stop_proxy_server() # 停止HTTP代理服务器
        logger.info("Client application finished.")

if __name__ == "__main__":
    main()