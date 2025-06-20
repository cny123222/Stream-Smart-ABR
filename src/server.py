import http.server
import socketserver
import os
import time
import logging
import urllib.parse
import AES

# --- 全局配置 ---
HOST = '0.0.0.0'  # 监听所有可用的网络接口
PORT = 8081       # 服务器监听的端口号
# BASE_SEGMENTS_DIR 应指向视频分片存储的根目录
# 期望的目录结构示例:
# BASE_SEGMENTS_DIR/
# └── 视频名称1/
#     ├── 质量1 (例如1080p-8000k)/
#     │   ├── ocean-1080p-8000k-0.ts
#     │   ├── ocean-1080p-8000k-1.ts
#     │   └── video1-1080p-8000k.m3u8  (媒体播放列表)
#     └── 质量2 (例如720p-4000k)/
#     │   └── ...
#     └── master.m3u8 (可选的主播放列表，指向不同质量的媒体播放列表)
BASE_SEGMENTS_DIR = "video_segments"  # 【重要】请修改为实际分片存储根目录
LOG_FILE = "transmission_log.txt" # 日志文件名
BUFFER_SIZE = 4096               # 文件传输时使用的缓冲区大小 (字节) - http.server内部处理

# --- 日志记录器设置 ---
logger = logging.getLogger('HLSServer')
logger.setLevel(logging.INFO)

fh = logging.FileHandler(LOG_FILE, encoding='utf-8')
fh.setLevel(logging.INFO)

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

# http.server 的 BaseHTTPRequestHandler 默认会进行一些日志记录
# client_ip 将从 HTTP 请求处理器中获取
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(client_ip)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)

logger.addHandler(fh)
logger.addHandler(ch)

class ClientIPLogAdapter(logging.LoggerAdapter):
    def process(self, msg, kwargs):
        if 'extra' not in kwargs:
            kwargs['extra'] = {}
        kwargs['extra'].update(self.extra)
        return msg, kwargs

def get_bitrate_from_filename(filename):
    """
    从分片文件名中提取码率信息。
    例如从 'video_name-1080p-8000k-000.ts' 中提取 '8000k'。
    """
    try:
        parts = filename.split('-')
        for part in reversed(parts):
            if part.endswith('k') and part[:-1].isdigit():
                return part
    except Exception:
        return "unknown_bitrate"
    return "unknown_bitrate"

class HLSRequestHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        """处理 HTTP GET 请求"""
        adapter = ClientIPLogAdapter(logger, {'client_ip': self.client_address[0]})
        
        try:
            parsed_path = urllib.parse.urlparse(self.path)
            request_path_cleaned = parsed_path.path.lstrip('/') # 移除开头的'/'
            
            # 安全性检查：防止目录遍历攻击
            # 将请求路径与基础目录结合，并规范化
            full_base_path = os.path.abspath(BASE_SEGMENTS_DIR)
            full_file_path = os.path.normpath(os.path.join(full_base_path, request_path_cleaned))
            
            # 确保规范化后的路径仍然在预期的服务目录下
            if not full_file_path.startswith(os.path.abspath(BASE_SEGMENTS_DIR)):
                adapter.warning(f"Path traversal attempt blocked: {request_path_cleaned} (resolved to: {full_file_path})")
                self.send_error(403, "Forbidden: Access is denied.")
                return

            if os.path.exists(full_file_path) and os.path.isfile(full_file_path):
                filename = os.path.basename(full_file_path)
                content_type = None
                is_binary = False
                data_to_send = None

                if filename.endswith(".m3u8"):
                    content_type = "application/vnd.apple.mpegurl" # HLS 播放列表
                    is_binary = False
                    with open(full_file_path, 'r', encoding='utf-8') as f:
                        data_to_send = f.read().encode('utf-8')
                    adapter.info(f"Serving M3U8 playlist: {request_path_cleaned}")

                elif filename.endswith(".ts"):
                    content_type = "video/MP2T" # MPEG2 传输流
                    is_binary = True
                    with open(full_file_path, 'rb') as f:
                        ts_data = f.read()
                    
                    # AES 加密分片数据
                    # 确保 AES.AES_KEY 在 AES.py 中定义或在此处可访问
                    send_start_time = time.time()
                    data_to_send = AES.aes_encrypt_cbc(ts_data, AES.AES_KEY) 
                    send_end_time = time.time()

                    file_size = len(data_to_send)
                    bitrate = get_bitrate_from_filename(filename)
                    send_duration = send_end_time - send_start_time
                    adapter.info(
                        f"Serving encrypted TS segment: {filename} ({file_size} bytes), "
                        f"bitrate(filename): {bitrate}, encryption & preparation time: {send_duration:.4f}s"
                    )
                else:
                    self.send_error(404, "File not found or unsupported type")
                    adapter.warning(f"Unsupported file type or no extension match found: {request_path_cleaned}")
                    return

                self.send_response(200)
                self.send_header("Content-type", content_type)
                self.send_header("Content-length", str(len(data_to_send)))
                self.send_header("Access-Control-Allow-Origin", "*") # 允许跨域请求
                self.end_headers()
                
                self.wfile.write(data_to_send)

            else:
                self.send_error(404, "File not found")
                adapter.warning(f"Requested file not found: {request_path_cleaned} (checked path: {full_file_path})")
        
        except ConnectionResetError:
            adapter.warning(f"Connection reset by client {self.client_address[0]}.")
        except Exception as e:
            adapter.error(f"Error handling GET request {self.path}: {e}", exc_info=True)
            try:
                self.send_error(500, "Internal server error")
            except BrokenPipeError: # 客户端可能已经断开连接
                adapter.warning(f"BrokenPipeError occurred while trying to send 500 error, client may have disconnected.")
            except Exception as e_send: # 其他发送错误
                adapter.error(f"Additional error occurred while trying to send 500 error: {e_send}")


    # 覆盖默认的日志方法以使用我们自定义的logger
    def log_message(self, format, *args):
        adapter = ClientIPLogAdapter(logger, {'client_ip': self.client_address[0]})
        adapter.info(format % args)

    def log_error(self, format, *args):
        adapter = ClientIPLogAdapter(logger, {'client_ip': self.client_address[0]})
        adapter.error(format % args)


class ThreadingHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True  # 允许主线程在工作线程仍在运行时退出
    allow_reuse_address = True # 允许服务器快速重启并重新绑定到同一地址和端口

def start_server():
    server_adapter = ClientIPLogAdapter(logger, {'client_ip': 'SERVER_MAIN'})
    try:
        # 使用我们自定义的 ThreadingHTTPServer 和 HLSRequestHandler
        httpd = ThreadingHTTPServer((HOST, PORT), HLSRequestHandler)
        
        server_adapter.info(f"HTTP HLS server is listening on {HOST}:{PORT}")
        server_adapter.info(f"Serving segments from directory: {os.path.abspath(BASE_SEGMENTS_DIR)}")
        server_adapter.info("Server is ready to handle concurrent requests.")
        
        httpd.serve_forever() # 启动服务器的无限循环来处理请求

    except OSError as e:
        server_adapter.error(f"Failed to start server: {e} (Hint: Is port {PORT} already in use?)")
    except KeyboardInterrupt:
        server_adapter.info("Server is shutting down due to keyboard interrupt...")
    finally:
        if 'httpd' in locals() and httpd:
            httpd.server_close() # 关闭服务器套接字
        server_adapter.info("Server has been successfully shut down.")

if __name__ == "__main__":
    main_adapter = ClientIPLogAdapter(logger, {'client_ip': 'SERVER_INIT'})
    
    # 检查 AES 密钥是否已定义 (假设它在 AES.py 中)
    if not hasattr(AES, 'AES_KEY') or not AES.AES_KEY:
        main_adapter.error("AES.AES_KEY is not defined in AES.py or is empty. Please ensure the key is set.")
    elif not callable(getattr(AES, 'aes_encrypt_cbc', None)):
        main_adapter.error("AES.aes_encrypt_cbc function is not defined in AES.py.")
    else:
        main_adapter.info(f"AES module loaded, AES_KEY exists.")

        if not os.path.exists(BASE_SEGMENTS_DIR) or not os.path.isdir(BASE_SEGMENTS_DIR):
            main_adapter.error(f"Base segments directory '{BASE_SEGMENTS_DIR}' not found or is not a directory.")
            main_adapter.error("Please create this directory first and ensure it contains HLS format video subdirectories (with .m3u8 and .ts files).")
        else:
            start_server()