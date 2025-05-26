import socket
import threading
import os
import time
import logging

# --- 全局配置 ---
HOST = '0.0.0.0'  # 监听所有可用的网络接口
PORT = 8081       # 服务器监听的端口号
# BASE_SEGMENTS_DIR 应指向视频分片存储的根目录
# 期望的目录结构示例:
# BASE_SEGMENTS_DIR/
# └── 视频名称1/
#     ├── 质量1 (例如1080p-8000k)/
#     │   ├── 分片-000.ts
#     │   ├── 分片-001.ts
#     │   └── 视频名称1-质量1.m3u8 (元数据文件)
#     └── 质量2 (例如720p-4000k)/
#         └── ...
# └── 视频名称2/
#     └── ...
BASE_SEGMENTS_DIR = "video_segments"  # 【重要】请修改为你的实际分片存储根目录
LOG_FILE = "transmission_log.txt"    # 日志文件名
BUFFER_SIZE = 4096                   # 文件传输时使用的缓冲区大小 (字节)

# --- 日志记录器设置 ---
# 获取名为 'StreamingServer' 的日志记录器实例
logger = logging.getLogger('StreamingServer')
logger.setLevel(logging.INFO) # 设置日志记录的最低级别为 INFO

# 文件处理器：将日志写入到文件
fh = logging.FileHandler(LOG_FILE, encoding='utf-8') # 使用utf-8编码写入日志文件
fh.setLevel(logging.INFO) # 文件处理器也只处理 INFO 及以上级别的日志

# 控制台处理器 (可选)：将日志输出到控制台
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

# 日志格式化器：定义日志的输出格式
# %(asctime)s: 时间, %(name)s: 记录器名, %(levelname)s: 日志级别,
# %(client_ip)s: 客户端IP (通过下面的适配器添加), %(message)s: 日志消息
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(client_ip)s - %(message)s')
fh.setFormatter(formatter) # 应用格式到文件处理器
ch.setFormatter(formatter) # 应用格式到控制台处理器

logger.addHandler(fh) # 添加文件处理器到记录器
logger.addHandler(ch) # 添加控制台处理器到记录器

# 自定义日志适配器，用于在日志记录中方便地添加 'client_ip' 属性
class ClientIPLogAdapter(logging.LoggerAdapter):
    # process 方法在实际记录日志前被调用，用于修改日志消息或关键字参数
    def process(self, msg, kwargs):
        # 如果 'extra' 字典没有在 kwargs 中，则初始化一个空字典
        # 'extra' 用于传递额外信息给日志记录，这些信息可以在格式化字符串中使用
        if 'extra' not in kwargs:
            kwargs['extra'] = {}
        # 将适配器初始化时传入的 'extra' 字典 (包含 client_ip) 更新到 kwargs['extra'] 中
        # 这样，格式化器中的 %(client_ip)s 就能获取到值
        kwargs['extra'].update(self.extra)
        return msg, kwargs # 返回原始消息和修改后的关键字参数

def get_bitrate_from_filename(filename):
    """
    从分片文件名中提取码率信息。
    例如从 'video_name-1080p-8000k-000.ts' 中提取 '8000k'。
    """
    try:
        parts = filename.split('-') # 使用 '-' 分割文件名
        # 从后向前遍历各部分，以更可靠地找到码率部分（如果视频名中也包含连字符）
        for part in reversed(parts):
            # 检查部分是否以 'k' 结尾，并且去掉 'k' 后剩余的是数字
            if part.endswith('k') and part[:-1].isdigit():
                return part # 返回码率字符串
    except Exception:
        return "unknown_bitrate" # 发生任何异常则返回未知
    return "unknown_bitrate" # 未找到则返回未知

def get_video_metadata(base_dir, video_name, quality_suffix):
    """
    计算并返回指定视频和质量的元数据。
    元数据包括：总分片数量 (total_segments) 和 平均分片时长 (avg_segment_duration)。
    优先尝试解析M3U8文件以获取精确信息，如果失败则回退到统计TS文件数量并使用预估时长。
    """
    # 构建特定质量视频分片所在的完整路径
    quality_path = os.path.join(base_dir, video_name, quality_suffix)
    # 创建一个用于此函数内部日志记录的适配器实例
    adapter = ClientIPLogAdapter(logger, {'client_ip': 'METADATA_FUNC'})

    # 检查路径是否存在且为目录
    if not os.path.isdir(quality_path):
        adapter.warning(f"元数据请求的路径不存在或不是目录: {quality_path}")
        return None, None # 返回 None 表示无法获取元数据

    # 构造M3U8文件的预期名称和完整路径
    m3u8_filename = f"{video_name}-{quality_suffix}.m3u8"
    m3u8_path = os.path.join(quality_path, m3u8_filename)
    
    segments_in_m3u8 = []      # 存储从M3U8中解析出的TS文件名
    durations_in_m3u8 = []     # 存储从M3U8中解析出的每个TS分片的时长

    # 检查M3U8文件是否存在
    if os.path.exists(m3u8_path):
        adapter.info(f"尝试解析M3U8文件: {m3u8_path}")
        try:
            # 以UTF-8编码打开并读取M3U8文件
            with open(m3u8_path, 'r', encoding='utf-8') as f_m3u8:
                for line in f_m3u8:
                    line = line.strip() # 去除行首尾的空白字符
                    if line.startswith("#EXTINF:"): # 如果行以 #EXTINF: 开头，表示包含分片时长信息
                        try:
                            # #EXTINF:5.000000, (后面可能有标题)
                            # 提取冒号后的时长部分，并去除逗号后的内容
                            duration_str = line.split(":")[1].split(",")[0]
                            durations_in_m3u8.append(float(duration_str)) # 转换为浮点数并存储
                        except (ValueError, IndexError):
                            adapter.warning(f"无法从M3U8行解析时长: {line}")
                    elif line and not line.startswith("#"): # 如果行非空且不以 # 开头，认为是TS文件名
                        segments_in_m3u8.append(line)
            
            # 校验从M3U8中解析出的分片数量和时长数量是否一致且有效
            if segments_in_m3u8 and durations_in_m3u8 and \
               len(segments_in_m3u8) == len(durations_in_m3u8):
                total_segments = len(segments_in_m3u8)
                # 计算平均分片时长，如果总分片数为0则默认为5.0秒 (防止除零错误)
                avg_segment_duration = sum(durations_in_m3u8) / total_segments if total_segments > 0 else 5.0
                adapter.info(f"从M3U8获取元数据: {total_segments} 个分片, 平均时长 {avg_segment_duration:.2f}秒")
                return total_segments, avg_segment_duration
            else:
                # M3U8文件内容不完整或不一致
                adapter.warning(f"M3U8文件 {m3u8_path} 解析不完整或不一致。"
                                f"找到 {len(segments_in_m3u8)} 个分片和 {len(durations_in_m3u8)} 个时长记录。")
        except Exception as e:
            adapter.error(f"解析M3U8文件 {m3u8_path} 时发生错误: {e}")
    
    # 如果M3U8文件不存在或解析失败，则回退到统计目录下的TS文件数量
    adapter.warning(f"M3U8文件未找到或解析失败: {quality_path}。回退到统计TS文件数量。")
    # 列出目录下所有符合命名规则 (video_name-quality_suffix-XXX.ts) 的TS文件
    ts_files = [f for f in os.listdir(quality_path) 
                if f.startswith(f"{video_name}-{quality_suffix}-") and f.endswith(".ts")]
    total_segments = len(ts_files) # TS文件数量即为总分片数
    
    if total_segments == 0:
        adapter.warning(f"在 {quality_path} 目录未找到TS分片文件。")
        return 0, 0.0 # 返回0个分片，时长为0

    # 使用预估的平均分片时长（例如5秒）。
    # 更理想的情况是，这个值应该与分片时实际使用的 -segment_time 参数一致。
    estimated_avg_duration = 5.0 
    adapter.info(f"通过统计TS文件获取元数据: {total_segments} 个分片, 预估平均时长 {estimated_avg_duration:.2f}秒")
    return total_segments, estimated_avg_duration


def handle_client(client_socket, client_address_tuple):
    """处理单个客户端连接的函数，在独立的线程中运行。"""
    client_ip_port = f"{client_address_tuple[0]}:{client_address_tuple[1]}"
    adapter = ClientIPLogAdapter(logger, {'client_ip': client_ip_port}) # 为此客户端的日志创建适配器
    
    adapter.info(f"新的客户端连接已建立。")
    try:
        while True: # 循环处理来自此客户端的多个请求
            client_socket.settimeout(300.0) # 为recv设置5分钟超时，防止客户端不发送数据导致线程永久阻塞
            try:
                request_data = client_socket.recv(1024).decode('utf-8') # 接收数据并解码
            except socket.timeout:
                adapter.warning("客户端空闲超时。正在关闭连接。")
                break # 超时则跳出循环，结束对此客户端的服务
            finally:
                client_socket.settimeout(None) # 完成recv后重置超时设置

            if not request_data: # 如果接收到空数据，表示客户端已关闭连接
                adapter.info(f"客户端主动关闭连接。")
                break

            request_stripped = request_data.strip() # 去除请求字符串首尾的空白字符
            adapter.info(f"收到请求: {request_stripped}")

            # --- 处理 GET <path_to_segment> 请求 ---
            if request_stripped.startswith("GET "):
                parts = request_stripped.split(" ", 1) # 按第一个空格分割 "GET" 和路径
                if len(parts) < 2: # 请求格式不正确
                    client_socket.sendall(b"ERROR 400 Invalid GET request format\n")
                    adapter.error(f"无效的GET请求格式: {request_stripped}")
                    continue # 继续等待下一个请求
                
                requested_path = parts[1].strip() # 获取请求的相对路径
                
                # 安全性检查：防止目录遍历攻击 (例如 "GET ../../secret_file")
                if ".." in requested_path:
                    client_socket.sendall(b"ERROR 400 Invalid path\n")
                    adapter.warning(f"检测到目录遍历尝试，已阻止路径: {requested_path}")
                    continue

                segment_filename = os.path.basename(requested_path) # 从路径中提取文件名
                # 构建分片文件在服务器上的完整物理路径
                full_segment_path = os.path.join(BASE_SEGMENTS_DIR, requested_path)
                full_segment_path = os.path.normpath(full_segment_path) # 规范化路径 (处理 / . .. 等)

                # 再次进行安全性检查：确保规范化后的路径仍在预期的服务目录下
                if not full_segment_path.startswith(os.path.normpath(BASE_SEGMENTS_DIR)):
                    client_socket.sendall(b"ERROR 400 Invalid path (outside base dir)\n")
                    adapter.warning(f"路径遍历尝试 (规范化后) 已阻止: {full_segment_path}")
                    continue
                
                # 检查文件是否存在且确实是一个文件
                if os.path.exists(full_segment_path) and os.path.isfile(full_segment_path):
                    try:
                        file_size = os.path.getsize(full_segment_path) # 获取文件大小
                        
                        # 响应客户端：先发送 "OK <file_size>\n" 头部
                        response_header = f"OK {file_size}\n"
                        client_socket.sendall(response_header.encode('utf-8'))
                        # adapter.info(f"为 {segment_filename} 发送头部信息: {response_header.strip()}") # 可以精简日志输出

                        send_start_time = time.time() # 记录开始发送的时间
                        sent_bytes = 0
                        # 以二进制读取模式打开文件并分块发送
                        with open(full_segment_path, 'rb') as f:
                            while True:
                                chunk = f.read(BUFFER_SIZE) # 读取一块数据
                                if not chunk: break # 文件读取完毕
                                client_socket.sendall(chunk) # 发送数据块
                                sent_bytes += len(chunk)
                        send_end_time = time.time() # 记录结束发送的时间
                        
                        # 校验发送的字节数是否与文件大小一致
                        if sent_bytes != file_size:
                             adapter.warning(f"文件大小不匹配 {segment_filename}. 期望 {file_size}, 发送 {sent_bytes}")

                        bitrate = get_bitrate_from_filename(segment_filename) # 从文件名提取码率
                        send_duration = send_end_time - send_start_time # 计算发送耗时
                        adapter.info(
                            f"已发送 {segment_filename} ({file_size} bytes) "
                            f"码率(文件名): {bitrate}, 发送耗时: {send_duration:.4f}s"
                        )
                    except ConnectionError as e: 
                        adapter.error(f"发送文件 {segment_filename} 时连接错误: {e}")
                        break # 连接出错，终止对此客户端的服务
                    except Exception as e:
                        adapter.error(f"发送文件 {segment_filename} 时发生错误: {e}")
                        try:
                            client_socket.sendall(b"ERROR 500 Server error during file send\n")
                        except: pass # 客户端可能已断开
                        break 
                else: # 文件未找到
                    adapter.warning(f"文件未找到: {full_segment_path} (客户端请求: {requested_path})")
                    client_socket.sendall(b"ERROR 404 File not found\n")
            
            # --- 处理 METADATA <video_name>/<quality_suffix> 请求 ---
            elif request_stripped.startswith("METADATA "):
                parts = request_stripped.split(" ", 1)
                if len(parts) < 2:
                    client_socket.sendall(b"ERROR 400 Invalid METADATA request format\n")
                    adapter.error(f"无效的METADATA请求格式: {request_stripped}")
                    continue

                path_parts_str = parts[1].strip() # 获取 video_name/quality_suffix 部分
                if ".." in path_parts_str: # 安全检查
                    client_socket.sendall(b"ERROR 400 Invalid METADATA path\n")
                    adapter.warning(f"METADATA请求中检测到目录遍历尝试: {path_parts_str}")
                    continue

                path_parts = path_parts_str.split('/') # 按 / 分割
                if len(path_parts) == 2: # 期望格式是 video_name/quality_suffix
                    video_name, quality_suffix = path_parts
                    # 调用函数获取元数据
                    total_segments, avg_duration = get_video_metadata(BASE_SEGMENTS_DIR, video_name, quality_suffix)
                    
                    if total_segments is not None and total_segments > 0 : # 确保获取到有效元数据且有分片
                        response = f"METADATA_OK {total_segments} {avg_duration:.2f}\n"
                        client_socket.sendall(response.encode('utf-8'))
                        adapter.info(f"已发送元数据 {video_name}/{quality_suffix}: {response.strip()}")
                    else: # 未找到元数据或没有分片
                        client_socket.sendall(b"ERROR 404 Metadata not found for specified video/quality or no segments\n")
                        adapter.warning(f"未找到元数据或无分片: {video_name}/{quality_suffix}")
                else: # METADATA 请求的路径格式不正确
                    client_socket.sendall(b"ERROR 400 Invalid METADATA path format (expected video_name/quality_suffix)\n")
                    adapter.error(f"无效的METADATA路径格式: {parts[1].strip()}")
            
            # --- 处理 QUIT 请求 ---
            elif request_stripped.upper() == "QUIT":
                 adapter.info(f"收到 QUIT 命令。正在关闭连接。")
                 break # 跳出循环，结束对此客户端的服务
            
            # --- 处理未知命令 ---
            else:
                adapter.warning(f"收到未知命令: {request_stripped}")
                client_socket.sendall(b"ERROR 400 Unknown command\n")

    except ConnectionResetError: # 客户端连接被重置
        adapter.warning(f"连接被对方重置。")
    except UnicodeDecodeError: # 客户端发送了非UTF-8编码的数据
        adapter.warning(f"收到非UTF-8编码数据。正在关闭连接。")
    except Exception as e: # 其他未预料的异常
        adapter.error(f"处理客户端时发生意外错误: {e}", exc_info=True) # exc_info=True 会记录完整的堆栈跟踪信息
    finally:
        client_socket.close() # 确保最终关闭套接字
        adapter.info(f"与客户端的连接已最终关闭。")


def start_server():
    """启动流媒体服务器。"""
    server_adapter = ClientIPLogAdapter(logger, {'client_ip': 'SERVER_MAIN'}) # 服务器主线程日志适配器
    # 创建一个TCP/IP套接字
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # 设置SO_REUSEADDR选项，允许服务器快速重启并重新绑定到同一地址和端口
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1) 
    try:
        # 将套接字绑定到指定的主机和端口
        server_socket.bind((HOST, PORT))
        # 开始监听传入连接，参数10表示允许的连接排队的最大数量
        server_socket.listen(10) 
        server_adapter.info(f"服务器正在监听 {HOST}:{PORT}")
        server_adapter.info(f"从目录提供分片服务: {os.path.abspath(BASE_SEGMENTS_DIR)}")

        while True: # 主循环，持续接受新的客户端连接
            try:
                # 接受一个新的客户端连接，accept()会阻塞直到有连接进来
                client_socket, client_address_tuple = server_socket.accept()
                # 为每个接受的连接创建一个新的线程来处理客户端请求
                # target=handle_client 指定线程要执行的函数
                # args=(...) 传递给函数的参数
                thread = threading.Thread(target=handle_client, args=(client_socket, client_address_tuple), name=f"ClientThread-{client_address_tuple[0]}")
                thread.daemon = True # 将线程设置为守护线程，这样主程序退出时它们也会退出
                thread.start() # 启动线程
            except Exception as e: # 捕获 accept 循环中可能发生的错误
                server_adapter.error(f"接受连接时发生错误: {e}")
                time.sleep(0.1) # 短暂休眠，防止因重复错误导致CPU占用过高

    except OSError as e: # 捕获服务器启动时可能发生的错误 (如端口被占用)
        server_adapter.error(f"服务器启动失败: {e} (提示: 端口 {PORT} 是否已被占用？)")
    except KeyboardInterrupt: # 捕获 Ctrl+C 中断信号
        server_adapter.info("服务器因键盘中断正在关闭...")
    finally:
        server_socket.close() # 确保服务器主套接字在退出时被关闭
        server_adapter.info("服务器已成功关闭。")

if __name__ == "__main__":
    main_adapter = ClientIPLogAdapter(logger, {'client_ip': 'SERVER_INIT'}) # 初始化日志适配器
    # 检查分片存储目录是否存在且为目录
    if not os.path.exists(BASE_SEGMENTS_DIR) or not os.path.isdir(BASE_SEGMENTS_DIR):
        main_adapter.error(f"基础分片目录 '{BASE_SEGMENTS_DIR}' 未找到或不是一个目录。")
        main_adapter.error("请先创建该目录，并确保其中包含已分片的视频子目录。")
    else:
        start_server() # 如果目录有效，则启动服务器