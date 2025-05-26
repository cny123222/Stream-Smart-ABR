# 测试环境配置

## 服务端配置

### 测试视频下载

- **Big Buck Bunny** (大雄兔)是一个非常经典的测试动画片
- 资源来自 Blender Foundation 的开放电影
- 我们使用的是 **1080p_30fps** 的版本作为高质量母片，大小约 **276.1MB**
- [点击此处下载](https://download.blender.org/demo/movies/BBB/bbb_sunflower_1080p_30fps_normal.mp4.zip)
- 下载后，视频文件放在项目的根文件夹下，并重命名为 `bbb_sunflower.mp4`

### 测试视频分片及转码

1. 按照 `docs/ffmpeg_tutorial` 安装 FFmpeg
2. 命令行运行
```bash
python src/segment_video.py
```
该过程可能需要十几分钟，等待的时候可以看看这个电影（bushi）

### 服务端运行

1. 在项目下运行（Windows 可能需要管理员权限）
```bash
python src/server.py
```
2. 若失败，尝试更换监听端口号

## 客户端

### 下载 VLC 播放器

1. 从 [VideoLAN 官方网站](https://www.videolan.org/vlc/) 下载并安装最新稳定版的 VLC 播放器
2. 找到 VLC 的安装目录（如 `C:\Program Files\VideoLAN\VLC\`），将其添加到 PATH
3. 修改 `client.py` 中的 `PLAYER_PATH = "C:\\Program Files\\VideoLAN\\VLC\\vlc.exe"` (替换为 `vlc.exe` 的实际地址)

### 安装 vlc 库
```bash
pip install python-vlc
```

### 客户端运行

1. 在 `client.py` 中设置 `SERVER_HOST` （本地测试设置 `127.0.0.1`），`SERVER_PORT` （与服务端端口号一致）
2. 在项目下运行（Windows 可能需要管理员权限）
```bash
python src/client.py
```