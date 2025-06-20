# 测试环境配置

## 服务端配置

### 测试视频下载

- **Big Buck Bunny** (大雄兔)是一个非常经典的测试动画片
- 资源来自 Blender Foundation 的开放电影
- 我们使用的是 **2160p_30fps** 的版本作为高质量母片
- [点击此处下载](https://download.blender.org/demo/movies/BBB/bbb_sunflower_2160p_30fps_normal.mp4.zip)
- 也可以使用 **1080p_30fps** 的版本
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

1. 在项目文件夹下运行
```bash
python src/server.py
```
- 注：Windows 需要管理员权限，推荐用管理员权限运行vscode，在vscode中的命令行窗口中操作
2. 若失败，尝试更换监听端口号

## 客户端

### 客户端运行

1. 在 `client.py` 中设置 `SERVER_HOST` （本地测试设置 `127.0.0.1`），`SERVER_PORT` （与服务端端口号一致）
2. 在项目文件夹下运行（Windows 需要管理员权限）
```bash
python src/client.py
```