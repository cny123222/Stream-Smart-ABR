# 测试环境配置

## 服务端配置

### 测试视频下载

- **Big Buck Bunny** (大雄兔)是一个非常经典的测试动画片
- 资源来自 Blender Foundation 的开放电影
- 我们使用的是 **1080p_30fps** 的版本作为高质量母片，大小约 **276.1MB**
- [点击此处下载](https://download.blender.org/demo/movies/BBB/bbb_sunflower_1080p_30fps_normal.mp4.zip)
- 下载后，视频文件放在项目的根文件夹下，并重命名为 `bbb_sunflower.mp4`

### 测试视频分片及转码

1. 按照 `tutorial/ffmpeg_tutorial` 安装 FFmpeg
2. 命令行运行
```bash
python src/segment_video.py
```
该过程可能需要十几分钟，等待的时候可以看看这个电影（bushi）

## 客户端

### 安装 vlc 库
```bash
pip install python-vlc
```