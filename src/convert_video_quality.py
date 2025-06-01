import subprocess
import os

def convert_video(original_video_path, output_dir, qualities):
    """
    将原视频转换为不同清晰度的视频。

    Args:
        original_video_path (str): 原视频文件的路径。
        output_dir (str): 输出不同清晰度视频的目录。
        qualities (dict): 不同清晰度的配置，键为清晰度名称，值为包含分辨率和比特率的字典。
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for quality_name, config in qualities.items():
        resolution = config['resolution']
        bitrate = config['bitrate']
        output_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(original_video_path))[0]}_{quality_name}.mp4")

        cmd = [
            'ffmpeg',
            '-i', original_video_path,
            '-vf', f'scale={resolution}',
            '-b:v', bitrate,
            '-c:a', 'copy',
            output_path
        ]

        try:
            subprocess.run(cmd, check=True)
            print(f"成功将视频转换为 {quality_name} 清晰度，保存路径: {output_path}")
        except subprocess.CalledProcessError as e:
            print(f"转换为 {quality_name} 清晰度时出错: {e}")

if __name__ == '__main__':
    # 原视频文件路径
    original_video_path = "bbb_sunflower.mp4"
    # 输出目录
    output_dir = "output_videos"
    # 不同清晰度配置，添加 2160p 清晰度
    qualities = {
        "360p": {"resolution": "640x360", "bitrate": "800k"},
        "480p": {"resolution": "854x480", "bitrate": "1500k"},
        "720p": {"resolution": "1280x720", "bitrate": "4000k"},
        "1080p": {"resolution": "1920x1080", "bitrate": "8000k"},
        "2160p": {"resolution": "3840x2160", "bitrate": "16000k"}  # 添加 2160p 配置
    }

    convert_video(original_video_path, output_dir, qualities)
