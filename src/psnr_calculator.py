import sys
import os

cmd = '''
ffmpeg -i D:\\Desktop\\BigWork\\Stream-Smart-ABR\\bbb_sunflower.mp4 -i {my_dir} -lavfi "[1:v]scale=w=3840:h=2160[rescaled];[0:v][rescaled]psnr" -f null -
'''
folder_path = 'output_videos'

for root, dirs, files in os.walk(folder_path):
    for file in files:
        # 拼接文件的完整路径
        file_path = os.path.join(root, file)
        print(file_path)
        os.system(cmd.replace('{my_dir}', file_path))
        print('-------------------')
