import subprocess
import time
import os
import ctypes

# 定义不同的参数组合
parameter_combinations = [
    # e.g. [3,1] # [决策, 网络环境]
    [2, 8]
]

for i in range(1, 4):
    for j in range(1, 9):
        qwq = [i, j]
        parameter_combinations.append(qwq)

for params in parameter_combinations:
    # 构建命令
    command = [
        'python',
        './src/client.py',
        str(params[0]),
        str(params[1]),
        f'./Test/case_{params[0]}_{params[1]}'
    ]
    try:
        # 启动子进程
        process = subprocess.Popen(command)
        print(f"Started client.py with parameters {params}.")
        # 等待 3 分钟（180 秒）
        time.sleep(180)
        if os.name == 'nt':  # Windows 系统
            # 加载 kernel32.dll
            kernel32 = ctypes.windll.kernel32
            # 打开进程
            handle = kernel32.OpenProcess(1, False, process.pid)
            # 终止进程
            kernel32.TerminateProcess(handle, 1)
            # 关闭句柄
            kernel32.CloseHandle(handle)
        else:  # Linux 和 macOS 系统
            process.terminate()
            process.wait(timeout=10)  # 等待进程结束，最多等待 10 秒
        print(f"Terminated client.py with parameters {params}.")
        # 等待子进程结束
        process.wait()
    except Exception as e:
        print(f"An error occurred while running client.py with parameters {params}: {e}")