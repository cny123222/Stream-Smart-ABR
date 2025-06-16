# STARS 用户手册

## 环境配置步骤

1. **安装 Python**
    - 推荐版本：Python 3.8 及以上
    - 可从 [Python 官网](https://www.python.org/downloads/) 下载并安装

2. **克隆项目代码**
    ```bash
    git clone https://github.com/cny123222/Stream-Smart-ABR.git
    cd Stream-Smart-ABR
    ```

3. **创建虚拟环境（可选）**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Windows: venv\Scripts\activate
    ```

4. **安装依赖库**
    ```bash
    pip install -r requirements.txt
    ```

5. **安装其他依赖（如有）**
    - 请根据 `README.md` 或 `requirements.txt` 补充安装必要的系统依赖。

6. **下载视频资源**
    - 请根据`ffmpeg_tutorial.md`和`preprocess.md`下载并切片必要的视频资源。
---

## 播放器调试说明

1. **启动后端服务**
    ```bash
    python src/server.py
    ```
    - 默认监听端口：`8081`（可在配置文件中修改）
    - **注意**：在 Windows 下，可能需要以管理员身份运行命令提示符（CMD）

2. **启动播放器前端**
    - 首先打开 `player.html`
    ```bash
    python utils/control.py
    ```
    - **注意**：在 Windows 下，可能需要以管理员身份运行命令提示符（CMD）
    - 脚本会自动遍历参数组合，依次调用 [client.py]，并将结果输出到 `test` 目录下。（需手动创建test目录）

3. **参数说明**
   - 每组测试参数包括：
     - ABR 决策逻辑编号（如 1~7）
     - 网络环境编号（如 1~9）
     - 输出路径（自动生成，格式如 `./test/case_1_1`）

4. **调试建议**
    - 使用浏览器开发者工具（F12）查看网络请求和控制台日志
    - 检查播放器与后端的通信是否正常
    - 控制台会输出详细日志并保存于transmission_log.txt，包括每次测试的参数、进程状态和异常信息，便于定位问题。
    - 可通过日志信息追踪每一步的执行情况，定位异常或性能瓶颈。
    - 如需单独调试某一组参数，可修改 control.py 中的 parameter_combinations，只保留需要的组合。
    - 若需调试 client.py 逻辑，可直接运行：python src/client.py <ABR编号> <网络环境编号> <输出路径>
    
    - 如需添加新的 ABR 决策或网络环境，请参考 `docs/test_tutorial.md`。
---

## 常见问题

- **依赖安装失败**：请确认 Python 版本和 pip 已正确安装，必要时升级 pip。
- **端口被占用**：修改配置文件中的端口号，或释放被占用端口。
- **视频无法播放**：检查视频文件路径和后端服务状态。

---

如需进一步帮助，请查阅项目 `README.md` 或提交 issue。