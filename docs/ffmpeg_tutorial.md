# FFmpeg 安装指南 🎬

FFmpeg 是一个强大的音视频处理工具，我们的项目会用它来切分视频。请确保在你的 Windows 或 macOS 电脑上正确安装它。

## Windows 系统安装 FFmpeg

Windows 上安装 FFmpeg 主要通过下载预编译好的文件并将其路径添加到系统环境变量中。

1.  **下载 FFmpeg:**

      * 访问 FFmpeg 官方下载页面：[https://ffmpeg.org/download.html](https://ffmpeg.org/download.html)
      * 在 "Get packages & executable files" 部分，找到 Windows  Builds 的图标 🪟。
      * 通常推荐选择 `gyan.dev` 或 `BtbN` 的版本。点击进入，下载最新 **release build** (非 nightly 或 dev) 的 `.zip` 或 `.7z` 压缩包 (例如 `ffmpeg-release-full.7z` 或类似的)。

2.  **解压 FFmpeg:**

      * 下载完成后，将压缩包解压到一个你喜欢的位置。例如，解压到 `C:\ffmpeg`。
      * 解压后，你会看到一个包含 `bin`, `doc`, `presets` 等文件夹的目录。我们需要的是 `bin` 文件夹里的 `ffmpeg.exe`, `ffprobe.exe`, `ffplay.exe`。

3.  **配置环境变量:**

      * **目标：** 让系统可以在任何路径下找到 `ffmpeg.exe`。
      * 在 Windows 搜索框中搜索 "**环境变量**" 并选择 "**编辑系统环境变量**"。
      * 在弹出的“系统属性”对话框中，点击右下角的 "**环境变量(N)...**" 按钮。
      * 在“环境变量”对话框中，找到下方的 "**系统变量(S)**" 列表，找到名为 `Path` 的变量，选中它，然后点击 "**编辑(E)...**"。
      * 在“编辑环境变量”对话框中，点击 "**新建(N)**"，然后输入你 FFmpeg `bin` 文件夹的完整路径。例如，如果你解压到了 `C:\ffmpeg`，那么这里就输入 `C:\ffmpeg\bin`。
         *(这是一个示意，具体界面可能略有不同)*
      * 点击 "**确定**" 保存所有打开的对话框。

4.  **验证安装:**

      * **重新打开**一个新的命令提示符 (CMD) 或 PowerShell 窗口 (很重要，旧窗口不会加载新的环境变量)。
      * 输入以下命令并回车：
        ```bash
        ffmpeg -version
        ```
      * 如果你看到 FFmpeg 的版本信息输出，说明安装成功！🎉


## macOS 系统安装 FFmpeg

macOS 上安装 FFmpeg 最简单的方法是使用 [Homebrew](https://brew.sh/) 包管理器。

1.  **安装 Homebrew (如果尚未安装):**

      * 打开“终端”应用程序 (Terminal，可以在“应用程序” -\> “实用工具”中找到)。
      * 粘贴并运行以下命令来安装 Homebrew (请从 [Homebrew 官网](https://brew.sh/) 复制最新的安装命令，因为它可能会更新)：
        ```bash
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
        ```
      * 按照终端提示完成安装过程。这可能需要输入你的用户密码。

2.  **使用 Homebrew 安装 FFmpeg:**

      * 在终端中，运行以下命令：
        ```bash
        brew install ffmpeg
        ```
      * Homebrew 会自动下载、编译并安装 FFmpeg 及其依赖。等待命令执行完成。

3.  **验证安装:**

      * 在终端中，输入以下命令并回车：
        ```bash
        ffmpeg -version
        ```
      * 如果你看到 FFmpeg 的版本信息输出，说明安装成功！🥳