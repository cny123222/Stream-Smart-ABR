# GitHub 项目协作 🛠️

## 第零步：首次设置

1.  **克隆仓库到本地：**
    打开 Git Bash 或终端，导航到你希望存放项目的文件夹。
    ```bash
    git clone https://github.com/cny123222/Stream-Smart-ABR.git
    ```

2.  **进入项目目录：**
    ```bash
    cd Stream-Smart-ABR
    ```

## 阶段一：开始新任务/开发新功能

1.  **确保你在 `main` 分支，并拉取最新代码：**
    （每次开始新任务前都这样做，以获取团队成员的最新更改）
    ```bash
    git checkout main
    git pull origin main
    ```

2.  **创建并切换到一个新的特性分支 (Feature Branch)：**
    给你的分支起一个描述性的名字（通常用英文），例如 `feature/user-login` 或 `fix/text-display-bug`。
    ```bash
    git checkout -b [你的特性分支名称]
    # 例如: git checkout -b feature/server-logging
    ```
    *你现在就在你的新分支上了，可以安全地进行修改，不会影响 `main` 分支。*

## 阶段二：在特性分支上进行开发

1.  **编写/修改代码：**
    在你的代码编辑器中进行必要的更改、添加新文件等。

2.  **查看更改状态 (可选)：**
    ```bash
    git status
    ```
    *这将显示哪些文件被修改、添加或删除了。*

3.  **将你的更改添加到暂存区：**
    ```bash
    git add .  # 添加所有更改过的文件到暂存区
    # 或者指定文件: git add [文件名1] [文件名2]
    ```

4.  **提交你的更改到本地分支：**
    编写一条清晰的提交信息，概括你所做的更改。
    ```bash
    git commit -m "这里写你的提交信息，例如：实现了用户登录功能"
    ```
    *你可以多次执行步骤 1、3、4，进行多次小的提交。*

5.  **将你的特性分支推送到 GitHub 远程仓库：**
    ```bash
    # 如果是第一次推送这个新创建的分支:
    git push -u origin [你的特性分支名称]
    # 例如: git push -u origin feature/server-logging

    # 如果之前已经推送过这个分支，之后再次推送时:
    git push
    ```
    *这会将你的本地分支及其提交上传到 GitHub，方便备份和后续的 Pull Request。*

## 阶段三：请求合并代码 (创建 Pull Request)

当你的特性分支上的功能开发完毕（或需要他人审查时）：

1.  **确保所有本地提交已推送到远程特性分支** (参考阶段二，步骤5)。

2.  **在 GitHub 上创建“拉取请求”(Pull Request - PR)：**
    * 打开你的浏览器，进入项目的 GitHub 仓库页面。
    * GitHub 通常会自动检测到你新推送的分支，并显示一个 "Compare & pull request" 按钮，点击它。
    * 如果没看到，可以点击仓库的 "Pull requests" 标签页，然后点击 "New pull request" 按钮。
    * **设置分支：**
        * `base` 分支应为 `main` (你希望代码合并到的目标分支)。
        * `compare` 分支应为你的特性分支 (例如 `feature/server-logging`)。
    * **填写 PR 信息：** 编写清晰的标题和描述，说明这个 PR 的目的和主要更改。
    * 点击 "Create pull request"。

3.  **代码审查 (Code Review)：**
    * 通知你的组员来审查你的 PR。他们可以在 PR 页面评论你的代码，提出修改建议。
    * 如果需要修改：回到你的**本地特性分支** (例如 `feature/server-logging`)，进行修改，然后重复**阶段二**的步骤 3 (add)、4 (commit)、5 (push)。你的 PR 会自动更新这些新的提交。

## 阶段四：合并与清理

1.  **合并 PR：**
    * 当代码审查通过，并且所有讨论都解决后，通常由项目负责人或指定成员在 GitHub 的 PR 页面点击 "Merge pull request" 按钮，将你的代码合并到 `main` 分支。

2.  **删除已合并的特性分支 (推荐)：**
    * 合并 PR 后，GitHub 通常会提供一个按钮来删除远程的特性分支。点击它。
    * 你也可以在本地删除这个不再需要的分支：
        ```bash
        git checkout main      # 切换回 main 分支
        git pull origin main   # 更新 main 分支
        git branch -d [你的特性分支名称] # 删除本地特性分支
        # 例如: git branch -d feature/server-logging
        ```

## 开始下一个任务？

重复**阶段一**的步骤 1 (`git checkout main`, `git pull origin main`)，然后创建新的特性分支，开始新的开发周期！

**重要提示 - 处理冲突 (Merge Conflicts)：**
如果在 `git pull origin main` (尝试更新 `main` 或将 `main` 合并到特性分支时) 或 GitHub 合并 PR 时出现冲突，Git 会提示你。你需要：
1.  打开 Git 标记为冲突的文件。
2.  手动编辑文件，解决 `<<<<<<<`, `=======`, `>>>>>>>` 标记之间的冲突内容，保留你需要的代码。
3.  解决后，使用 `git add [已解决冲突的文件名]` 将文件标记为已解决。
4.  然后 `git commit` (Git 通常会自动生成一条合并提交信息，你可以直接使用或修改)。
5.  如果是在解决 `pull` 时的冲突，完成后可能需要 `git push`。如果是在特性分支上解决与 `main` 的冲突，解决并提交后，再 `push` 你的特性分支。