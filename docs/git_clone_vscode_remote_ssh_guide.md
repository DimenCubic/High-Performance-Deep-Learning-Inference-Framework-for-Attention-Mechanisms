# Git Clone 项目并使用 VS Code 连接远程服务器

## 1. 在服务器上克隆 GitHub 项目

先通过 SSH 登录服务器。

进入项目存放目录：

```bash
cd /root
```

克隆 GitHub 仓库：

```bash
git clone https://github.com/<你的用户名>/<你的仓库名>.git
```

例如：

```bash
git clone https://github.com/USERNAME/High-Performance-Deep-Learning-Inference-Framework-for-Attention-Mechanisms.git
```

进入项目目录：

```bash
cd High-Performance-Deep-Learning-Inference-Framework-for-Attention-Mechanisms
```

查看项目文件：

```bash
ls
```

检查 Git 状态：

```bash
git status
```

正常情况下会看到类似：

```text
On branch main
Your branch is up to date with 'origin/main'.

nothing to commit, working tree clean
```

如果仓库是 Private Repository，HTTPS 方式可能需要 GitHub Personal Access Token。

也可以使用 SSH：

```bash
git clone git@github.com:<你的用户名>/<你的仓库名>.git
```

服务器上常用 Git 命令：

```bash
git pull
git status
git add .
git commit -m "Update project"
git push
```

---

## 2. 在 Mac 上安装 VS Code Remote - SSH

打开 VS Code。

进入：

```text
Extensions
```

搜索：

```text
Remote - SSH
```

安装 Microsoft 官方的：

```text
Remote - SSH
```

---

## 3. 获取服务器 SSH 命令

在云服务器控制台找到 SSH 登录信息。

格式通常类似：

```bash
ssh -p <端口> root@<服务器地址>
```

例如：

```bash
ssh -p 12345 root@example.server.com
```

实际使用时，应复制服务器平台提供的真实 SSH 命令。

---

## 4. 将服务器添加到 VS Code

在 VS Code 中按：

```text
Cmd + Shift + P
```

搜索：

```text
Remote-SSH: Add New SSH Host...
```

选择后，粘贴完整 SSH 命令：

```bash
ssh -p <端口> root@<服务器地址>
```

例如：

```bash
ssh -p 12345 root@example.server.com
```

然后选择保存到：

```text
~/.ssh/config
```

---

## 5. 使用 VS Code 连接服务器

再次按：

```text
Cmd + Shift + P
```

搜索：

```text
Remote-SSH: Connect to Host...
```

选择刚刚添加的服务器。

如果第一次连接时询问服务器系统类型：

```text
Select the platform of the remote host
```

选择：

```text
Linux
```

如果要求输入密码，输入服务器平台提供的 SSH 密码。

连接成功后，VS Code 左下角通常会显示：

```text
SSH: <服务器名称>
```

这说明 VS Code 已经进入远程开发模式。

---

## 6. 在 VS Code 中打开服务器项目

连接服务器成功后，点击：

```text
File
→ Open Folder
```

打开服务器上的项目目录：

```text
/root/High-Performance-Deep-Learning-Inference-Framework-for-Attention-Mechanisms
```

点击：

```text
OK
```

之后 VS Code Explorer 中看到的：

```text
CMakeLists.txt
include/
src/
tests/
models/
benchmark/
docs/
```

都是服务器上的文件，而不是 Mac 本地文件。

---

## 7. 使用 VS Code 远程终端

在 VS Code 中打开：

```text
Terminal
→ New Terminal
```

终端应该显示类似：

```bash
root@server:~/High-Performance-Deep-Learning-Inference-Framework-for-Attention-Mechanisms#
```

这说明当前 Terminal 运行在远程服务器。

此时运行的命令全部发生在服务器上。

例如：

```bash
git status
```

查看服务器项目的 Git 状态。

```bash
nvidia-smi
```

查看服务器 GPU。

```bash
nvcc --version
```

使用服务器 CUDA 编译器。

```bash
cmake -S . -B build
cmake --build build
```

在服务器上编译项目。

---

## 8. 推荐的日常开发流程

以后开发时可以直接使用：

```text
启动云服务器
↓
VS Code
↓
Remote-SSH: Connect to Host
↓
打开服务器项目目录
↓
直接修改服务器上的代码
↓
使用 VS Code Terminal 编译和运行
↓
git commit
↓
git push
```

不需要：

```text
Mac 修改文件
↓
手动上传到服务器
↓
服务器测试
↓
再下载回来
```

推荐直接通过：

```text
Mac VS Code
↓
Remote-SSH
↓
Linux Server
↓
Git Repository
```

进行远程开发。

---

## 9. 开发完成后同步到 GitHub

在 VS Code 的远程 Terminal 中：

```bash
git status
git add .
git commit -m "Update project"
git push
```

这样服务器上的最新代码就会同步到 GitHub。

下次如果重新创建一个服务器实例，只需要重新：

```bash
cd /root
git clone https://github.com/<你的用户名>/<你的仓库名>.git
```

然后再使用 VS Code Remote-SSH 连接该服务器即可。
