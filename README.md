# Real-ESRGAN 中文图形工具（GUI）

先说明白：这个仓库是 **GUI 工具**，不是超分算法原仓库。  
核心算法来自上游项目 [xinntao/Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)。

## 这个项目是干什么的

把 Real-ESRGAN 做成中文图形界面，方便直接点选使用，不用手敲命令。

你可以用它做这些事：

- 图片超分（单张或整文件夹）
- 视频超分（可尝试保留原音轨）
- 自动识别本地模型文件，并直接选择
- 一键打包 Windows `exe`

## 怎么用（最短流程）

### 1. 安装环境

CPU 版：

```powershell
.\scripts\setup_env.ps1 -Torch cpu
```

NVIDIA（CUDA 12.1）：

```powershell
.\scripts\setup_env.ps1 -Torch cu121
```

### 2. 打开程序

```powershell
.\scripts\run_dev.ps1
```

### 3. 开始处理

1. 选 `图片超分` 或 `视频超分`
2. 选输入路径和输出目录
3. 选模型（可用“本地模型”下拉框直接选）
4. 点 `开始超分`
5. 右侧看进度和日志

## 打包 EXE

默认打包（不带模型，体积更小）：

```powershell
.\scripts\build_exe.ps1
```

生成文件：

```text
dist/RealESRGAN_CN_GUI/RealESRGAN_CN_GUI.exe
```

如果你要把本地模型一起打包：

```powershell
.\scripts\build_exe.ps1 -IncludeLocalWeights
```

## 模型下载

GUI 里可直接点“下载模型”。  
也可以用脚本：

```powershell
.\scripts\download_models.ps1
```

下载全部模型：

```powershell
.\scripts\download_models.ps1 -All
```

## 常见问题

### 1) 提示找不到模型

- 点“识别”重新扫描
- 点“下载模型”自动下载
- 或把 `.pth` 放到 `weights/` 目录

### 2) 显存不够

- 调整 `Tile`
- 关闭人脸修复
- 换更轻量模型

### 3) 视频没声音

- 检查 `ffmpeg` 路径是否正确
- 未配置时会输出无音频视频

## 目录结构

```text
.
├─ main.py
├─ requirements.txt
├─ scripts/
│  ├─ setup_env.ps1
│  ├─ run_dev.ps1
│  ├─ build_exe.ps1
│  └─ download_models.ps1
├─ src/
│  └─ realesrgan_gui/
└─ third_party/
   └─ Real-ESRGAN-0.3.0/
```

## 致谢和许可证

- 上游项目：`xinntao/Real-ESRGAN`  
  https://github.com/xinntao/Real-ESRGAN
- 上游许可证：[`third_party/Real-ESRGAN-0.3.0/LICENSE`](third_party/Real-ESRGAN-0.3.0/LICENSE)
- 第三方说明：[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)
