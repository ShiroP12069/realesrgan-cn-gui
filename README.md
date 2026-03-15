# Real-ESRGAN 超分工具（中文 GUI）

> 本仓库是 **GUI 封装项目**。  
> Real-ESRGAN 的核心算法、模型结构和原始推理能力来自上游仓库 [xinntao/Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)。

一个面向 Windows 用户的 Real-ESRGAN 图形化工具，目标是“拿来就能用”：  
- 中文界面  
- 图片/视频超分  
- 本地模型自动识别与直接选择  
- 一键打包 EXE

## 项目定位

- 你正在看的仓库：GUI、参数交互、模型管理、打包脚本
- 上游仓库：超分算法、模型架构、底层推理实现
- 适合人群：不想频繁敲命令、需要可视化操作和快速分发的人

## 主要功能

- 图片超分：单图或文件夹批处理
- 视频超分：支持帧率设置，尝试保留原音轨
- 模型管理：自动扫描本地 `.pth`，可直接下拉选择
- 参数可视化：放大倍率、降噪、Tile、FP32、GPU ID 等
- 任务控制：进度条、实时日志、停止任务
- 配置记忆：自动保存上次设置

## 快速开始

### 1) 安装环境

CPU 版：

```powershell
.\scripts\setup_env.ps1 -Torch cpu
```

NVIDIA（CUDA 12.1）：

```powershell
.\scripts\setup_env.ps1 -Torch cu121
```

### 2) 启动 GUI

```powershell
.\scripts\run_dev.ps1
```

### 3) 常规使用流程

1. 选择 `图片超分` 或 `视频超分`
2. 选择输入和输出路径
3. 选择模型（可用本地模型下拉框直接选）
4. 点击 `开始超分`
5. 在右侧查看进度与日志

## EXE 打包与模型分离

默认打包不带权重（体积更小）：

```powershell
.\scripts\build_exe.ps1
```

输出：

```text
dist/RealESRGAN_CN_GUI/RealESRGAN_CN_GUI.exe
```

首次使用可在 GUI 内下载模型，或提前下载：

```powershell
.\scripts\download_models.ps1
```

下载全部模型：

```powershell
.\scripts\download_models.ps1 -All
```

如果你确实要把本地权重一起打包：

```powershell
.\scripts\build_exe.ps1 -IncludeLocalWeights
```

## 仓库结构

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
│     ├─ app.py
│     └─ engine.py
└─ third_party/
   └─ Real-ESRGAN-0.3.0/
```

## 常见问题

### Q1: 程序提示找不到模型怎么办？

- 先点“识别”扫描模型目录
- 或点“下载模型”自动下载
- 或手动将 `.pth` 放到 `weights/` 目录

### Q2: 显存不足怎么办？

- 减小 `Tile`（或尝试不同 Tile 值）
- 关闭人脸修复
- 选择更轻量模型

### Q3: 视频没有音频怎么办？

- 检查是否安装并配置 `ffmpeg`
- 未配置时会退化为输出无音频视频

## 上游与许可证

- 上游项目：`xinntao/Real-ESRGAN`  
  https://github.com/xinntao/Real-ESRGAN
- 上游许可证：[`third_party/Real-ESRGAN-0.3.0/LICENSE`](third_party/Real-ESRGAN-0.3.0/LICENSE)
- 第三方声明：[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)
