# Real-ESRGAN 中文 GUI（可打包 EXE）

这是一个基于 `Real-ESRGAN` 源码封装的中文桌面 GUI 项目，目标是：

- 功能完整：支持图片增强、视频增强、模型下载、日志、进度、任务停止
- 使用方便：可拖拽输入、参数可视化、配置自动记忆
- 易于发布：内置 `PyInstaller` 打包脚本，可直接生成 `exe`

## 功能亮点

- 中文界面，参数命名与 Real-ESRGAN 官方参数一致
- 支持图片文件/文件夹批量增强
- 支持视频增强（可尝试自动合并原音轨）
- 模型权重自动下载，支持手动指定 `.pth`
- 支持 GPU ID、Tile、Denoise、FP32、人脸增强等选项
- 支持一键恢复默认参数

## 项目结构

```text
.
├─ main.py
├─ requirements.txt
├─ scripts/
│  ├─ setup_env.ps1
│  ├─ run_dev.ps1
│  └─ build_exe.ps1
├─ src/
│  └─ realesrgan_gui/
│     ├─ app.py
│     └─ engine.py
└─ third_party/
   └─ Real-ESRGAN-0.3.0/
```

## 1. 环境安装

在 PowerShell 中执行（CPU 版）：

```powershell
.\scripts\setup_env.ps1 -Torch cpu
```

如果你使用 NVIDIA 并希望安装 CUDA 12.1 版：

```powershell
.\scripts\setup_env.ps1 -Torch cu121
```

## 2. 开发运行

```powershell
.\scripts\run_dev.ps1
```

## 3. 打包 EXE（与模型分离）

```powershell
.\scripts\build_exe.ps1
```

生成位置：

```text
dist/RealESRGAN_CN_GUI/RealESRGAN_CN_GUI.exe
```

说明：

- 默认打包不包含模型权重，包更小
- 首次运行时可在 GUI 里点击“下载模型”，或运行脚本提前下载

## 4. 独立下载模型（可选）

下载默认模型：

```powershell
.\scripts\download_models.ps1
```

下载指定模型：

```powershell
.\scripts\download_models.ps1 -Models RealESRGAN_x4plus,realesr-animevideov3
```

下载全部模型：

```powershell
.\scripts\download_models.ps1 -All
```

## 5. 如果你确实想把本地模型一起打包

```powershell
.\scripts\build_exe.ps1 -IncludeLocalWeights
```

## 使用说明

1. 选择“图片增强”或“视频增强”
2. 设置输入路径和输出目录
3. 选择模型与参数（首次可先点“下载模型”）
4. 点击“开始增强”
5. 在右侧查看日志和进度

## 注意事项

- 首次运行会下载模型权重，需要联网
- 视频音轨合并依赖 `ffmpeg`，如果未安装也可输出无音频视频
- 显存不足时，请减小 `Tile` 或关闭 `人脸增强`
- 使用 anime 模型时不建议启用人脸增强

## 许可证

- GUI 封装代码：遵循本仓库许可证（可按需补充）
- Real-ESRGAN 源码：见 [`third_party/Real-ESRGAN-0.3.0/LICENSE`](third_party/Real-ESRGAN-0.3.0/LICENSE)
