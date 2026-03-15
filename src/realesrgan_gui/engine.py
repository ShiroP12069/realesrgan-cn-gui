from __future__ import annotations

import shutil
import subprocess
import sys
import threading
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".avi", ".flv", ".wmv", ".m4v", ".ts"}


def project_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[2]


def source_root() -> Path:
    return project_root() / "third_party" / "Real-ESRGAN-0.3.0"


def ensure_source_path() -> None:
    src = source_root()
    if src.exists() and str(src) not in sys.path:
        sys.path.insert(0, str(src))


@dataclass(slots=True)
class InferenceConfig:
    mode: str = "image"
    input_path: str = ""
    output_dir: str = ""
    model_name: str = "RealESRGAN_x4plus"
    outscale: float = 4.0
    denoise_strength: float = 0.5
    model_path: str = ""
    suffix: str = "out"
    tile: int = 0
    tile_pad: int = 10
    pre_pad: int = 0
    face_enhance: bool = False
    fp32: bool = False
    alpha_upsampler: str = "realesrgan"
    ext: str = "auto"
    gpu_id: int | None = None
    fps: float | None = None
    ffmpeg_bin: str = "ffmpeg"


class StopRequested(Exception):
    pass


class RealESRGANEngine:
    def __init__(self, stop_event: threading.Event | None = None) -> None:
        self.stop_event = stop_event or threading.Event()

    def stop(self) -> None:
        self.stop_event.set()

    def run(
        self,
        config: InferenceConfig,
        log_cb: Callable[[str], None],
        progress_cb: Callable[[int, int], None],
    ) -> str:
        self._validate_input(config)
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        log_cb("正在初始化模型...")
        upsampler, face_enhancer = self._build_upsampler(config, log_cb)
        if config.mode == "video":
            count = self._run_video(config, upsampler, face_enhancer, log_cb, progress_cb)
            return f"任务完成：视频已处理，共 {count} 帧。"
        count = self._run_images(config, upsampler, face_enhancer, log_cb, progress_cb)
        return f"任务完成：图片处理 {count} 张。"

    def download_model(self, model_name: str, custom_model_path: str, log_cb: Callable[[str], None]) -> str:
        cfg = InferenceConfig(model_name=model_name, model_path=custom_model_path)
        self._resolve_model(cfg, log_cb)
        return "模型已就绪。"

    def _validate_input(self, config: InferenceConfig) -> None:
        input_path = Path(config.input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"输入路径不存在：{input_path}")
        if config.mode == "image":
            if input_path.is_file() and input_path.suffix.lower() not in IMAGE_EXTS:
                raise ValueError("图片模式只支持常见图片格式文件。")
        if config.mode == "video":
            if not input_path.is_file() or input_path.suffix.lower() not in VIDEO_EXTS:
                raise ValueError("视频模式请选择单个视频文件。")

    def _check_stop(self) -> None:
        if self.stop_event.is_set():
            raise StopRequested("任务已停止")

    @staticmethod
    def _lazy_import_modules():
        ensure_source_path()
        # basicsr 1.4.x 依赖旧路径 torchvision.transforms.functional_tensor。
        # 新版 torchvision 将其重命名为 _functional_tensor，这里做兼容映射。
        try:
            import torchvision.transforms.functional_tensor  # type: ignore # noqa: F401
        except Exception:
            try:
                import torchvision.transforms._functional_tensor as _ft  # type: ignore

                shim = types.ModuleType("torchvision.transforms.functional_tensor")
                shim.__dict__.update(_ft.__dict__)
                sys.modules["torchvision.transforms.functional_tensor"] = shim
            except Exception:
                pass
        try:
            from basicsr.archs.rrdbnet_arch import RRDBNet  # type: ignore
            from basicsr.utils.download_util import load_file_from_url  # type: ignore
            from realesrgan import RealESRGANer  # type: ignore
            from realesrgan.archs.srvgg_arch import SRVGGNetCompact  # type: ignore
        except Exception as exc:  # pragma: no cover - env dependent
            raise RuntimeError(
                "缺少依赖。请先运行 scripts/setup_env.ps1 安装环境。"
            ) from exc
        return RRDBNet, SRVGGNetCompact, RealESRGANer, load_file_from_url

    def _model_info(self, model_name: str):
        RRDBNet, SRVGGNetCompact, _, _ = self._lazy_import_modules()
        if model_name == "RealESRGAN_x4plus":
            return RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4), 4, [
                "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
            ]
        if model_name == "RealESRNet_x4plus":
            return RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4), 4, [
                "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth"
            ]
        if model_name == "RealESRGAN_x4plus_anime_6B":
            return RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4), 4, [
                "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth"
            ]
        if model_name == "RealESRGAN_x2plus":
            return RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2), 2, [
                "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth"
            ]
        if model_name == "realesr-animevideov3":
            return SRVGGNetCompact(
                num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type="prelu"
            ), 4, ["https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth"]
        if model_name == "realesr-general-x4v3":
            return SRVGGNetCompact(
                num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type="prelu"
            ), 4, [
                "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth",
                "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth",
            ]
        raise ValueError(f"不支持的模型：{model_name}")

    def _default_weights_dir(self) -> Path:
        if getattr(sys, "frozen", False):
            folder = project_root() / "weights"
        else:
            folder = source_root() / "weights"
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    def _resolve_model(self, config: InferenceConfig, log_cb: Callable[[str], None]):
        _, _, _, load_file_from_url = self._lazy_import_modules()
        _, _, urls = self._model_info(config.model_name)
        if config.model_path.strip():
            path = Path(config.model_path.strip())
            if not path.exists():
                raise FileNotFoundError(f"模型文件不存在：{path}")
            model_path: str | list[str] = str(path)
        else:
            model_dir = self._default_weights_dir()
            local = model_dir / f"{config.model_name}.pth"
            if not local.exists():
                for url in urls:
                    log_cb(f"下载模型：{url}")
                    downloaded = load_file_from_url(url=url, model_dir=str(model_dir), progress=True, file_name=None)
                    local = Path(downloaded)
            model_path = str(local)

        dni_weight = None
        if config.model_name == "realesr-general-x4v3" and abs(config.denoise_strength - 1.0) > 1e-6:
            wdn_model = str(model_path).replace("realesr-general-x4v3", "realesr-general-wdn-x4v3")
            if not Path(wdn_model).exists():
                model_dir = self._default_weights_dir()
                wdn_url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth"
                log_cb(f"下载模型：{wdn_url}")
                load_file_from_url(url=wdn_url, model_dir=str(model_dir), progress=True, file_name=None)
            model_path = [str(model_path), wdn_model]
            dni_weight = [config.denoise_strength, 1 - config.denoise_strength]
        return model_path, dni_weight

    def _build_upsampler(self, config: InferenceConfig, log_cb: Callable[[str], None]):
        _, _, RealESRGANer, _ = self._lazy_import_modules()
        model, netscale, _ = self._model_info(config.model_name)
        model_path, dni_weight = self._resolve_model(config, log_cb)
        upsampler = RealESRGANer(
            scale=netscale,
            model_path=model_path,
            dni_weight=dni_weight,
            model=model,
            tile=config.tile,
            tile_pad=config.tile_pad,
            pre_pad=config.pre_pad,
            half=not config.fp32,
            gpu_id=config.gpu_id,
        )
        face_enhancer = None
        if config.face_enhance:
            try:
                from gfpgan import GFPGANer  # type: ignore
            except Exception as exc:
                raise RuntimeError("人脸增强依赖加载失败，请确认已安装 gfpgan。") from exc
            face_enhancer = GFPGANer(
                model_path="https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth",
                upscale=config.outscale,
                arch="clean",
                channel_multiplier=2,
                bg_upsampler=upsampler,
            )
        return upsampler, face_enhancer

    def _collect_images(self, input_path: str) -> list[Path]:
        p = Path(input_path)
        if p.is_file():
            return [p]
        if p.is_dir():
            return sorted([x for x in p.iterdir() if x.is_file() and x.suffix.lower() in IMAGE_EXTS])
        return []

    @staticmethod
    def _build_output_path(config: InferenceConfig, src: Path, rgba_mode: bool) -> Path:
        extension = src.suffix.lower().replace(".", "")
        if config.ext != "auto":
            extension = config.ext
        if rgba_mode:
            extension = "png"
        suffix = f"_{config.suffix}" if config.suffix else ""
        return Path(config.output_dir) / f"{src.stem}{suffix}.{extension}"

    def _run_images(self, config: InferenceConfig, upsampler, face_enhancer, log_cb, progress_cb) -> int:
        images = self._collect_images(config.input_path)
        if not images:
            raise ValueError("未找到可处理的图片。")
        total = len(images)
        progress_cb(0, total)
        processed = 0
        for idx, img_path in enumerate(images, start=1):
            self._check_stop()
            img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
            if img is None:
                log_cb(f"跳过：无法读取 {img_path.name}")
                progress_cb(idx, total)
                continue
            rgba_mode = len(img.shape) == 3 and img.shape[2] == 4
            try:
                if face_enhancer is not None:
                    _, _, output = face_enhancer.enhance(
                        img, has_aligned=False, only_center_face=False, paste_back=True
                    )
                else:
                    output, _ = upsampler.enhance(
                        img, outscale=config.outscale, alpha_upsampler=config.alpha_upsampler
                    )
            except RuntimeError as error:
                log_cb(f"失败：{img_path.name} -> {error}")
            else:
                save_path = self._build_output_path(config, img_path, rgba_mode)
                cv2.imwrite(str(save_path), output)
                processed += 1
                log_cb(f"完成：{img_path.name}")
            progress_cb(idx, total)
        return processed

    def _run_video(self, config: InferenceConfig, upsampler, face_enhancer, log_cb, progress_cb) -> int:
        input_video = Path(config.input_path)
        cap = cv2.VideoCapture(str(input_video))
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频：{input_video}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        src_fps = cap.get(cv2.CAP_PROP_FPS)
        fps = config.fps if config.fps else (src_fps if src_fps and src_fps > 0 else 24.0)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        total = total if total > 0 else 0
        out_w = max(1, int(width * config.outscale))
        out_h = max(1, int(height * config.outscale))

        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        name = input_video.stem
        suffix = f"_{config.suffix}" if config.suffix else ""
        final_output = output_dir / f"{name}{suffix}.mp4"
        temp_output = output_dir / f"{name}{suffix}_temp_noaudio.mp4"

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(temp_output), fourcc, fps, (out_w, out_h))
        if not writer.isOpened():
            cap.release()
            raise RuntimeError("视频写入器初始化失败，请检查输出目录权限。")

        processed = 0
        progress_cb(0, total)
        try:
            while True:
                self._check_stop()
                ret, frame = cap.read()
                if not ret:
                    break
                try:
                    if face_enhancer is not None:
                        _, _, output = face_enhancer.enhance(
                            frame, has_aligned=False, only_center_face=False, paste_back=True
                        )
                    else:
                        output, _ = upsampler.enhance(
                            frame, outscale=config.outscale, alpha_upsampler=config.alpha_upsampler
                        )
                except RuntimeError as error:
                    log_cb(f"第 {processed + 1} 帧处理失败：{error}")
                    continue
                writer.write(output)
                processed += 1
                progress_cb(processed, total)
        except StopRequested:
            if temp_output.exists():
                temp_output.unlink(missing_ok=True)
            raise
        finally:
            cap.release()
            writer.release()

        self._merge_audio(config.ffmpeg_bin, input_video, temp_output, final_output, log_cb)
        return processed

    @staticmethod
    def _merge_audio(ffmpeg_bin: str, source_video: Path, temp_video: Path, final_video: Path, log_cb) -> None:
        cmd = [
            ffmpeg_bin,
            "-y",
            "-i",
            str(temp_video),
            "-i",
            str(source_video),
            "-map",
            "0:v:0",
            "-map",
            "1:a?",
            "-c:v",
            "copy",
            "-c:a",
            "copy",
            str(final_video),
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        except FileNotFoundError:
            log_cb("未找到 ffmpeg，输出将不包含原音轨。")
            if final_video.exists():
                final_video.unlink(missing_ok=True)
            shutil.move(str(temp_video), str(final_video))
            return
        if result.returncode == 0:
            if temp_video.exists():
                temp_video.unlink()
            log_cb("视频处理完成，音轨已合并。")
            return
        log_cb("ffmpeg 合并音轨失败，已保留无音频视频。")
        log_cb(result.stderr.strip() or result.stdout.strip() or "未知错误。")
        if final_video.exists():
            final_video.unlink(missing_ok=True)
        shutil.move(str(temp_video), str(final_video))
