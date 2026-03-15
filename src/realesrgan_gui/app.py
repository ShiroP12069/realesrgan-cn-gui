from __future__ import annotations

import sys
import threading
import traceback
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QObject, QSettings, Qt, QThread, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QFont, QIcon
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QProgressBar,
    QScrollArea,
    QSplitter,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from .engine import InferenceConfig, RealESRGANEngine, StopRequested, project_root


class DropPathEdit(QLineEdit):
    pathDropped = Signal(str)

    def __init__(self, placeholder: str) -> None:
        super().__init__()
        self.setAcceptDrops(True)
        self.setPlaceholderText(placeholder)

    def dragEnterEvent(self, event):  # noqa: N802
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            return
        super().dragEnterEvent(event)

    def dropEvent(self, event):  # noqa: N802
        urls = event.mimeData().urls()
        if not urls:
            super().dropEvent(event)
            return
        path = urls[0].toLocalFile()
        if path:
            self.setText(path)
            self.pathDropped.emit(path)
            event.acceptProposedAction()
            return
        super().dropEvent(event)


class InferenceWorker(QObject):
    log = Signal(str)
    progress = Signal(int, int)
    finished = Signal(bool, str)

    def __init__(self, config: InferenceConfig) -> None:
        super().__init__()
        self.config = config
        self.stop_event = threading.Event()

    def run(self) -> None:
        engine = RealESRGANEngine(self.stop_event)
        try:
            message = engine.run(self.config, self.log.emit, self.progress.emit)
        except StopRequested:
            self.finished.emit(False, "任务已停止。")
            return
        except Exception as exc:
            self.log.emit(traceback.format_exc())
            self.finished.emit(False, f"任务失败：{exc}")
            return
        self.finished.emit(True, message)

    def stop(self) -> None:
        self.stop_event.set()


class ModelDownloadWorker(QObject):
    log = Signal(str)
    finished = Signal(bool, str)

    def __init__(self, model_name: str, model_path: str) -> None:
        super().__init__()
        self.model_name = model_name
        self.model_path = model_path

    def run(self) -> None:
        engine = RealESRGANEngine()
        try:
            msg = engine.download_model(self.model_name, self.model_path, self.log.emit)
        except Exception as exc:
            self.log.emit(traceback.format_exc())
            self.finished.emit(False, f"模型准备失败：{exc}")
            return
        self.finished.emit(True, msg)


class MainWindow(QMainWindow):
    SETTINGS_KEYS = [
        "mode",
        "input_path",
        "output_dir",
        "model_name",
        "model_path",
        "outscale",
        "denoise_strength",
        "suffix",
        "tile",
        "tile_pad",
        "pre_pad",
        "face_enhance",
        "fp32",
        "alpha_upsampler",
        "ext",
        "gpu_id",
        "fps",
        "ffmpeg_bin",
    ]

    MODE_LABELS = {"图片超分": "image", "视频超分": "video"}
    MODE_VALUES = {v: k for k, v in MODE_LABELS.items()}

    MODELS = [
        "RealESRGAN_x4plus",
        "RealESRNet_x4plus",
        "RealESRGAN_x4plus_anime_6B",
        "RealESRGAN_x2plus",
        "realesr-animevideov3",
        "realesr-general-x4v3",
    ]

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Real-ESRGAN超分工具")
        self.resize(1180, 760)
        self.setMinimumSize(1024, 680)
        self.settings = QSettings("Codex", "RealESRGAN-CN-GUI")
        self.detected_model_paths: dict[str, str] = {}
        self.worker_thread: QThread | None = None
        self.worker: InferenceWorker | None = None
        self.download_thread: QThread | None = None
        self.download_worker: ModelDownloadWorker | None = None
        self._build_ui()
        self._init_tooltips()
        self._apply_stylesheet()
        self._load_settings()
        self._auto_detect_models()
        self._update_mode_ui()

    def _build_ui(self) -> None:
        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(12)

        header = QFrame()
        header.setObjectName("headerCard")
        header_layout = QVBoxLayout(header)
        header_layout.setContentsMargins(20, 14, 20, 14)
        title = QLabel("Real-ESRGAN超分工具")
        title.setObjectName("titleLabel")
        title.setWordWrap(True)
        header_layout.addWidget(title)
        root.addWidget(header)

        left_panel = self._build_left_panel()
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setFrameShape(QFrame.NoFrame)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        left_scroll.setWidget(left_panel)
        right_panel = self._build_right_panel()

        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(8)
        splitter.addWidget(left_scroll)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 5)
        splitter.setStretchFactor(1, 6)
        splitter.setSizes([620, 760])
        root.addWidget(splitter, 1)

        self.setCentralWidget(central)

    def _build_left_panel(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("card")
        panel.setMinimumWidth(520)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        basic_group = QGroupBox("任务设置")
        basic_form = QFormLayout(basic_group)
        self._configure_form_layout(basic_form)

        self.mode_combo = QComboBox()
        self.mode_combo.addItems(list(self.MODE_LABELS.keys()))
        self.mode_combo.currentTextChanged.connect(self._update_mode_ui)
        basic_form.addRow("模式", self.mode_combo)

        input_row = QWidget()
        input_layout = QHBoxLayout(input_row)
        input_layout.setContentsMargins(0, 0, 0, 0)
        input_layout.setSpacing(8)
        self.input_edit = DropPathEdit("可拖拽文件或文件夹到这里")
        btn_input = QPushButton("浏览")
        btn_input.clicked.connect(self._browse_input)
        btn_input.setToolTip("选择输入文件或目录。")
        input_layout.addWidget(self.input_edit, 1)
        input_layout.addWidget(btn_input)
        basic_form.addRow("输入路径", input_row)

        output_row = QWidget()
        output_layout = QHBoxLayout(output_row)
        output_layout.setContentsMargins(0, 0, 0, 0)
        output_layout.setSpacing(8)
        self.output_edit = DropPathEdit("选择输出目录")
        btn_output = QPushButton("浏览")
        btn_output.clicked.connect(self._browse_output)
        btn_open = QPushButton("打开")
        btn_open.clicked.connect(self._open_output_dir)
        btn_output.setToolTip("选择输出目录。")
        btn_open.setToolTip("打开输出目录。")
        output_layout.addWidget(self.output_edit, 1)
        output_layout.addWidget(btn_output)
        output_layout.addWidget(btn_open)
        basic_form.addRow("输出目录", output_row)

        model_group = QGroupBox("模型参数")
        model_form = QFormLayout(model_group)
        self._configure_form_layout(model_form)

        self.model_combo = QComboBox()
        self.model_combo.addItems(self.MODELS)
        self.model_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.model_combo.setMinimumContentsLength(28)
        model_form.addRow("模型", self.model_combo)

        model_path_row = QWidget()
        model_path_layout = QHBoxLayout(model_path_row)
        model_path_layout.setContentsMargins(0, 0, 0, 0)
        model_path_layout.setSpacing(8)
        self.model_path_edit = QLineEdit()
        self.model_path_edit.setPlaceholderText("可选：指定模型文件或模型目录")
        self.model_path_edit.editingFinished.connect(self._auto_detect_models)
        btn_model_path = QPushButton("浏览")
        btn_model_path.clicked.connect(self._browse_model_file)
        btn_model_dir = QPushButton("目录")
        btn_model_dir.clicked.connect(self._browse_model_dir)
        btn_model_path.setToolTip("选择单个模型文件。")
        btn_model_dir.setToolTip("选择模型所在目录。")

        model_path_layout.addWidget(self.model_path_edit, 1)
        model_path_layout.addWidget(btn_model_path)
        model_path_layout.addWidget(btn_model_dir)
        model_form.addRow("模型文件", model_path_row)

        model_action_row = QWidget()
        model_action_layout = QHBoxLayout(model_action_row)
        model_action_layout.setContentsMargins(0, 0, 0, 0)
        model_action_layout.setSpacing(8)
        btn_detect_model = QPushButton("识别")
        btn_detect_model.clicked.connect(self._auto_detect_models)
        self.detect_btn = btn_detect_model
        btn_download_model = QPushButton("下载模型")
        btn_download_model.clicked.connect(self._download_model)
        self.download_btn = btn_download_model
        model_action_layout.addWidget(btn_detect_model)
        model_action_layout.addWidget(btn_download_model)
        model_action_layout.addStretch(1)
        model_form.addRow("模型操作", model_action_row)

        self.detected_model_combo = QComboBox()
        self.detected_model_combo.currentIndexChanged.connect(self._on_detected_model_selected)
        model_form.addRow("本地模型", self.detected_model_combo)

        self.outscale_spin = QDoubleSpinBox()
        self.outscale_spin.setRange(1.0, 8.0)
        self.outscale_spin.setSingleStep(0.1)
        self.outscale_spin.setValue(4.0)
        model_form.addRow("放大倍率", self.outscale_spin)

        self.denoise_spin = QDoubleSpinBox()
        self.denoise_spin.setRange(0.0, 1.0)
        self.denoise_spin.setSingleStep(0.05)
        self.denoise_spin.setValue(0.5)
        self.denoise_spin.setToolTip("仅 realesr-general-x4v3 生效")
        model_form.addRow("降噪强度", self.denoise_spin)

        self.gpu_combo = QComboBox()
        self.gpu_combo.addItems(["自动", "0", "1", "2", "3"])
        model_form.addRow("GPU ID", self.gpu_combo)

        self.face_checkbox = QCheckBox("启用人脸修复（GFPGAN）")
        self.fp32_checkbox = QCheckBox("使用 FP32 精度（显存占用更高）")
        model_form.addRow("", self.face_checkbox)
        model_form.addRow("", self.fp32_checkbox)

        advanced_group = QGroupBox("高级设置")
        advanced_form = QFormLayout(advanced_group)
        self._configure_form_layout(advanced_form)

        self.suffix_edit = QLineEdit("out")
        advanced_form.addRow("输出后缀", self.suffix_edit)

        self.ext_combo = QComboBox()
        self.ext_combo.addItems(["auto", "jpg", "png"])
        advanced_form.addRow("输出格式", self.ext_combo)

        self.alpha_combo = QComboBox()
        self.alpha_combo.addItems(["realesrgan", "bicubic"])
        advanced_form.addRow("透明通道处理", self.alpha_combo)

        self.tile_spin = QSpinBox()
        self.tile_spin.setRange(0, 2000)
        self.tile_spin.setValue(0)
        advanced_form.addRow("Tile", self.tile_spin)

        self.tile_pad_spin = QSpinBox()
        self.tile_pad_spin.setRange(0, 200)
        self.tile_pad_spin.setValue(10)
        advanced_form.addRow("Tile Pad", self.tile_pad_spin)

        self.pre_pad_spin = QSpinBox()
        self.pre_pad_spin.setRange(0, 200)
        self.pre_pad_spin.setValue(0)
        advanced_form.addRow("Pre Pad", self.pre_pad_spin)

        video_group = QGroupBox("视频设置")
        video_form = QFormLayout(video_group)
        self._configure_form_layout(video_form)

        self.fps_spin = QDoubleSpinBox()
        self.fps_spin.setRange(0, 240)
        self.fps_spin.setValue(0)
        self.fps_spin.setDecimals(2)
        self.fps_spin.setSingleStep(1.0)
        self.fps_spin.setToolTip("0 表示跟随原视频 FPS")
        video_form.addRow("输出 FPS", self.fps_spin)

        ffmpeg_row = QWidget()
        ffmpeg_layout = QHBoxLayout(ffmpeg_row)
        ffmpeg_layout.setContentsMargins(0, 0, 0, 0)
        ffmpeg_layout.setSpacing(8)
        self.ffmpeg_edit = QLineEdit("ffmpeg")
        btn_ffmpeg = QPushButton("浏览")
        btn_ffmpeg.clicked.connect(self._browse_ffmpeg)
        btn_ffmpeg.setToolTip("选择 ffmpeg 可执行文件。")
        ffmpeg_layout.addWidget(self.ffmpeg_edit, 1)
        ffmpeg_layout.addWidget(btn_ffmpeg)
        video_form.addRow("ffmpeg", ffmpeg_row)

        action_row = QWidget()
        action_layout = QGridLayout(action_row)
        action_layout.setContentsMargins(0, 0, 0, 0)
        action_layout.setHorizontalSpacing(8)
        action_layout.setVerticalSpacing(8)
        self.start_btn = QPushButton("开始超分")
        self.start_btn.setObjectName("primaryBtn")
        self.stop_btn = QPushButton("停止任务")
        self.stop_btn.setObjectName("secondaryBtn")
        self.stop_btn.setEnabled(False)
        self.reset_btn = QPushButton("恢复默认")
        self.reset_btn.setObjectName("secondaryBtn")
        self.start_btn.clicked.connect(self._start_task)
        self.stop_btn.clicked.connect(self._stop_task)
        self.reset_btn.clicked.connect(self._reset_defaults)
        action_layout.addWidget(self.start_btn, 0, 0)
        action_layout.addWidget(self.stop_btn, 0, 1)
        action_layout.addWidget(self.reset_btn, 1, 0, 1, 2)

        layout.addWidget(basic_group)
        layout.addWidget(model_group)
        layout.addWidget(advanced_group)
        layout.addWidget(video_group)
        layout.addWidget(action_row)
        layout.addStretch(1)
        self.video_group = video_group
        return panel

    @staticmethod
    def _configure_form_layout(form: QFormLayout) -> None:
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(10)
        form.setRowWrapPolicy(QFormLayout.WrapLongRows)
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)

    def _build_right_panel(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("card")
        panel.setMinimumWidth(460)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        self.status_label = QLabel("就绪")
        self.status_label.setObjectName("statusLabel")

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("%p%")

        self.log_edit = QTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setPlaceholderText("运行日志会显示在这里...")

        log_label = QLabel("日志")
        log_label.setObjectName("tipLabel")

        layout.addWidget(self.status_label)
        layout.addWidget(self.progress_bar)
        layout.addWidget(log_label)
        layout.addWidget(self.log_edit, 1)
        return panel

    def _init_tooltips(self) -> None:
        self.mode_combo.setToolTip("选择处理类型：图片超分或视频超分。")
        self.input_edit.setToolTip("图片模式支持单图或文件夹；视频模式支持单个视频文件。")
        self.output_edit.setToolTip("超分结果保存目录。")
        self.model_combo.setToolTip("选择模型类型，决定超分风格和速度。")
        self.model_path_edit.setToolTip("可填模型文件或模型目录，留空则自动使用默认目录。")
        self.detected_model_combo.setToolTip("自动识别到的本地模型，可直接选择并应用。")
        self.detect_btn.setToolTip("重新扫描模型目录。")
        self.download_btn.setToolTip("下载当前模型到本地。")
        self.outscale_spin.setToolTip("最终放大倍数，常用 2 或 4。")
        self.denoise_spin.setToolTip("仅对 realesr-general-x4v3 生效，值越大降噪越强。")
        self.gpu_combo.setToolTip("指定 GPU 编号；自动时由程序选择。")
        self.face_checkbox.setToolTip("对人脸区域进行修复，适合人像。")
        self.fp32_checkbox.setToolTip("提高兼容性但会增加显存占用和耗时。")
        self.suffix_edit.setToolTip("输出文件名后缀，例如 out。")
        self.ext_combo.setToolTip("图片输出格式。auto 表示跟随原图格式。")
        self.alpha_combo.setToolTip("透明通道处理方式。")
        self.tile_spin.setToolTip("分块超分，显存不足时调大/调小可避免爆显存。0 为不分块。")
        self.tile_pad_spin.setToolTip("分块边缘填充，减少接缝。")
        self.pre_pad_spin.setToolTip("输入边界预填充，通常保持 0。")
        self.fps_spin.setToolTip("视频输出帧率，0 为跟随原视频。")
        self.ffmpeg_edit.setToolTip("ffmpeg 可执行文件路径，用于音轨合并。")
        self.start_btn.setToolTip("开始执行超分任务。")
        self.stop_btn.setToolTip("停止当前任务。")
        self.reset_btn.setToolTip("将参数恢复为默认值。")
        self.progress_bar.setToolTip("任务进度。")
        self.log_edit.setToolTip("运行日志与错误信息。")

    def _apply_stylesheet(self) -> None:
        self.setFont(QFont("Microsoft YaHei UI", 10))
        self.setStyleSheet(
            """
            QMainWindow {
                background: qlineargradient(
                    x1: 0, y1: 0, x2: 1, y2: 1,
                    stop: 0 #eef4f9,
                    stop: 1 #e8f1f8
                );
            }
            QFrame#headerCard {
                border-radius: 14px;
                background: qlineargradient(
                    x1: 0, y1: 0, x2: 1, y2: 1,
                    stop: 0 #0f3148,
                    stop: 1 #1d6a87
                );
            }
            QLabel#titleLabel {
                color: #ffffff;
                font-size: 22px;
                font-weight: 700;
            }
            QFrame#card {
                background: #fdfefe;
                border: 1px solid #d5e3ee;
                border-radius: 12px;
            }
            QLabel#statusLabel {
                font-size: 15px;
                font-weight: 600;
                color: #20455f;
            }
            QLabel#tipLabel {
                color: #4a6c84;
                font-size: 12px;
            }
            QGroupBox {
                border: 1px solid #d2dfea;
                border-radius: 10px;
                margin-top: 12px;
                padding: 10px 10px 12px 10px;
                font-size: 13px;
                font-weight: 600;
                color: #234862;
                background: #fbfdff;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 6px;
            }
            QLabel, QCheckBox {
                color: #1f3f55;
            }
            QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QTextEdit {
                border: 1px solid #c8d9e8;
                border-radius: 8px;
                background: #ffffff;
                padding: 6px 8px;
                color: #163449;
                min-height: 18px;
            }
            QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus, QTextEdit:focus {
                border: 1px solid #2b7ea8;
            }
            QPushButton {
                border: none;
                border-radius: 8px;
                padding: 7px 14px;
                color: #ffffff;
                background: #2f789f;
                font-weight: 600;
                min-height: 34px;
            }
            QPushButton:hover {
                background: #276887;
            }
            QPushButton:disabled {
                background: #9bb9cb;
            }
            QPushButton#primaryBtn {
                background: #1f8f68;
            }
            QPushButton#primaryBtn:hover {
                background: #187250;
            }
            QPushButton#secondaryBtn {
                background: #5c7f95;
            }
            QPushButton#secondaryBtn:hover {
                background: #4c6a7d;
            }
            QProgressBar {
                border: 1px solid #c7d8e7;
                border-radius: 7px;
                background: #f0f6fb;
                text-align: center;
                color: #1d425a;
                min-height: 24px;
            }
            QProgressBar::chunk {
                border-radius: 6px;
                background: qlineargradient(
                    x1: 0, y1: 0, x2: 1, y2: 0,
                    stop: 0 #2f789f,
                    stop: 1 #4da5c8
                );
            }
            QSplitter::handle {
                background: #d5e3ee;
                border-radius: 3px;
            }
            """
        )

    def _append_log(self, text: str) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_edit.append(f"[{timestamp}] {text}")

    def _browse_input(self) -> None:
        mode = self.MODE_LABELS[self.mode_combo.currentText()]
        if mode == "video":
            path, _ = QFileDialog.getOpenFileName(
                self,
                "选择视频",
                self.input_edit.text() or str(project_root()),
                "视频文件 (*.mp4 *.mkv *.mov *.avi *.flv *.wmv *.m4v *.ts)",
            )
            if path:
                self.input_edit.setText(path)
            return
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择图片文件（或取消后选择文件夹）",
            self.input_edit.text() or str(project_root()),
            "图片文件 (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp)",
        )
        if path:
            self.input_edit.setText(path)
            return
        folder = QFileDialog.getExistingDirectory(self, "选择图片文件夹", self.input_edit.text() or str(project_root()))
        if folder:
            self.input_edit.setText(folder)

    def _browse_output(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "选择输出目录", self.output_edit.text() or str(project_root()))
        if folder:
            self.output_edit.setText(folder)

    def _browse_model_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "选择模型文件", self.model_path_edit.text() or str(project_root()), "模型文件 (*.pth)"
        )
        if path:
            self.model_path_edit.setText(path)
            self._auto_detect_models()

    def _browse_model_dir(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "选择模型目录", self.model_path_edit.text() or str(project_root()))
        if folder:
            self.model_path_edit.setText(folder)
            self._auto_detect_models()

    def _browse_ffmpeg(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "选择 ffmpeg 可执行文件", self.ffmpeg_edit.text() or str(project_root()), "可执行文件 (*.exe)"
        )
        if path:
            self.ffmpeg_edit.setText(path)

    def _open_output_dir(self) -> None:
        out = self.output_edit.text().strip()
        if not out:
            QMessageBox.warning(self, "提示", "请先设置输出目录。")
            return
        path = Path(out)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path.resolve())))

    def _current_mode_value(self) -> str:
        return self.MODE_LABELS[self.mode_combo.currentText()]

    def _update_mode_ui(self) -> None:
        is_video = self._current_mode_value() == "video"
        self.video_group.setEnabled(is_video)
        self.ext_combo.setEnabled(not is_video)

    def _candidate_model_dirs(self) -> list[Path]:
        candidates: list[Path] = []
        manual = self.model_path_edit.text().strip()
        if manual:
            p = Path(manual)
            if p.is_dir():
                candidates.append(p)
            elif p.is_file() and p.suffix.lower() == ".pth":
                candidates.append(p.parent)
        candidates.append(project_root() / "weights")
        candidates.append(project_root() / "third_party" / "Real-ESRGAN-0.3.0" / "weights")

        uniq: list[Path] = []
        seen: set[str] = set()
        for d in candidates:
            key = str(d.resolve()) if d.exists() else str(d)
            if key in seen:
                continue
            seen.add(key)
            if d.exists() and d.is_dir():
                uniq.append(d)
        return uniq

    def _auto_detect_models(self) -> None:
        found: dict[str, str] = {}
        for model_name in self.MODELS:
            for folder in self._candidate_model_dirs():
                candidate = folder / f"{model_name}.pth"
                if candidate.is_file():
                    found[model_name] = str(candidate.resolve())
                    break
        self.detected_model_paths = found
        self._refresh_detected_model_combo()

    def _refresh_detected_model_combo(self) -> None:
        previous_path = self.detected_model_combo.currentData(Qt.UserRole)
        self.detected_model_combo.blockSignals(True)
        self.detected_model_combo.clear()
        self.detected_model_combo.addItem("自动匹配（按模型名）", None)
        self.detected_model_combo.setItemData(0, "", Qt.UserRole)
        self.detected_model_combo.setItemData(0, "", Qt.UserRole + 1)

        index_to_select = 0
        for model_name in self.MODELS:
            path = self.detected_model_paths.get(model_name)
            if not path:
                continue
            label = f"{model_name} ({Path(path).parent.name})"
            self.detected_model_combo.addItem(label)
            idx = self.detected_model_combo.count() - 1
            self.detected_model_combo.setItemData(idx, path, Qt.UserRole)
            self.detected_model_combo.setItemData(idx, model_name, Qt.UserRole + 1)
            self.detected_model_combo.setItemData(idx, path, Qt.ToolTipRole)
            if previous_path and previous_path == path:
                index_to_select = idx

        self.detected_model_combo.setCurrentIndex(index_to_select)
        self.detected_model_combo.setEnabled(self.detected_model_combo.count() > 1)
        if self.detected_model_combo.count() > 1:
            self.detected_model_combo.setToolTip("已识别本地模型，选择后会自动填入路径。")
        else:
            self.detected_model_combo.setToolTip("未识别到本地模型，请选择目录后点击“识别”。")
        self.detected_model_combo.blockSignals(False)

    def _on_detected_model_selected(self, *_args) -> None:
        path = self.detected_model_combo.currentData(Qt.UserRole)
        model_name = self.detected_model_combo.currentData(Qt.UserRole + 1)
        if path:
            self.model_path_edit.setText(str(path))
        if model_name and model_name in self.MODELS:
            self.model_combo.setCurrentText(model_name)

    def _resolved_model_path_for_ui(self, model_name: str) -> str:
        manual = self.model_path_edit.text().strip()
        if manual:
            return manual
        return self.detected_model_paths.get(model_name, "")

    def _collect_config(self) -> InferenceConfig:
        gpu_text = self.gpu_combo.currentText()
        gpu_id = None if gpu_text == "自动" else int(gpu_text)
        fps = self.fps_spin.value()
        model_name = self.model_combo.currentText()
        return InferenceConfig(
            mode=self._current_mode_value(),
            input_path=self.input_edit.text().strip(),
            output_dir=self.output_edit.text().strip(),
            model_name=model_name,
            model_path=self._resolved_model_path_for_ui(model_name),
            outscale=self.outscale_spin.value(),
            denoise_strength=self.denoise_spin.value(),
            suffix=self.suffix_edit.text().strip(),
            tile=self.tile_spin.value(),
            tile_pad=self.tile_pad_spin.value(),
            pre_pad=self.pre_pad_spin.value(),
            face_enhance=self.face_checkbox.isChecked(),
            fp32=self.fp32_checkbox.isChecked(),
            alpha_upsampler=self.alpha_combo.currentText(),
            ext=self.ext_combo.currentText(),
            gpu_id=gpu_id,
            fps=None if fps <= 0 else fps,
            ffmpeg_bin=self.ffmpeg_edit.text().strip() or "ffmpeg",
        )

    def _validate_before_run(self, cfg: InferenceConfig) -> str | None:
        if not cfg.input_path:
            return "请输入输入路径。"
        if not cfg.output_dir:
            return "请选择输出目录。"
        return None

    def _start_task(self) -> None:
        if self.worker_thread is not None:
            QMessageBox.information(self, "提示", "当前已有任务在运行。")
            return
        cfg = self._collect_config()
        error = self._validate_before_run(cfg)
        if error:
            QMessageBox.warning(self, "参数检查", error)
            return
        self._save_settings()
        self._append_log("开始超分任务...")
        self.status_label.setText("运行中")
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)

        self.worker_thread = QThread(self)
        self.worker = InferenceWorker(cfg)
        self.worker.moveToThread(self.worker_thread)
        self.worker_thread.started.connect(self.worker.run)
        self.worker.log.connect(self._append_log)
        self.worker.progress.connect(self._on_progress)
        self.worker.finished.connect(self._on_task_finished)
        self.worker.finished.connect(self.worker_thread.quit)
        self.worker_thread.finished.connect(self._cleanup_worker)
        self.worker_thread.start()

    def _stop_task(self) -> None:
        if self.worker is None:
            return
        self._append_log("收到停止请求，正在等待当前步骤结束...")
        self.worker.stop()
        self.stop_btn.setEnabled(False)
        self.status_label.setText("停止中")

    def _cleanup_worker(self) -> None:
        if self.worker is not None:
            self.worker.deleteLater()
        if self.worker_thread is not None:
            self.worker_thread.deleteLater()
        self.worker = None
        self.worker_thread = None

    def _on_progress(self, current: int, total: int) -> None:
        if total <= 0:
            self.progress_bar.setRange(0, 0)
            return
        if self.progress_bar.minimum() == 0 and self.progress_bar.maximum() == 0:
            self.progress_bar.setRange(0, 100)
        percent = int(min(100, (current / total) * 100))
        self.progress_bar.setValue(percent)
        self.status_label.setText(f"运行中：{current}/{total}")

    def _on_task_finished(self, success: bool, message: str) -> None:
        self._append_log(message)
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_label.setText("完成" if success else "失败/已停止")
        if success:
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(100)
            QMessageBox.information(self, "完成", message)
        else:
            self.progress_bar.setRange(0, 100)
            QMessageBox.warning(self, "结果", message)

    def _download_model(self) -> None:
        if self.download_thread is not None:
            QMessageBox.information(self, "提示", "模型下载任务已在运行。")
            return
        model_name = self.model_combo.currentText()
        model_path = self._resolved_model_path_for_ui(model_name)
        self.download_btn.setEnabled(False)
        self._append_log(f"开始准备模型：{model_name}")
        self.download_thread = QThread(self)
        self.download_worker = ModelDownloadWorker(model_name, model_path)
        self.download_worker.moveToThread(self.download_thread)
        self.download_thread.started.connect(self.download_worker.run)
        self.download_worker.log.connect(self._append_log)
        self.download_worker.finished.connect(self._on_model_download_finished)
        self.download_worker.finished.connect(self.download_thread.quit)
        self.download_thread.finished.connect(self._cleanup_download_worker)
        self.download_thread.start()

    def _on_model_download_finished(self, success: bool, message: str) -> None:
        self._append_log(message)
        self._auto_detect_models()
        if success:
            QMessageBox.information(self, "模型", message)
            return
        QMessageBox.warning(self, "模型", message)

    def _cleanup_download_worker(self) -> None:
        if self.download_worker is not None:
            self.download_worker.deleteLater()
        if self.download_thread is not None:
            self.download_thread.deleteLater()
        self.download_worker = None
        self.download_thread = None
        self.download_btn.setEnabled(True)

    def _reset_defaults(self) -> None:
        self.mode_combo.setCurrentText("图片超分")
        self.input_edit.clear()
        self.output_edit.setText(str(project_root() / "outputs"))
        self.model_combo.setCurrentText("RealESRGAN_x4plus")
        self.model_path_edit.clear()
        self.outscale_spin.setValue(4.0)
        self.denoise_spin.setValue(0.5)
        self.suffix_edit.setText("out")
        self.tile_spin.setValue(0)
        self.tile_pad_spin.setValue(10)
        self.pre_pad_spin.setValue(0)
        self.face_checkbox.setChecked(False)
        self.fp32_checkbox.setChecked(False)
        self.alpha_combo.setCurrentText("realesrgan")
        self.ext_combo.setCurrentText("auto")
        self.gpu_combo.setCurrentText("自动")
        self.fps_spin.setValue(0)
        self.ffmpeg_edit.setText("ffmpeg")
        self._update_mode_ui()
        self._auto_detect_models()
        self._append_log("参数已恢复默认。")

    def _load_settings(self) -> None:
        default_output = str(project_root() / "outputs")
        self.output_edit.setText(default_output)
        mode = self.settings.value("mode", "image")
        self.mode_combo.setCurrentText(self.MODE_VALUES.get(mode, "图片超分"))
        self.input_edit.setText(self.settings.value("input_path", ""))
        self.output_edit.setText(self.settings.value("output_dir", default_output))
        self.model_combo.setCurrentText(self.settings.value("model_name", "RealESRGAN_x4plus"))
        self.model_path_edit.setText(self.settings.value("model_path", ""))
        self.outscale_spin.setValue(float(self.settings.value("outscale", 4.0)))
        self.denoise_spin.setValue(float(self.settings.value("denoise_strength", 0.5)))
        self.suffix_edit.setText(self.settings.value("suffix", "out"))
        self.tile_spin.setValue(int(self.settings.value("tile", 0)))
        self.tile_pad_spin.setValue(int(self.settings.value("tile_pad", 10)))
        self.pre_pad_spin.setValue(int(self.settings.value("pre_pad", 0)))
        self.face_checkbox.setChecked(self.settings.value("face_enhance", False, bool))
        self.fp32_checkbox.setChecked(self.settings.value("fp32", False, bool))
        self.alpha_combo.setCurrentText(self.settings.value("alpha_upsampler", "realesrgan"))
        self.ext_combo.setCurrentText(self.settings.value("ext", "auto"))
        gpu_value = self.settings.value("gpu_id", "自动")
        self.gpu_combo.setCurrentText(str(gpu_value))
        self.fps_spin.setValue(float(self.settings.value("fps", 0.0)))
        self.ffmpeg_edit.setText(self.settings.value("ffmpeg_bin", "ffmpeg"))

    def _save_settings(self) -> None:
        cfg = self._collect_config()
        data = asdict(cfg)
        data["gpu_id"] = "自动" if cfg.gpu_id is None else str(cfg.gpu_id)
        data["fps"] = 0.0 if cfg.fps is None else cfg.fps
        for key in self.SETTINGS_KEYS:
            self.settings.setValue(key, data[key])

    def closeEvent(self, event):  # noqa: N802
        self._save_settings()
        if self.worker is not None:
            self.worker.stop()
        super().closeEvent(event)


def run_gui() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("RealESRGAN-CN-GUI")
    app.setWindowIcon(QIcon())
    window = MainWindow()
    window.show()
    return app.exec()
