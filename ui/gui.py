import json, sys, os, time
from pathlib import Path

try:
    from PyQt6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QPushButton, QLabel, QSlider, QFileDialog, QComboBox,
        QProgressBar, QTextEdit, QFrame, QSizePolicy, QCheckBox,
        QInputDialog, QGroupBox,
    )
    from PyQt6.QtCore import Qt, QTimer, QRect, QPoint
    from PyQt6.QtGui import QFont, QPainter, QColor, QPen, QBrush, QShortcut, QKeySequence, QIcon, QPixmap, QAction
    HAS_PYQT = True
except ImportError:
    HAS_PYQT = False


DARK_QSS = """
QMainWindow, QWidget { background-color: #1a1a2e; color: #e0e0e0; }
QGroupBox {
    border: 1px solid #2d2d5e; border-radius: 6px; margin-top: 14px;
    padding: 12px 8px 8px 8px; font-weight: bold; color: #7c7cf0;
}
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 6px; }
QPushButton {
    background-color: #2d2d5e; color: #e0e0e0; border: 1px solid #3d3d7e;
    border-radius: 5px; padding: 6px 14px; font-size: 12px;
}
QPushButton:hover { background-color: #3d3d7e; border-color: #5c5cff; }
QPushButton:pressed { background-color: #1a1a4e; }
QPushButton:disabled { background-color: #252540; color: #555; border-color: #333; }
QPushButton#startBtn {
    background-color: #2d8a4e; border-color: #3dae5e; font-size: 14px; font-weight: bold; padding: 10px;
}
QPushButton#startBtn:hover { background-color: #3dae5e; }
QPushButton#startBtn:disabled { background-color: #252540; border-color: #333; color: #555; }
QPushButton#startBtn.active { background-color: #c0392b; border-color: #e74c3c; }
QPushButton#startBtn.active:hover { background-color: #e74c3c; }
QPushButton#miniBtn {
    background-color: transparent; border: 1px solid #3d3d7e;
    border-radius: 4px; padding: 4px 8px; font-size: 11px; color: #888;
}
QPushButton#miniBtn:hover { color: #e0e0e0; border-color: #5c5cff; }
QComboBox {
    background-color: #2d2d5e; color: #e0e0e0; border: 1px solid #3d3d7e;
    border-radius: 5px; padding: 4px 8px; min-height: 24px;
}
QComboBox:hover { border-color: #5c5cff; }
QComboBox::drop-down { border: none; width: 20px; }
QComboBox QAbstractItemView {
    background-color: #1a1a2e; color: #e0e0e0; selection-background-color: #3d3d7e;
    border: 1px solid #3d3d7e;
}
QSlider::groove:horizontal { background: #2d2d5e; height: 4px; border-radius: 2px; }
QSlider::handle:horizontal {
    background: #7c7cf0; width: 14px; height: 14px; margin: -5px 0;
    border-radius: 7px;
}
QSlider::sub-page:horizontal { background: #5c5cff; height: 4px; border-radius: 2px; }
QTextEdit {
    background-color: #0d0d1a; color: #a0a0c0; border: 1px solid #2d2d5e;
    border-radius: 4px; padding: 4px; font-family: Consolas, monospace; font-size: 11px;
}
QProgressBar {
    background-color: #0d0d1a; border: 1px solid #2d2d5e; border-radius: 4px;
    text-align: center; color: #e0e0e0;
}
QProgressBar::chunk { background-color: #5c5cff; border-radius: 3px; }
QCheckBox { color: #a0a0c0; spacing: 6px; }
QCheckBox::indicator { width: 14px; height: 14px; border-radius: 3px; border: 1px solid #3d3d7e; background: #2d2d5e; }
QCheckBox::indicator:checked { background: #5c5cff; border-color: #7c7cf0; }
QFrame#header {
    background-color: #12122a; border-bottom: 1px solid #2d2d5e;
    border-radius: 0px; padding: 4px;
}
"""


class WaveformWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.data = []
        self.setMinimumHeight(60)
        self.setMaximumHeight(80)

    def update_data(self, samples):
        if len(samples) > 500:
            samples = samples[::len(samples)//500]
        self.data = samples
        self.update()

    def paintEvent(self, event):
        if not self.data:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        mid = h / 2
        p.setPen(QPen(QColor(0x5c, 0x5c, 0xff), 1.5))
        step = w / max(len(self.data), 1)
        points = [QPoint(int(i * step), int(mid + d * mid * 0.8)) for i, d in enumerate(self.data)]
        for i in range(len(points) - 1):
            p.drawLine(points[i], points[i + 1])
        p.setPen(QPen(QColor(0x3d, 0x3d, 0x7e), 1))
        p.drawLine(0, int(mid), w, int(mid))


class VUMeter(QWidget):
    def __init__(self, label="", parent=None):
        super().__init__(parent)
        self.label = label
        self.level = 0.0
        self.peak = 0.0
        self.setMinimumSize(80, 16)
        self.setMaximumHeight(18)

    def set_level(self, val):
        self.level = max(0.0, min(1.0, val))
        self.peak = max(self.peak, self.level * 1.1)
        self.peak *= 0.98
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        w, h = self.width(), self.height()
        p.setPen(Qt.PenStyle.NoPen)
        bg = QColor(0x0d, 0x0d, 0x1a)
        p.setBrush(bg)
        p.drawRoundedRect(0, 0, w, h, 3, 3)
        bar_w = int((w - 4) * min(self.level, 1.0))
        if bar_w > 0:
            if self.level < 0.6:
                c = QColor(0x2d, 0x8a, 0x4e)
            elif self.level < 0.85:
                c = QColor(0xd4, 0xac, 0x0d)
            else:
                c = QColor(0xc0, 0x39, 0x2b)
            p.setBrush(c)
            p.drawRoundedRect(2, 2, bar_w, h - 4, 2, 2)
        peak_x = int((w - 4) * min(self.peak, 1.0)) + 2
        if 2 < peak_x < w - 2:
            p.setPen(QPen(QColor(0xe0, 0xe0, 0xe0), 1))
            p.drawLine(peak_x, 2, peak_x, h - 2)
        if self.label:
            p.setPen(QColor(0xa0, 0xa0, 0xc0))
            p.drawText(QRect(4, 0, w - 8, h), Qt.AlignmentFlag.AlignVCenter, self.label)


class ProfileManager:
    PROFILES_DIR = Path("models/profiles")

    def __init__(self):
        self.PROFILES_DIR.mkdir(parents=True, exist_ok=True)

    def list_profiles(self):
        return sorted([p.stem for p in self.PROFILES_DIR.glob("*.json")])

    def save(self, name, config):
        with open(self.PROFILES_DIR / f"{name}.json", "w") as f:
            json.dump(config, f, indent=2)

    def load(self, name):
        with open(self.PROFILES_DIR / f"{name}.json") as f:
            return json.load(f)

    def delete(self, name):
        p = self.PROFILES_DIR / f"{name}.json"
        if p.exists():
            p.unlink()


class VoiceModWindow(QMainWindow):
    def __init__(self, config_path: str):
        super().__init__()
        self.config_path = config_path
        with open(config_path) as f:
            self.cfg = json.load(f)
        self.pipeline = None
        self.model_path = self.cfg["rvc"]["model_path"]
        self.is_running = False
        self.compact = False
        self.waveform_data = []
        self.profile_mgr = ProfileManager()
        self._normal_size = (600, 760)
        self._compact_size = (400, 200)
        self._setup_ui()
        self._setup_shortcuts()
        self._setup_timers()
        self._refresh_profiles()

    def _setup_ui(self):
        self.setWindowTitle("VoiceMod - RVC Cloner")
        self.setMinimumSize(400, 400)
        self.resize(*self._normal_size)
        self.setStyleSheet(DARK_QSS)

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        root.addWidget(self._build_header())
        self._body = QWidget()
        self._body_layout = QVBoxLayout(self._body)
        self._body_layout.setContentsMargins(10, 6, 10, 10)
        self._body_layout.setSpacing(8)
        self._build_body()
        root.addWidget(self._body, 1)
        root.addWidget(self._build_footer())
        self._log("VoiceMod iniciado. Carga o entrena un modelo.")

    def _build_header(self):
        h = QFrame()
        h.setObjectName("header")
        h.setMaximumHeight(36)
        lo = QHBoxLayout(h)
        lo.setContentsMargins(10, 2, 10, 2)

        icon = QLabel("🎙")
        icon.setStyleSheet("font-size: 16px;")
        lo.addWidget(icon)

        title = QLabel("VoiceMod")
        title.setStyleSheet("font-size: 14px; font-weight: bold; color: #7c7cf0;")
        lo.addWidget(title)

        self.status_indicator = QLabel("○")
        self.status_indicator.setStyleSheet("font-size: 14px; color: #555;")
        self.status_indicator.setToolTip("Estado del pipeline")
        lo.addWidget(self.status_indicator)

        self.status_text_short = QLabel("Inactivo")
        self.status_text_short.setStyleSheet("font-size: 11px; color: #888;")
        lo.addWidget(self.status_text_short)

        lo.addStretch()

        self.compact_btn = QPushButton("⊞ Mini")
        self.compact_btn.setObjectName("miniBtn")
        self.compact_btn.clicked.connect(self._toggle_compact)
        self.compact_btn.setToolTip("Modo compacto (Ctrl+M)")
        lo.addWidget(self.compact_btn)

        return h

    def _build_body(self):
        lo = self._body_layout
        lo.addWidget(self._build_profile_section())
        lo.addWidget(self._build_model_section())
        lo.addWidget(self._build_device_section())
        lo.addWidget(self._build_audio_test_section())
        lo.addWidget(self._build_control_section())
        lo.addWidget(self._build_meters())
        lo.addWidget(self._build_waveform())
        lo.addWidget(self._build_log())

    def _build_profile_section(self):
        g = QGroupBox("Perfil")
        lo = QHBoxLayout(g)
        self.profile_combo = QComboBox()
        self.profile_combo.setMinimumWidth(160)
        self.profile_combo.currentTextChanged.connect(self._load_selected_profile)
        lo.addWidget(self.profile_combo)
        for txt, tip, slot in [("+ Agregar", "Guardar configuracion actual como nuevo perfil", self._save_profile),
                                ("Eliminar", "Eliminar perfil seleccionado", self._delete_profile)]:
            b = QPushButton(txt)
            b.setMinimumWidth(72)
            b.setToolTip(tip)
            b.clicked.connect(slot)
            lo.addWidget(b)
        lo.addStretch()
        return g

    def _build_model_section(self):
        g = QGroupBox("Modelo")
        lo = QHBoxLayout(g)
        self.model_label = QLabel(os.path.basename(self.model_path) if os.path.exists(self.model_path) else "Sin modelo")
        self.model_label.setStyleSheet("color: #a0a0c0; font-size: 11px;")
        lo.addWidget(self.model_label, 1)
        self.load_btn = QPushButton("Cargar")
        self.load_btn.clicked.connect(self._load_model)
        lo.addWidget(self.load_btn)
        self.train_btn = QPushButton("Entrenar")
        self.train_btn.clicked.connect(self._start_training)
        lo.addWidget(self.train_btn)
        self.train_progress = QProgressBar()
        self.train_progress.setVisible(False)
        self.train_progress.setMaximumHeight(18)
        lo.addWidget(self.train_progress)
        return g

    def _build_device_section(self):
        g = QGroupBox("Dispositivos de Audio")
        lo = QHBoxLayout(g)
        ilo = QVBoxLayout()
        ilo.addWidget(QLabel("Entrada (Micrófono):"))
        self.input_device_combo = QComboBox()
        self.input_device_combo.setMinimumWidth(180)
        self.input_device_combo.currentIndexChanged.connect(self._on_input_device_changed)
        ilo.addWidget(self.input_device_combo)
        lo.addLayout(ilo, 1)
        olo = QVBoxLayout()
        olo.addWidget(QLabel("Salida (Voicemeeter/Meet):"))
        self.output_device_combo = QComboBox()
        self.output_device_combo.setMinimumWidth(180)
        self.output_device_combo.currentIndexChanged.connect(self._on_output_device_changed)
        olo.addWidget(self.output_device_combo)
        lo.addLayout(olo, 1)
        self._refresh_devices()
        return g

    def _refresh_devices(self):
        try:
            import sounddevice as sd
            current_in = self.cfg["audio"].get("input_device")
            current_out = self.cfg["audio"].get("output_device")
            self.input_device_combo.blockSignals(True)
            self.output_device_combo.blockSignals(True)
            self.input_device_combo.clear()
            self.output_device_combo.clear()
            devices = sd.query_devices()
            in_idx = 0
            out_idx = 0
            self.input_device_combo.addItem("Predeterminado", -1)
            self.output_device_combo.addItem("Predeterminado", -1)
            for i, d in enumerate(devices):
                name = f"{d['name']} ({i})"
                if d["max_input_channels"] > 0:
                    self.input_device_combo.addItem(name, i)
                    if current_in == i or (current_in is None and d.get("default", False)):
                        in_idx = self.input_device_combo.count() - 1
                if d["max_output_channels"] > 0:
                    self.output_device_combo.addItem(name, i)
                    if current_out == i or (current_out is None and d.get("default", False)):
                        out_idx = self.output_device_combo.count() - 1
            self.input_device_combo.setCurrentIndex(in_idx)
            self.output_device_combo.setCurrentIndex(out_idx)
        except Exception as e:
            self.input_device_combo.addItem("Error detectando dispositivos")
            self.output_device_combo.addItem("Error detectando dispositivos")
            self._log(f"Error detectando dispositivos: {e}")
        finally:
            self.input_device_combo.blockSignals(False)
            self.output_device_combo.blockSignals(False)

    def _on_input_device_changed(self, idx):
        dev = self.input_device_combo.currentData()
        self.cfg["audio"]["input_device"] = dev
        self._save_config()

    def _on_output_device_changed(self, idx):
        dev = self.output_device_combo.currentData()
        self.cfg["audio"]["output_device"] = dev
        self._save_config()

    def _save_config(self):
        with open(self.config_path, "w") as f:
            json.dump(self.cfg, f, indent=2)

    def _build_audio_test_section(self):
        g = QGroupBox("Prueba de Audio")
        lo = QHBoxLayout(g)

        self.test_mic_btn = QPushButton("Probar Microfono")
        self.test_mic_btn.clicked.connect(self._test_mic)
        lo.addWidget(self.test_mic_btn)

        self.test_output_btn = QPushButton("Probar Salida")
        self.test_output_btn.clicked.connect(self._test_output)
        lo.addWidget(self.test_output_btn)

        return g

    def _test_mic(self):
        import threading, numpy as np, sounddevice as sd
        cfg = self.cfg["audio"]
        sr = cfg["sample_rate"]
        dur = 3

        def run():
            self.test_mic_btn.setEnabled(False)
            self.test_mic_btn.setText("Grabando...")
            self._log("Grabando microfono por 3 segundos...")
            try:
                audio = sd.rec(int(sr * dur), samplerate=sr, channels=1,
                              dtype=np.float32, device=cfg.get("input_device"))
                sd.wait()
                self._log("Reproduciendo grabacion...")
                sd.play(audio, sr, device=cfg.get("output_device"))
                sd.wait()
                self._log("Prueba de microfono completada")
            except Exception as e:
                self._log(f"Error: {e}")
            finally:
                self.test_mic_btn.setText("Probar Microfono")
                self.test_mic_btn.setEnabled(True)

        threading.Thread(target=run, daemon=True).start()

    def _test_output(self):
        import threading, numpy as np, sounddevice as sd
        cfg = self.cfg["audio"]
        sr = cfg["sample_rate"]
        dur = 4

        def run():
            self.test_output_btn.setEnabled(False)
            self.test_output_btn.setText("Grab+Convert...")
            self._log("Grabando 4s, procesando con RVC y reproduciendo en salida virtual...")
            try:
                audio = sd.rec(int(sr * dur), samplerate=sr, channels=1,
                              dtype=np.float32, device=cfg.get("input_device"))
                sd.wait()
                self._log("Audio capturado, aplicando conversion de voz...")
                processed = audio.copy()
                try:
                    from core.vc_engine import VoiceConverter
                    vc = VoiceConverter(self.config_path)
                    if vc.target_stats is not None:
                        converted = vc.convert(audio.squeeze(), sr)
                        converted = np.asarray(converted)
                        if not np.all(np.isfinite(converted)):
                            converted = audio.squeeze()
                            self._log("Audio tenia NaN, usando original")
                        processed = converted.reshape(-1, 1)
                        self._log("Conversion aplicada (voz clonada)")
                    else:
                        self._log("Estadisticas target no cargadas, usando original")
                except Exception as e:
                    self._log(f"Error en conversion: {e}, reproduciendo original")

                sd.play(processed, sr, device=cfg.get("output_device"))
                sd.wait()
                self._log("Prueba de salida completada — verificá Meet")
            except Exception as e:
                self._log(f"Error: {e}")
            finally:
                self.test_output_btn.setText("Probar Salida")
                self.test_output_btn.setEnabled(True)

        threading.Thread(target=run, daemon=True).start()

    def _build_control_section(self):
        g = QGroupBox("Control")
        lo = QVBoxLayout(g)
        top = QHBoxLayout()
        self.start_btn = QPushButton("▶ Iniciar")
        self.start_btn.setObjectName("startBtn")
        self.start_btn.clicked.connect(self._toggle_pipeline)
        self.start_btn.setEnabled(os.path.exists(self.model_path))
        self.start_btn.setMinimumHeight(36)
        top.addWidget(self.start_btn, 2)

        pitch_lo = QHBoxLayout()
        pitch_lo.addWidget(QLabel("Tono:"))
        self.pitch_slider = QSlider(Qt.Orientation.Horizontal)
        self.pitch_slider.setMinimum(-12)
        self.pitch_slider.setMaximum(12)
        self.pitch_slider.setValue(self.cfg["rvc"]["f0_up_key"])
        self.pitch_slider.valueChanged.connect(self._update_pitch)
        self.pitch_label = QLabel(f"{self.cfg['rvc']['f0_up_key']}st")
        self.pitch_label.setFixedWidth(32)
        pitch_lo.addWidget(self.pitch_slider, 1)
        pitch_lo.addWidget(self.pitch_label)
        top.addLayout(pitch_lo, 3)
        lo.addLayout(top)

        bottom = QHBoxLayout()
        self.passthrough_cb = QCheckBox("Passthrough (voz original)")
        self.passthrough_cb.toggled.connect(self._toggle_passthrough)
        bottom.addWidget(self.passthrough_cb)
        bottom.addStretch()
        self.hotkey_label = QLabel("Space=Toggle  M=ModoMini  P=Perfiles")
        self.hotkey_label.setStyleSheet("color: #555; font-size: 10px;")
        bottom.addWidget(self.hotkey_label)
        lo.addLayout(bottom)
        return g

    def _build_meters(self):
        w = QWidget()
        lo = QHBoxLayout(w)
        lo.setContentsMargins(0, 0, 0, 0)
        self.input_vu = VUMeter("IN")
        self.output_vu = VUMeter("OUT")
        self.input_vu.setToolTip("Nivel de entrada (micrófono)")
        self.output_vu.setToolTip("Nivel de salida (voz convertida)")
        lo.addWidget(self.input_vu, 1)
        lo.addWidget(self.output_vu, 1)
        return w

    def _build_waveform(self):
        self.waveform = WaveformWidget()
        self.waveform.setToolTip("Forma de onda en tiempo real")
        return self.waveform

    def _build_log(self):
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(100)
        return self.log

    def _build_footer(self):
        f = QFrame()
        f.setObjectName("header")
        f.setMaximumHeight(28)
        lo = QHBoxLayout(f)
        lo.setContentsMargins(10, 2, 10, 2)
        self.latency_label = QLabel("Latencia: -- ms")
        self.latency_label.setStyleSheet("color: #888; font-size: 11px;")
        lo.addWidget(self.latency_label)
        lo.addStretch()
        self.footer_status = QLabel("⚪ Inactivo")
        self.footer_status.setStyleSheet("color: #555; font-size: 11px;")
        lo.addWidget(self.footer_status)
        return f

    def _setup_shortcuts(self):
        QShortcut(QKeySequence("Space"), self).activated.connect(self._toggle_pipeline)
        QShortcut(QKeySequence("M"), self).activated.connect(self._toggle_compact)
        QShortcut(QKeySequence("P"), self).activated.connect(self._focus_profile)
        QShortcut(QKeySequence("L"), self).activated.connect(self._load_model)
        QShortcut(QKeySequence("Escape"), self).activated.connect(lambda: self._set_compact(False))

    def _setup_timers(self):
        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self._update_status)
        self.status_timer.start(80)
        self.ui_timer = QTimer()
        self.ui_timer.timeout.connect(self._update_ui)
        self.ui_timer.start(1000)

    def _log(self, msg):
        self.log.append(msg)

    def _refresh_profiles(self):
        current = self.profile_combo.currentText()
        self.profile_combo.blockSignals(True)
        self.profile_combo.clear()
        self.profile_combo.addItem("-- Seleccionar perfil --")
        for p in self.profile_mgr.list_profiles():
            self.profile_combo.addItem(p)
        idx = self.profile_combo.findText(current)
        if idx >= 0:
            self.profile_combo.setCurrentIndex(idx)
        self.profile_combo.blockSignals(False)

    def _save_profile(self):
        name, ok = QInputDialog.getText(self, "Agregar perfil", "Nombre del perfil:")
        if ok and name.strip():
            with open(self.config_path) as f:
                cfg = json.load(f)
            self.profile_mgr.save(name.strip(), cfg)
            self._refresh_profiles()
            self._log(f"Perfil guardado: {name}")

    def _load_profile(self):
        self._load_selected_profile(self.profile_combo.currentText())

    def _load_selected_profile(self, name):
        if not name or name.startswith("--"):
            return
        try:
            data = self.profile_mgr.load(name)
            with open(self.config_path, "w") as f:
                json.dump(data, f, indent=2)
            self.cfg = data
            self.model_path = data["rvc"]["model_path"]
            self.model_label.setText(os.path.basename(self.model_path) if os.path.exists(self.model_path) else "Sin modelo")
            self.pitch_slider.setValue(data["rvc"]["f0_up_key"])
            self.start_btn.setEnabled(os.path.exists(self.model_path))
            self._log(f"Perfil cargado: {name}")
        except Exception as e:
            self._log(f"Error cargando perfil: {e}")

    def _delete_profile(self):
        name = self.profile_combo.currentText()
        if not name or name.startswith("--"):
            return
        self.profile_mgr.delete(name)
        self._refresh_profiles()
        self._log(f"Perfil eliminado: {name}")

    def _focus_profile(self):
        self.profile_combo.setFocus()
        self.profile_combo.showPopup()

    def _load_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "Seleccionar modelo .pth",
            str(Path("models/trained")), "Modelo RVC (*.pth);;PyTorch (*.pth)")
        if path:
            self.model_path = path
            self.model_label.setText(os.path.basename(path))
            self.cfg["rvc"]["model_path"] = path
            with open(self.config_path, "w") as f:
                json.dump(self.cfg, f, indent=2)
            self.start_btn.setEnabled(True)
            self._log(f"Modelo cargado: {path}")

    def _start_training(self):
        path, _ = QFileDialog.getOpenFileName(self, "Seleccionar audio de muestra",
            "", "Audio (*.wav *.mp3 *.m4a *.flac *.ogg)")
        if not path:
            return
        self.train_progress.setVisible(True)
        self.train_progress.setRange(0, 0)
        self.train_btn.setEnabled(False)
        self._log(f"Entrenando desde: {path}")
        self._training_path = path
        import threading
        t = threading.Thread(target=self._run_training, daemon=True)
        t.start()

    def _run_training(self):
        import traceback
        try:
            from train import Trainer
            trainer = Trainer(self.config_path)
            p = trainer.run_pipeline(self._training_path)
            self._on_training_done(p)
        except Exception as e:
            tb = traceback.format_exc()
            self._log(f"Error entrenando:\n{tb}")
            self._on_training_error()

    def _on_training_done(self, model_path):
        self.model_path = model_path
        self.model_label.setText(os.path.basename(model_path))
        self.start_btn.setEnabled(True)
        self.train_progress.setVisible(False)
        self.train_btn.setEnabled(True)
        self._log(f"Entrenamiento completado: {model_path}")

    def _on_training_error(self):
        self.train_progress.setVisible(False)
        self.train_btn.setEnabled(True)

    def _toggle_pipeline(self):
        if self.is_running:
            self._stop_pipeline()
        else:
            self._start_pipeline()

    def _start_pipeline(self):
        try:
            from pipeline import RealtimePipeline
            self.pipeline = RealtimePipeline(self.config_path)
            self.pipeline.start()
            self.is_running = True
            self.start_btn.setText("⏹ Detener")
            self.start_btn.setProperty("class", "active")
            self.start_btn.style().unpolish(self.start_btn)
            self.start_btn.style().polish(self.start_btn)
            self.status_indicator.setText("●")
            self.status_indicator.setStyleSheet("font-size: 14px; color: #2d8a4e;")
            self.status_text_short.setText("Activo")
            self.footer_status.setText("🟢 Activo")
            self.footer_status.setStyleSheet("color: #2d8a4e; font-size: 11px;")
            self._log("Pipeline ACTIVO")
        except Exception as e:
            self._log(f"Error: {e}")

    def _stop_pipeline(self):
        if self.pipeline:
            self.pipeline.stop()
            if hasattr(self.pipeline, 'rvc'):
                if hasattr(self.pipeline, 'converter'):
                    pass
            self.pipeline = None
        import gc; gc.collect()
        self.is_running = False
        self.start_btn.setText("▶ Iniciar")
        self.start_btn.setProperty("class", "")
        self.start_btn.style().unpolish(self.start_btn)
        self.start_btn.style().polish(self.start_btn)
        self.status_indicator.setText("○")
        self.status_indicator.setStyleSheet("font-size: 14px; color: #555;")
        self.status_text_short.setText("Inactivo")
        self.footer_status.setText("⚪ Inactivo")
        self.footer_status.setStyleSheet("color: #555; font-size: 11px;")
        self._log("Pipeline detenido")

    def _toggle_passthrough(self, enabled):
        self._log(f"Passthrough: {'ON' if enabled else 'OFF'}")

    def _update_pitch(self, value):
        self.pitch_label.setText(f"{value}st")
        self.cfg["rvc"]["f0_up_key"] = value
        if self.pipeline and hasattr(self.pipeline, 'converter') and self.pipeline.converter:
            self.pipeline.converter.f0_up_key = value

    def _toggle_compact(self):
        self._set_compact(not self.compact)

    def _set_compact(self, on):
        self.compact = on
        if on:
            self._body.hide()
            self.resize(*self._compact_size)
            self.compact_btn.setText("⊞ Normal")
            self.setWindowTitle("VoiceMod [Mini]")
        else:
            self._body.show()
            self.resize(*self._normal_size)
            self.compact_btn.setText("⊞ Mini")
            self.setWindowTitle("VoiceMod - RVC Cloner")

    def _update_status(self):
        if self.pipeline and self.is_running:
            audio = getattr(self.pipeline, '_last_input', None)
            if audio is not None and len(audio) > 0:
                self.waveform_data = audio.squeeze().tolist()[:500]
                self.waveform.update_data(self.waveform_data)
                rms = float((audio ** 2).mean() ** 0.5) * 4
                self.input_vu.set_level(rms)
            output = getattr(self.pipeline, '_last_output', None)
            if output is not None and len(output) > 0:
                rms = float((output ** 2).mean() ** 0.5) * 4
                self.output_vu.set_level(rms)
        else:
            self.input_vu.set_level(0)
            self.output_vu.set_level(0)

    def _update_ui(self):
        if self.pipeline and self.is_running:
            lat = self.pipeline.get_latency_ms()
            self.latency_label.setText(f"Latencia: {lat:.0f} ms")
        else:
            self.latency_label.setText("Latencia: -- ms")


class VoiceModGUI:
    def __init__(self, config_path="config/settings.json"):
        self.config_path = config_path
        with open(config_path) as f:
            self.cfg = json.load(f)
        self.pipeline = None
        self.app = None
        self.window = None

    def run(self):
        if not HAS_PYQT:
            print("PyQt6 no instalado. Usando modo terminal.")
            self._run_terminal()
            return
        self.app = QApplication(sys.argv)
        self.app.setStyle("Fusion")
        self.window = VoiceModWindow(self.config_path)
        self.window.show()
        sys.exit(self.app.exec())

    def _run_terminal(self):
        print("\n=== VoiceMod - Modo Terminal ===")
        print("1. Iniciar pipeline")
        print("2. Entrenar modelo")
        print("3. Salir")
        c = input("Opcion: ").strip()
        if c == "1":
            self._start_pipeline()
        elif c == "2":
            p = input("Ruta del audio: ").strip()
            if os.path.exists(p):
                self._train(p)
            else:
                print("Archivo no encontrado")

    def _start_pipeline(self):
        from pipeline import RealtimePipeline
        self.pipeline = RealtimePipeline(self.config_path)
        self.pipeline.start()
        input("Pipeline activo. Enter para detener.")
        self.pipeline.stop()

    def _train(self, audio_path):
        from train import Trainer
        Trainer(self.config_path).run_pipeline(audio_path)
