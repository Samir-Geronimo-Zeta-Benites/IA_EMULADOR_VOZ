import json
import time
import numpy as np
from typing import Optional


class VoicemeeterManager:
    def __init__(self, config_path="config/settings.json"):
        with open(config_path) as f:
            cfg = json.load(f)
        vm_cfg = cfg.get("voicemeeter", {})
        self.enabled = vm_cfg.get("enabled", True)
        self.strip_index = vm_cfg.get("strip_index", 0)
        self.output_channel = vm_cfg.get("output_channel", "B1")
        self.vban_ip = vm_cfg.get("vban_ip", "127.0.0.1")
        self.vban_port = vm_cfg.get("vban_port", 6980)
        self._vm = None

    def connect(self) -> bool:
        if not self.enabled:
            return False
        try:
            import voicemeeter

            self._vm = voicemeeter.Idler(self._get_vm_type())
            self._vm.login()
            print(
                f"Voicemeeter conectado (strip {self.strip_index} → "
                f"{self.output_channel})"
            )
            return True
        except ImportError:
            print("voicemeeter-api no instalada")
            return False
        except Exception as e:
            print(f"Error conectando Voicemeeter: {e}")
            return False

    def _get_vm_type(self) -> str:
        try:
            import voicemeeter
            types = voicemeeter.api.get_device_names()
            for t in ["banana", "potato", "basic"]:
                if any(t in n.lower() for n in types):
                    return t
        except Exception:
            pass
        return "banana"

    def set_strip_output(self, strip: int = None, output: str = None):
        if self._vm is None:
            return
        s = strip if strip is not None else self.strip_index
        o = output if output is not None else self.output_channel
        try:
            if hasattr(self._vm, "strip"):
                self._vm.strip[s].A3 = "B" in o and "3" in o
                self._vm.strip[s].A4 = "B" in o and "4" in o
                self._vm.strip[s].A5 = "B" in o and "5" in o
                if "B1" in o:
                    self._vm.strip[s].B1 = True
                if "B2" in o:
                    self._vm.strip[s].B2 = True
                self._vm.apply()
        except Exception as e:
            print(f"Error configurando Voicemeeter: {e}")

    def send_audio(self, audio: np.ndarray, sr: int = 48000):
        if self._vm is None or not self.enabled:
            return
        try:
            if hasattr(self._vm, "vban"):
                self._vm.vban.stream(
                    audio.tobytes(),
                    sample_rate=sr,
                    bit_depth=16,
                    channels=1,
                    ip=self.vban_ip,
                    port=self.vban_port,
                    stream_name="VoicemodOutput",
                )
        except Exception as e:
            print(f"Error enviando audio a Voicemeeter: {e}")

    def disconnect(self):
        if self._vm is not None:
            try:
                self._vm.logout()
            except Exception:
                pass
            self._vm = None

    @staticmethod
    def list_vban_outputs() -> list:
        return [
            {"name": "VBAN", "ip": "127.0.0.1", "port": 6980},
        ]
