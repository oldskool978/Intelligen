from __future__ import annotations
import os
import re
import json
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
from pydantic import BaseModel, Field, field_validator, model_validator

try:
    from pipeline.prompt_compiler import clean_caption, normalize_lyrics
except ImportError:
    try:
        from Intelligen.pipeline.prompt_compiler import clean_caption, normalize_lyrics
    except ImportError:
        def clean_caption(c: str) -> str:
            return re.sub(r"\s+", " ", c).strip()
        def normalize_lyrics(l: str) -> str:
            return re.sub(r"\r\n", "\n", l).strip()

INTELLIGEN_ROOT = Path(__file__).resolve().parent
DEFAULT_PRESET_FILENAME = "default.json"
SUPPORTED_SCHEDULERS = ["heun", "euler", "native"]
SUPPORTED_NOISE_TOPOLOGIES = ["blue_noise", "gaussian"]

BASELINE_ENGINE_DEFAULTS: Dict[str, Any] = {
    "temperature": 0.9192,
    "ar_guidance_scale": 1.5200,
    "top_p": 0.9600,
    "top_k": 47,
    "top_k_layers": [47, 47, 47, 45, 39, 37, 38, 39],
    "scheduler_type": "heun",
    "num_inference_steps": 42,
    "guidance_scale": 1.7800,
    "noise_topology": "blue_noise",
    "blue_noise_alpha": 0.7500,
    "enable_pm_diffusion": True,
    "pm_iterations": 5,
    "pm_conductance": 0.1500,
    "pm_lambda": 0.2000,
    "apply_declick": True,
    "cpu_offload": False,
}

_PRESET_CACHE: Dict[str, Any] = {
    "mtime": -1.0,
    "path": None,
    "has_custom_default": False,
    "defaults": dict(BASELINE_ENGINE_DEFAULTS),
}

def parse_k_vector(val: Union[List[int], str, int, float]) -> Optional[List[int]]:
    if isinstance(val, list):
        if len(val) == 8:
            return [max(1, min(500, int(x))) for x in val]
        elif len(val) == 1:
            return [max(1, min(500, int(val[0])))] * 8
    elif isinstance(val, (int, float)):
        return [max(1, min(500, int(val)))] * 8
    elif isinstance(val, str):
        matches = re.findall(r"(?:=\s*|\b)(\d+)\b", val)
        if len(matches) == 8:
            return [max(1, min(500, int(x))) for x in matches]
        parts = [p.strip() for p in val.split(",") if p.strip()]
        if len(parts) == 8:
            return [max(1, min(500, int(x))) for x in parts]
        elif len(parts) == 1:
            return [max(1, min(500, int(parts[0])))] * 8
    return None

def harvest_engine_preset(data: Dict[str, Any]) -> Dict[str, Any]:
    if "engine_defaults" in data and isinstance(data["engine_defaults"], dict):
        data = data["engine_defaults"]
    harvested: Dict[str, Any] = {}
    t_val = data.get("temperature", data.get("T", data.get("temp")))
    if t_val is not None:
        try:
            harvested["temperature"] = max(0.0001, min(3.0, float(t_val)))
        except (ValueError, TypeError):
            pass
    ar_cfg = data.get("ar_guidance_scale", data.get("ar_cfg", data.get("AR CFG")))
    if ar_cfg is not None:
        try:
            harvested["ar_guidance_scale"] = max(0.0, min(10.0, float(ar_cfg)))
        except (ValueError, TypeError):
            pass
    top_p = data.get("top_p", data.get("Top-P"))
    if top_p is not None:
        try:
            harvested["top_p"] = max(0.0001, min(1.0, float(top_p)))
        except (ValueError, TypeError):
            pass
    top_k = data.get("top_k", data.get("Top-K"))
    if top_k is not None:
        try:
            harvested["top_k"] = max(1, min(500, int(top_k)))
        except (ValueError, TypeError):
            pass
    k_vec = data.get("top_k_layers", data.get("k_vector", data.get("hierarchical_k", data.get("Hierarchical K-Vector"))))
    if k_vec is not None:
        parsed_k = parse_k_vector(k_vec)
        if parsed_k:
            harvested["top_k_layers"] = parsed_k
            if "top_k" not in harvested:
                harvested["top_k"] = parsed_k[0]
    sched = data.get("scheduler_type", data.get("solver", data.get("trajectory", data.get("ODE Solver Trajectory"))))
    if sched is not None and isinstance(sched, str):
        sched_clean = sched.strip().lower()
        if sched_clean in SUPPORTED_SCHEDULERS:
            harvested["scheduler_type"] = sched_clean
    steps = data.get("num_inference_steps", data.get("steps", data.get("Steps", data.get("inference_steps"))))
    if steps is not None:
        try:
            harvested["num_inference_steps"] = max(1, min(200, int(steps)))
        except (ValueError, TypeError):
            pass
    dit_cfg = data.get("guidance_scale", data.get("dit_guidance", data.get("DiT Guidance")))
    if dit_cfg is not None:
        try:
            harvested["guidance_scale"] = max(0.0, min(20.0, float(dit_cfg)))
        except (ValueError, TypeError):
            pass
    topo = data.get("noise_topology", data.get("topology", data.get("Latent Prior Topology")))
    if topo is not None and isinstance(topo, str):
        cleaned_topo = topo.split("(")[0].strip().lower()
        if cleaned_topo in SUPPORTED_NOISE_TOPOLOGIES:
            harvested["noise_topology"] = cleaned_topo
    alpha = data.get("blue_noise_alpha", data.get("alpha", data.get("Alpha")))
    if alpha is not None:
        try:
            harvested["blue_noise_alpha"] = max(0.0, min(2.0, float(alpha)))
        except (ValueError, TypeError):
            pass
    pm_enable = data.get("enable_pm_diffusion", data.get("pm_diffusion", data.get("1D Temporal PM PDE")))
    if pm_enable is not None:
        if isinstance(pm_enable, bool):
            harvested["enable_pm_diffusion"] = pm_enable
        elif isinstance(pm_enable, str):
            harvested["enable_pm_diffusion"] = "enable" in pm_enable.lower()
    iters = data.get("pm_iterations", data.get("iters", data.get("Iters")))
    if iters is not None:
        try:
            harvested["pm_iterations"] = max(1, min(30, int(iters)))
        except (ValueError, TypeError):
            pass
    pm_k = data.get("pm_conductance", data.get("k", data.get("K")))
    if pm_k is not None:
        try:
            harvested["pm_conductance"] = max(0.0001, min(5.0, float(pm_k)))
        except (ValueError, TypeError):
            pass
    pm_lam = data.get("pm_lambda", data.get("lambda", data.get("Lambda")))
    if pm_lam is not None:
        try:
            harvested["pm_lambda"] = max(0.0001, min(0.25, float(pm_lam)))
        except (ValueError, TypeError):
            pass
    declick = data.get("apply_declick", data.get("declick", data.get("DSP Boundary De-Click")))
    if declick is not None:
        if isinstance(declick, bool):
            harvested["apply_declick"] = declick
        elif isinstance(declick, str):
            harvested["apply_declick"] = "enable" in declick.lower()
    offload = data.get("cpu_offload", data.get("cpu_streaming", data.get("Memory CPU Streaming")))
    if offload is not None:
        if isinstance(offload, bool):
            harvested["cpu_offload"] = offload
        elif isinstance(offload, str):
            harvested["cpu_offload"] = "enable" in offload.lower()
    return harvested

def locate_default_preset_file() -> Optional[Path]:
    candidates = [
        INTELLIGEN_ROOT / DEFAULT_PRESET_FILENAME,
        INTELLIGEN_ROOT / "presets" / DEFAULT_PRESET_FILENAME,
        INTELLIGEN_ROOT.parent / "config" / DEFAULT_PRESET_FILENAME,
        INTELLIGEN_ROOT.parent / "webui" / "config" / DEFAULT_PRESET_FILENAME,
    ]
    env_preset = os.environ.get("TUNEBLOOM_DEFAULT_JSON")
    if env_preset:
        candidates.insert(0, Path(env_preset).resolve())
    for c in candidates:
        if c.exists() and c.is_file():
            return c
    return None

def get_active_engine_defaults() -> Dict[str, Any]:
    global _PRESET_CACHE
    preset_file = locate_default_preset_file()
    if preset_file is None:
        if _PRESET_CACHE["has_custom_default"]:
            _PRESET_CACHE["has_custom_default"] = False
            _PRESET_CACHE["mtime"] = -1.0
            _PRESET_CACHE["path"] = None
            _PRESET_CACHE["defaults"] = dict(BASELINE_ENGINE_DEFAULTS)
        return dict(_PRESET_CACHE["defaults"])
    try:
        current_mtime = preset_file.stat().st_mtime
    except OSError:
        return dict(_PRESET_CACHE["defaults"])
    if _PRESET_CACHE["path"] == preset_file and _PRESET_CACHE["mtime"] == current_mtime:
        return dict(_PRESET_CACHE["defaults"])
    try:
        with open(preset_file, "r", encoding="utf-8") as f:
            raw_payload = json.load(f)
        if isinstance(raw_payload, dict):
            extracted = harvest_engine_preset(raw_payload)
            resolved = dict(BASELINE_ENGINE_DEFAULTS)
            resolved.update(extracted)
            _PRESET_CACHE["mtime"] = current_mtime
            _PRESET_CACHE["path"] = preset_file
            _PRESET_CACHE["has_custom_default"] = True
            _PRESET_CACHE["defaults"] = resolved
            return dict(resolved)
    except Exception:
        pass
    return dict(_PRESET_CACHE["defaults"])

def has_custom_default_preset() -> bool:
    get_active_engine_defaults()
    return bool(_PRESET_CACHE["has_custom_default"])

class GenerationRequest(BaseModel):
    genre: str = Field(default="", max_length=60)
    subgenre: str = Field(default="", max_length=60)
    bpm: int = Field(default=0, ge=0, le=300)
    key: str = Field(default="", max_length=30)
    mood: str = Field(default="", max_length=200)
    vocals: str = Field(default="", max_length=300)
    arrangement: str = Field(default="", max_length=300)
    lyrics: str = Field(default="", max_length=4000)
    raw_prompt: Optional[str] = Field(default=None, max_length=5000)
    prompt: Optional[str] = Field(default=None, max_length=5000)
    temperature: Optional[float] = Field(default=0.9192, ge=0.0001, le=3.0)
    top_p: Optional[float] = Field(default=0.9600, ge=0.0001, le=1.0)
    top_k: Optional[int] = Field(default=47, ge=1, le=500)
    top_k_layers: List[int] = Field(
        default_factory=lambda: [47, 47, 47, 45, 39, 37, 38, 39]
    )
    ar_guidance_scale: Optional[float] = Field(default=1.5200, ge=0.0, le=10.0)
    scheduler_type: str = Field(default="heun")
    num_inference_steps: Optional[int] = Field(default=42, ge=1, le=200)
    guidance_scale: Optional[float] = Field(default=1.7800, ge=0.0, le=20.0)
    noise_topology: str = Field(default="blue_noise")
    blue_noise_alpha: float = Field(default=0.7500, ge=0.0, le=2.0)
    enable_pm_diffusion: bool = Field(default=True)
    pm_iterations: int = Field(default=5, ge=1, le=30)
    pm_conductance: float = Field(default=0.1500, ge=0.0001, le=5.0)
    pm_lambda: float = Field(default=0.2000, ge=0.0001, le=0.25)
    audio_duration: float = Field(default=300.0, ge=1.0, le=600.0)
    seed: Optional[int] = Field(default=None, ge=0)
    output_path: str = Field(default="output.wav")
    device: str = Field(default="cuda")
    apply_declick: bool = Field(default=True)
    cpu_offload: bool = Field(default=False)
    repo_id: Optional[str] = Field(default=None)
    blocks: Optional[List[Dict[str, Any]]] = Field(default_factory=list)

    @field_validator("bpm", mode="before")
    @classmethod
    def coerce_bpm(cls, v: Any) -> int:
        if v is None or (isinstance(v, str) and not v.strip()):
            return 0
        try:
            val = int(v)
            return max(0, min(300, val))
        except (ValueError, TypeError):
            return 0

    @field_validator("seed", mode="before")
    @classmethod
    def coerce_seed(cls, v: Any) -> Optional[int]:
        if v is None or (isinstance(v, str) and not v.strip()):
            return None
        try:
            return int(v)
        except (ValueError, TypeError):
            return None

    @field_validator(
        "temperature",
        "top_p",
        "ar_guidance_scale",
        "guidance_scale",
        "blue_noise_alpha",
        "pm_conductance",
        "pm_lambda",
        "audio_duration",
        mode="before",
    )
    @classmethod
    def coerce_floats(cls, v: Any) -> Any:
        if v is None or (isinstance(v, str) and not v.strip()):
            return None
        try:
            return float(v)
        except (ValueError, TypeError):
            return v

    @field_validator("top_k_layers", mode="before")
    @classmethod
    def coerce_top_k_layers(cls, v: Any) -> List[int]:
        if v is None:
            return list(get_active_engine_defaults()["top_k_layers"])
        parsed = parse_k_vector(v)
        if parsed:
            return parsed
        return list(get_active_engine_defaults()["top_k_layers"])

    @model_validator(mode="after")
    def synchronize_active_engine_preset(self) -> GenerationRequest:
        active = get_active_engine_defaults()
        use_custom = has_custom_default_preset()
        if self.temperature is None or (use_custom and self.temperature == BASELINE_ENGINE_DEFAULTS["temperature"]):
            self.temperature = float(active["temperature"])
        if self.ar_guidance_scale is None or (use_custom and self.ar_guidance_scale == BASELINE_ENGINE_DEFAULTS["ar_guidance_scale"]):
            self.ar_guidance_scale = float(active["ar_guidance_scale"])
        if self.top_p is None or (use_custom and self.top_p == BASELINE_ENGINE_DEFAULTS["top_p"]):
            self.top_p = float(active["top_p"])
        if self.top_k_layers is None or (use_custom and self.top_k_layers == BASELINE_ENGINE_DEFAULTS["top_k_layers"]):
            self.top_k_layers = list(active["top_k_layers"])
            self.top_k = active["top_k"]
        if self.top_k is None or (use_custom and self.top_k == BASELINE_ENGINE_DEFAULTS["top_k"]):
            self.top_k = self.top_k_layers[0] if self.top_k_layers else active["top_k"]
        if self.scheduler_type is None or (use_custom and self.scheduler_type == BASELINE_ENGINE_DEFAULTS["scheduler_type"]):
            self.scheduler_type = str(active["scheduler_type"])
        if self.num_inference_steps is None or (use_custom and self.num_inference_steps == BASELINE_ENGINE_DEFAULTS["num_inference_steps"]):
            self.num_inference_steps = int(active["num_inference_steps"])
        if self.guidance_scale is None or (use_custom and self.guidance_scale == BASELINE_ENGINE_DEFAULTS["guidance_scale"]):
            self.guidance_scale = float(active["guidance_scale"])
        if self.noise_topology is None or (use_custom and self.noise_topology == BASELINE_ENGINE_DEFAULTS["noise_topology"]):
            self.noise_topology = str(active["noise_topology"])
        if self.blue_noise_alpha is None or (use_custom and self.blue_noise_alpha == BASELINE_ENGINE_DEFAULTS["blue_noise_alpha"]):
            self.blue_noise_alpha = float(active["blue_noise_alpha"])
        if self.enable_pm_diffusion is None or (use_custom and self.enable_pm_diffusion == BASELINE_ENGINE_DEFAULTS["enable_pm_diffusion"]):
            self.enable_pm_diffusion = bool(active["enable_pm_diffusion"])
        if self.pm_iterations is None or (use_custom and self.pm_iterations == BASELINE_ENGINE_DEFAULTS["pm_iterations"]):
            self.pm_iterations = int(active["pm_iterations"])
        if self.pm_conductance is None or (use_custom and self.pm_conductance == BASELINE_ENGINE_DEFAULTS["pm_conductance"]):
            self.pm_conductance = float(active["pm_conductance"])
        if self.pm_lambda is None or (use_custom and self.pm_lambda == BASELINE_ENGINE_DEFAULTS["pm_lambda"]):
            self.pm_lambda = float(active["pm_lambda"])
        if self.apply_declick is None or (use_custom and self.apply_declick == BASELINE_ENGINE_DEFAULTS["apply_declick"]):
            self.apply_declick = bool(active["apply_declick"])
        if self.cpu_offload is None or (use_custom and self.cpu_offload == BASELINE_ENGINE_DEFAULTS["cpu_offload"]):
            self.cpu_offload = bool(active["cpu_offload"])
        if self.top_k_layers and len(self.top_k_layers) == 8:
            self.top_k = self.top_k_layers[0]
        return self

    def set_macro_k(self, fundamental: int, acoustic: int, fine: int) -> None:
        f_val = max(1, min(500, int(fundamental)))
        a_val = max(1, min(500, int(acoustic)))
        fn_val = max(1, min(500, int(fine)))
        self.top_k_layers = [f_val, a_val, a_val, a_val, fn_val, fn_val, fn_val, fn_val]
        self.top_k = f_val

    def set_layer_k(self, layer_idx: int, k_val: int) -> None:
        if not (0 <= layer_idx <= 7):
            raise IndexError("Layer index must be within [0, 7]")
        self.top_k_layers[layer_idx] = max(1, min(500, int(k_val)))
        if layer_idx == 0:
            self.top_k = self.top_k_layers[0]

    def resolve_top_k_layers(self) -> List[int]:
        if len(self.top_k_layers) == 8:
            return [int(k) for k in self.top_k_layers]
        fallback_k = int(self.top_k) if self.top_k is not None else get_active_engine_defaults()["top_k"]
        return [fallback_k] * 8

    def compile_prompt(self) -> str:
        candidate_prompt = self.prompt or self.raw_prompt
        if candidate_prompt and candidate_prompt.strip():
            return clean_caption(candidate_prompt.strip())
        key_clean = self.key.strip() if self.key else ""
        attr_parts = []
        if self.bpm and self.bpm > 0:
            attr_parts.append(f"bpm is {self.bpm}")
        if key_clean:
            key_match = re.match(r"^([A-G][b#]?)\s*(major|minor|m)?", key_clean, re.IGNORECASE)
            if key_match:
                key_root = key_match.group(1).upper()
                if len(key_root) > 1 and key_root[1] == "B":
                    key_root = key_root[0] + "b"
                mode_token = (key_match.group(2) or "").lower()
                scale_mode = "major" if mode_token == "major" else "minor"
                attr_parts.append(f"key is {key_root}, and scale is {scale_mode}")
            else:
                attr_parts.append(f"key is {key_clean}")
        genre_desc = " / ".join(filter(None, [self.genre.strip(), self.subgenre.strip()]))
        if genre_desc:
            attr_parts.append(genre_desc)
        segments = []
        if attr_parts:
            segments.append(f"Basic Attributes: {'. '.join(attr_parts)}.")
        if self.mood and self.mood.strip():
            m = self.mood.strip()
            segments.append(f"Mood: {m if m.endswith('.') else m + '.'}")
        if self.vocals and self.vocals.strip():
            v = self.vocals.strip()
            segments.append(f"Vocals: {v if v.endswith('.') else v + '.'}")
        if self.arrangement and self.arrangement.strip():
            a = self.arrangement.strip()
            segments.append(f"Arrangement: {a if a.endswith('.') else a + '.'}")
        compiled = " ".join(segments).strip()
        return clean_caption(compiled) if compiled else "Instrumental Music"

    def sanitize_lyrics(self) -> str:
        if self.blocks and len(self.blocks) > 0:
            compiled_blocks = []
            for b in self.blocks:
                if not isinstance(b, dict):
                    continue
                lbl = b.get("label") or b.get("type") or "verse"
                txt = (b.get("text") or "").strip()
                compiled_blocks.append(f"[{lbl.strip()}]\n{txt}" if txt else f"[{lbl.strip()}]")
            return normalize_lyrics("\n\n".join(compiled_blocks))
        if self.lyrics and self.lyrics.strip():
            return normalize_lyrics(self.lyrics)
        return ""

    def validate(self) -> None:
        if self.audio_duration <= 0.0 or self.audio_duration > 600.0:
            raise ValueError(f"Duration {self.audio_duration}s out of bounds (0.0 < t <= 600.0s).")
        if self.bpm is not None and self.bpm != 0 and (self.bpm < 30 or self.bpm > 300):
            raise ValueError(f"BPM {self.bpm} out of practical range (30-300 or 0 for unmetered).")
        if self.scheduler_type not in SUPPORTED_SCHEDULERS:
            raise ValueError(f"Scheduler '{self.scheduler_type}' invalid. Must be one of: {SUPPORTED_SCHEDULERS}")
        if self.num_inference_steps is not None and (self.num_inference_steps < 1 or self.num_inference_steps > 200):
            raise ValueError(f"Inference steps {self.num_inference_steps} out of bounds (1-200).")
        if self.guidance_scale is not None and (self.guidance_scale < 0.0 or self.guidance_scale > 20.0):
            raise ValueError(f"DiT Guidance scale {self.guidance_scale} out of bounds (0.0-20.0).")
        if self.ar_guidance_scale is not None and (self.ar_guidance_scale < 0.0 or self.ar_guidance_scale > 10.0):
            raise ValueError(f"AR Guidance scale {self.ar_guidance_scale} out of bounds (0.0-10.0).")
        if self.temperature is not None and (self.temperature <= 0.0 or self.temperature > 3.0):
            raise ValueError(f"Temperature {self.temperature} out of bounds (0.0 < T <= 3.0).")
        if self.top_p is not None and (self.top_p <= 0.0 or self.top_p > 1.0):
            raise ValueError(f"Top-P {self.top_p} out of bounds (0.0 < p <= 1.0).")
        if self.top_k is not None and (self.top_k < 1 or self.top_k > 500):
            raise ValueError(f"Top-K {self.top_k} out of bounds (1-500).")
        if self.noise_topology not in SUPPORTED_NOISE_TOPOLOGIES:
            raise ValueError(f"Noise topology '{self.noise_topology}' invalid. Must be one of: {SUPPORTED_NOISE_TOPOLOGIES}")
        if self.blue_noise_alpha < 0.0 or self.blue_noise_alpha > 2.0:
            raise ValueError(f"Blue noise alpha {self.blue_noise_alpha} out of bounds (0.0-2.0).")
        if self.pm_iterations < 1 or self.pm_iterations > 30:
            raise ValueError(f"Perona-Malik iterations {self.pm_iterations} out of bounds (1-30).")
        if self.pm_conductance <= 0.0 or self.pm_conductance > 5.0:
            raise ValueError(f"Perona-Malik conductance {self.pm_conductance} out of bounds (0.0 < K <= 5.0).")
        if self.pm_lambda <= 0.0 or self.pm_lambda > 0.25:
            raise ValueError(f"Perona-Malik lambda {self.pm_lambda} exceeds stability bound (0.0 < lambda <= 0.25).")

    def save_preset(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_suffix(".tmp")
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(self.model_dump(), f, indent=2, ensure_ascii=False)
        temp_path.replace(path)
        if path.name == DEFAULT_PRESET_FILENAME:
            global _PRESET_CACHE
            _PRESET_CACHE["mtime"] = -1.0
            get_active_engine_defaults()

    @classmethod
    def load_preset(cls, path: Path) -> GenerationRequest:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)

class GenerationResponse(BaseModel):
    output_path: str
    sample_rate: int
    duration_seconds: float
    total_samples: int
    generation_time_seconds: float
    real_time_factor: float
    peak_vram_gb: float
    cpu_offload_active: bool
    scheduler_used: str
    noise_topology_used: str
    pm_diffusion_used: bool
    declick_applied: bool
    peak_linear: float
    peak_dbfs: float
    rms_dbfs: float
    crest_factor_db: float
    effective_prompt: str
    top_k_vector_used: List[int]