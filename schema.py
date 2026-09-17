from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator

try:
    from pipeline.prompt_compiler import clean_caption, normalize_lyrics, _LEADING_TAGS_RE
except ImportError:
    try:
        from Intelligen.pipeline.prompt_compiler import clean_caption, normalize_lyrics, _LEADING_TAGS_RE
    except ImportError:
        _LEADING_TAGS_RE = re.compile(r"^[ \t]*((?:\[[^\]]+\][ \t]*)+)")

        def clean_caption(c: str) -> str:
            return re.sub(r"[ \t]{2,}", " ", re.sub(r"\n{2,}", "\n", c)).strip()

        def normalize_lyrics(l: str) -> str:
            return re.sub(r"\r\n", "\n", l).strip()

INTELLIGEN_ROOT = Path(__file__).resolve().parent
DEFAULT_PRESET_FILENAME = "default.json"
SUPPORTED_SOLVERS = ["heun", "euler", "ipndm", "sde_gpu_pp", "multitree"]

BASELINE_ENGINE_DEFAULTS: Dict[str, Any] = {
    "temperature": 0.9192,
    "ar_guidance_scale": 1.5200,
    "top_p": 1.0000,
    "top_k": 47,
    "top_k_layers": [47, 47, 47, 45, 39, 37, 38, 39],
    "early_instrumental_solver": "heun",
    "late_instrumental_solver": "ipndm",
    "early_vocal_solver": "heun",
    "late_vocal_solver": "ipndm",
    "handoff_threshold": 0.3257,
    "num_inference_steps": 42,
    "instrumental_guidance_scale": 1.7800,
    "vocal_guidance_scale": 1.7800,
    "early_instrumental_cfg": 1.7800,
    "late_instrumental_cfg": 1.0000,
    "early_vocal_cfg": 1.7800,
    "late_vocal_cfg": 1.0000,
    "eta": 0.0,
    "s_noise": 1.0,
    "vocoder_batch_size": 2,
    "apply_declick": True,
    "cpu_offload": False,
    "is_instrumental": False,
    "instrumental_branch": "cues",
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

    if "temperature" in data and data["temperature"] is not None:
        try:
            harvested["temperature"] = max(0.0001, min(3.0, float(data["temperature"])))
        except (ValueError, TypeError):
            pass

    if "ar_guidance_scale" in data and data["ar_guidance_scale"] is not None:
        try:
            harvested["ar_guidance_scale"] = max(0.0, min(10.0, float(data["ar_guidance_scale"])))
        except (ValueError, TypeError):
            pass

    if "top_p" in data and data["top_p"] is not None:
        try:
            harvested["top_p"] = max(0.0001, min(1.0, float(data["top_p"])))
        except (ValueError, TypeError):
            pass

    if "top_k" in data and data["top_k"] is not None:
        try:
            harvested["top_k"] = max(1, min(500, int(data["top_k"])))
        except (ValueError, TypeError):
            pass

    if "top_k_layers" in data and data["top_k_layers"] is not None:
        parsed_k = parse_k_vector(data["top_k_layers"])
        if parsed_k:
            harvested["top_k_layers"] = parsed_k
            if "top_k" not in harvested:
                harvested["top_k"] = parsed_k[0]

    if "early_instrumental_solver" in data and isinstance(data["early_instrumental_solver"], str):
        solver = data["early_instrumental_solver"].strip().lower()
        if solver in SUPPORTED_SOLVERS:
            harvested["early_instrumental_solver"] = solver

    if "late_instrumental_solver" in data and isinstance(data["late_instrumental_solver"], str):
        solver = data["late_instrumental_solver"].strip().lower()
        if solver in SUPPORTED_SOLVERS:
            harvested["late_instrumental_solver"] = solver

    if "early_vocal_solver" in data and isinstance(data["early_vocal_solver"], str):
        solver = data["early_vocal_solver"].strip().lower()
        if solver in SUPPORTED_SOLVERS:
            harvested["early_vocal_solver"] = solver

    if "late_vocal_solver" in data and isinstance(data["late_vocal_solver"], str):
        solver = data["late_vocal_solver"].strip().lower()
        if solver in SUPPORTED_SOLVERS:
            harvested["late_vocal_solver"] = solver

    if "handoff_threshold" in data and data["handoff_threshold"] is not None:
        try:
            harvested["handoff_threshold"] = max(0.0, min(1.0, float(data["handoff_threshold"])))
        except (ValueError, TypeError):
            pass

    if "num_inference_steps" in data and data["num_inference_steps"] is not None:
        try:
            harvested["num_inference_steps"] = max(1, min(200, int(data["num_inference_steps"])))
        except (ValueError, TypeError):
            pass

    if "instrumental_guidance_scale" in data and data["instrumental_guidance_scale"] is not None:
        try:
            val = max(0.0, min(20.0, float(data["instrumental_guidance_scale"])))
            harvested["instrumental_guidance_scale"] = val
            if "early_instrumental_cfg" not in data:
                harvested["early_instrumental_cfg"] = val
        except (ValueError, TypeError):
            pass

    if "early_instrumental_cfg" in data and data["early_instrumental_cfg"] is not None:
        try:
            harvested["early_instrumental_cfg"] = max(0.0, min(20.0, float(data["early_instrumental_cfg"])))
        except (ValueError, TypeError):
            pass

    if "late_instrumental_cfg" in data and data["late_instrumental_cfg"] is not None:
        try:
            harvested["late_instrumental_cfg"] = max(0.0, min(20.0, float(data["late_instrumental_cfg"])))
        except (ValueError, TypeError):
            pass

    if "vocal_guidance_scale" in data and data["vocal_guidance_scale"] is not None:
        try:
            val = max(0.0, min(20.0, float(data["vocal_guidance_scale"])))
            harvested["vocal_guidance_scale"] = val
            if "early_vocal_cfg" not in data:
                harvested["early_vocal_cfg"] = val
        except (ValueError, TypeError):
            pass

    if "early_vocal_cfg" in data and data["early_vocal_cfg"] is not None:
        try:
            harvested["early_vocal_cfg"] = max(0.0, min(20.0, float(data["early_vocal_cfg"])))
        except (ValueError, TypeError):
            pass

    if "late_vocal_cfg" in data and data["late_vocal_cfg"] is not None:
        try:
            harvested["late_vocal_cfg"] = max(0.0, min(20.0, float(data["late_vocal_cfg"])))
        except (ValueError, TypeError):
            pass

    if "eta" in data and data["eta"] is not None:
        try:
            harvested["eta"] = max(0.0, min(1.0, float(data["eta"])))
        except (ValueError, TypeError):
            pass

    if "s_noise" in data and data["s_noise"] is not None:
        try:
            harvested["s_noise"] = max(0.0, min(5.0, float(data["s_noise"])))
        except (ValueError, TypeError):
            pass

    if "vocoder_batch_size" in data and data["vocoder_batch_size"] is not None:
        try:
            harvested["vocoder_batch_size"] = max(1, min(32, int(data["vocoder_batch_size"])))
        except (ValueError, TypeError):
            pass

    if "apply_declick" in data and data["apply_declick"] is not None:
        harvested["apply_declick"] = bool(data["apply_declick"])

    if "cpu_offload" in data and data["cpu_offload"] is not None:
        harvested["cpu_offload"] = bool(data["cpu_offload"])

    if "is_instrumental" in data and data["is_instrumental"] is not None:
        harvested["is_instrumental"] = bool(data["is_instrumental"])

    if "instrumental_branch" in data and isinstance(data["instrumental_branch"], str):
        branch_str = data["instrumental_branch"].strip().lower()
        if branch_str in ("cues", "tags_only"):
            harvested["instrumental_branch"] = branch_str

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
    mood: str = Field(default="", max_length=400)
    vocals: str = Field(default="", max_length=800)
    vocal_lead: Optional[str] = Field(default="", max_length=800)
    instrumental_lead: Optional[str] = Field(default="", max_length=800)
    arrangement: str = Field(default="", max_length=1500)
    lyrics: str = Field(default="", max_length=4000)
    instrumental_lyrics: str = Field(default="", max_length=4000)
    is_instrumental: bool = Field(default=False)
    instrumental_branch: str = Field(default="cues")
    raw_prompt: Optional[str] = Field(default=None, max_length=5000)
    prompt: Optional[str] = Field(default=None, max_length=5000)
    temperature: Optional[float] = Field(default=None, ge=0.0001, le=3.0)
    top_p: Optional[float] = Field(default=None, ge=0.0001, le=1.0)
    top_k: Optional[int] = Field(default=None, ge=1, le=500)
    top_k_layers: Optional[List[int]] = Field(default=None)
    ar_guidance_scale: Optional[float] = Field(default=None, ge=0.0, le=10.0)
    early_instrumental_solver: Optional[str] = Field(default=None)
    late_instrumental_solver: Optional[str] = Field(default=None)
    early_vocal_solver: Optional[str] = Field(default=None)
    late_vocal_solver: Optional[str] = Field(default=None)
    handoff_threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    num_inference_steps: Optional[int] = Field(default=None, ge=1, le=200)
    instrumental_guidance_scale: Optional[float] = Field(default=None, ge=0.0, le=20.0)
    early_instrumental_cfg: Optional[float] = Field(default=None, ge=0.0, le=20.0)
    late_instrumental_cfg: Optional[float] = Field(default=None, ge=0.0, le=20.0)
    vocal_guidance_scale: Optional[float] = Field(default=None, ge=0.0, le=20.0)
    early_vocal_cfg: Optional[float] = Field(default=None, ge=0.0, le=20.0)
    late_vocal_cfg: Optional[float] = Field(default=None, ge=0.0, le=20.0)
    eta: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    s_noise: Optional[float] = Field(default=None, ge=0.0, le=5.0)
    vocoder_batch_size: Optional[int] = Field(default=None, ge=1, le=32)
    audio_duration: float = Field(default=300.0, ge=1.0, le=600.0)
    seed: Optional[int] = Field(default=None, ge=0)
    output_path: str = Field(default="output.wav")
    device: str = Field(default="cuda")
    apply_declick: Optional[bool] = Field(default=None)
    cpu_offload: Optional[bool] = Field(default=None)
    repo_id: Optional[str] = Field(default=None)
    blocks: Optional[List[Dict[str, Any]]] = Field(default_factory=list)
    instrumental_blocks: Optional[List[Dict[str, Any]]] = Field(default_factory=list)

    @field_validator("bpm", mode="before")
    @classmethod
    def coerce_bpm(cls, v: Any) -> int:
        if v is None or (isinstance(v, str) and not v.strip()):
            return 0
        try:
            return max(0, min(300, int(v)))
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

    @field_validator("vocoder_batch_size", mode="before")
    @classmethod
    def coerce_vocoder_batch_size(cls, v: Any) -> Optional[int]:
        if v is None or (isinstance(v, str) and not v.strip()):
            return None
        try:
            return max(1, min(32, int(v)))
        except (ValueError, TypeError):
            return None

    @field_validator(
        "temperature",
        "top_p",
        "ar_guidance_scale",
        "instrumental_guidance_scale",
        "early_instrumental_cfg",
        "late_instrumental_cfg",
        "vocal_guidance_scale",
        "early_vocal_cfg",
        "late_vocal_cfg",
        "eta",
        "s_noise",
        "handoff_threshold",
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
    def coerce_top_k_layers(cls, v: Any) -> Optional[List[int]]:
        if v is None or (isinstance(v, str) and not v.strip()):
            return None
        return parse_k_vector(v)

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

        if self.early_instrumental_solver is None or (use_custom and self.early_instrumental_solver == BASELINE_ENGINE_DEFAULTS["early_instrumental_solver"]):
            self.early_instrumental_solver = str(active["early_instrumental_solver"])

        if self.late_instrumental_solver is None or (use_custom and self.late_instrumental_solver == BASELINE_ENGINE_DEFAULTS["late_instrumental_solver"]):
            self.late_instrumental_solver = str(active["late_instrumental_solver"])

        if self.early_vocal_solver is None or (use_custom and self.early_vocal_solver == BASELINE_ENGINE_DEFAULTS["early_vocal_solver"]):
            self.early_vocal_solver = str(active["early_vocal_solver"])

        if self.late_vocal_solver is None or (use_custom and self.late_vocal_solver == BASELINE_ENGINE_DEFAULTS["late_vocal_solver"]):
            self.late_vocal_solver = str(active["late_vocal_solver"])

        if self.handoff_threshold is None or (use_custom and self.handoff_threshold == BASELINE_ENGINE_DEFAULTS["handoff_threshold"]):
            self.handoff_threshold = float(active["handoff_threshold"])

        if self.num_inference_steps is None or (use_custom and self.num_inference_steps == BASELINE_ENGINE_DEFAULTS["num_inference_steps"]):
            self.num_inference_steps = int(active["num_inference_steps"])

        if self.early_instrumental_cfg is None or (use_custom and self.early_instrumental_cfg == BASELINE_ENGINE_DEFAULTS["early_instrumental_cfg"]):
            self.early_instrumental_cfg = float(active["early_instrumental_cfg"])

        if self.late_instrumental_cfg is None or (use_custom and self.late_instrumental_cfg == BASELINE_ENGINE_DEFAULTS["late_instrumental_cfg"]):
            self.late_instrumental_cfg = float(active["late_instrumental_cfg"])

        if self.instrumental_guidance_scale is None:
            self.instrumental_guidance_scale = self.early_instrumental_cfg

        if self.early_vocal_cfg is None or (use_custom and self.early_vocal_cfg == BASELINE_ENGINE_DEFAULTS["early_vocal_cfg"]):
            self.early_vocal_cfg = float(active["early_vocal_cfg"])

        if self.late_vocal_cfg is None or (use_custom and self.late_vocal_cfg == BASELINE_ENGINE_DEFAULTS["late_vocal_cfg"]):
            self.late_vocal_cfg = float(active["late_vocal_cfg"])

        if self.vocal_guidance_scale is None:
            self.vocal_guidance_scale = self.early_vocal_cfg

        if self.eta is None or (use_custom and self.eta == BASELINE_ENGINE_DEFAULTS["eta"]):
            self.eta = float(active["eta"])

        if self.s_noise is None or (use_custom and self.s_noise == BASELINE_ENGINE_DEFAULTS["s_noise"]):
            self.s_noise = float(active["s_noise"])

        if self.vocoder_batch_size is None or (use_custom and self.vocoder_batch_size == BASELINE_ENGINE_DEFAULTS["vocoder_batch_size"]):
            self.vocoder_batch_size = int(active.get("vocoder_batch_size", 2))

        if self.apply_declick is None or (use_custom and self.apply_declick == BASELINE_ENGINE_DEFAULTS["apply_declick"]):
            self.apply_declick = bool(active["apply_declick"])

        if self.cpu_offload is None or (use_custom and self.cpu_offload == BASELINE_ENGINE_DEFAULTS["cpu_offload"]):
            self.cpu_offload = bool(active["cpu_offload"])

        if self.top_k_layers and len(self.top_k_layers) == 8:
            self.top_k = self.top_k_layers[0]

        if not self.is_instrumental:
            if not self.vocal_lead and self.vocals:
                if not self.instrumental_lead or self.vocals != self.instrumental_lead:
                    self.vocal_lead = self.vocals
                else:
                    self.vocals = ""
            elif not self.vocals and self.vocal_lead:
                self.vocals = self.vocal_lead
        else:
            if not self.instrumental_lead and self.vocals:
                if not self.vocal_lead or self.vocals != self.vocal_lead:
                    self.instrumental_lead = self.vocals
                else:
                    self.vocals = ""
            elif not self.vocals and self.instrumental_lead:
                self.vocals = self.instrumental_lead

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
            key_match = re.match(r"^([A-G][b#]?)(?:\s*(major|minor|min|maj|m))?\s*$", key_clean, re.IGNORECASE)
            if key_match:
                raw_root = key_match.group(1)
                if len(raw_root) > 1:
                    key_root = raw_root[0].upper() + raw_root[1:].lower()
                else:
                    key_root = raw_root.upper()
                mode_token = (key_match.group(2) or "").lower()
                scale_mode = "minor" if mode_token in ("minor", "min", "m") else "major"
                attr_parts.append(f"key is {key_root}, and scale is {scale_mode}")
            else:
                attr_parts.append(f"key is {key_clean}")

        genre_desc = " / ".join(filter(None, [self.genre.strip(), self.subgenre.strip()]))
        if genre_desc:
            attr_parts.append(genre_desc)

        global_meta_lines = ["Global Metadata"]
        if attr_parts:
            global_meta_lines.append(f"Basic Attributes: {'. '.join(attr_parts)}.")

        mood_clean = self.mood.strip() if self.mood else ""
        if mood_clean:
            if not mood_clean.endswith("."):
                mood_clean += "."
            global_meta_lines.append(f"Global Emotional Progression: {mood_clean}")

        vocal_lines = ["Vocal Details"]
        if self.is_instrumental:
            if self.instrumental_lead is not None and self.instrumental_lead.strip():
                lead_clean = self.instrumental_lead.strip()
            elif self.vocals and self.vocals.strip() and self.vocals.strip() != (self.vocal_lead or "").strip():
                lead_clean = self.vocals.strip()
            elif self.vocals and self.vocals.strip() and not (self.vocal_lead or "").strip():
                lead_clean = self.vocals.strip()
            else:
                lead_clean = ""

            if lead_clean:
                if not lead_clean.endswith("."):
                    lead_clean += "."

                is_explicit_non_vocal = bool(
                    re.search(
                        r"\b(no|without|zero|strictly\s+no)\s+(vocals?|singing|voices?|choirs?)\b",
                        lead_clean,
                        re.IGNORECASE,
                    )
                    or re.search(r"\bnon-?vocals?\b", lead_clean, re.IGNORECASE)
                )

                has_vocal_textures = bool(
                    re.search(
                        r"\b(vocals?|voices?|singers?|singing|choirs?|choral|talkbox|vocoders?|humming|hums?|scat|chants?|chanting|vocalise|ad-?libs?|falsetto)\b",
                        lead_clean,
                        re.IGNORECASE,
                    )
                    or re.search(
                        r"\b(vocal\s+chops?|voice\s+chops?|chopped\s+vocals?)\b",
                        lead_clean,
                        re.IGNORECASE,
                    )
                    or re.search(
                        r"\b(soprano|alto|tenor|baritone)\s+(vocals?|voices?|singers?|lead|delivery|choral|harmonies?|range|melisma)\b",
                        lead_clean,
                        re.IGNORECASE,
                    )
                    or re.search(
                        r"\b(female|male|boy|girl|children|gospel)\s+(vocals?|voices?|singers?|choir|harmonies?)\b",
                        lead_clean,
                        re.IGNORECASE,
                    )
                )

                if not is_explicit_non_vocal and has_vocal_textures:
                    vocal_lines.append(
                        f"Wordless vocal textures and acoustic character: {lead_clean} Strictly wordless vocalizations, no spoken or sung lyrics."
                    )
                else:
                    vocal_lines.append(
                        f"Instrumental composition. Lead acoustic melody: {lead_clean} Strictly no vocals, voices, or choral layers."
                    )
            else:
                vocal_lines.append("Instrumental composition. Strictly no vocals, voices, or choral layers.")
        else:
            vocals_clean = (self.vocal_lead or self.vocals or "").strip()
            if vocals_clean:
                if not vocals_clean.endswith("."):
                    vocals_clean += "."
                vocal_lines.append(vocals_clean)
            else:
                vocal_lines.append("Vocal performance.")

        arr_lines = ["Arrangement"]
        arr_clean = self.arrangement.strip() if self.arrangement else ""
        if arr_clean:
            if not arr_clean.endswith("."):
                arr_clean += "."
            arr_lines.append(arr_clean)
        else:
            if self.is_instrumental:
                arr_lines.append("Dynamic full acoustic and electronic arrangement with prominent instrumental leads.")
            else:
                arr_lines.append("Dynamic full acoustic arrangement.")

        sections = [
            "\n".join(global_meta_lines),
            "\n".join(vocal_lines),
            "\n".join(arr_lines),
        ]
        compiled = "\n\n".join(sections).strip()
        return clean_caption(compiled)

    def sanitize_lyrics(self) -> str:
        if self.is_instrumental:
            branch = self.instrumental_branch.lower()
            if branch == "cues":
                active_blocks = self.instrumental_blocks if (self.instrumental_blocks and len(self.instrumental_blocks) > 0) else self.blocks
                if active_blocks and len(active_blocks) > 0:
                    compiled = []
                    for b in active_blocks:
                        if not isinstance(b, dict):
                            continue
                        lbl = b.get("label") or b.get("type") or "verse"
                        txt = (b.get("text") or "").strip()
                        clean_lbl = re.sub(r"[\[\]]", "", str(lbl)).strip() or "verse"
                        if txt:
                            clean_txt = txt.strip()
                            if not (clean_txt.startswith("(") and clean_txt.endswith(")")):
                                clean_txt = f"({clean_txt})"
                            compiled.append(f"[{clean_lbl}]\n{clean_txt}")
                        else:
                            compiled.append(f"[{clean_lbl}]")
                    return normalize_lyrics("\n\n".join(compiled))

                active_raw = self.instrumental_lyrics if self.instrumental_lyrics.strip() else self.lyrics
                if active_raw and active_raw.strip():
                    lines = active_raw.replace("\r\n", "\n").splitlines()
                    compiled = []
                    for line in lines:
                        trimmed = line.strip()
                        if not trimmed:
                            continue
                        match = _LEADING_TAGS_RE.match(trimmed)
                        if match:
                            leading_tags = re.findall(r"\[[^\]]+\]", match.group(1))
                            for t in leading_tags:
                                compiled.append(t.strip())
                            remainder = trimmed[match.end():].strip()
                            if remainder:
                                if not (remainder.startswith("(") and remainder.endswith(")")):
                                    remainder = f"({remainder})"
                                compiled.append(remainder)
                        else:
                            if not (trimmed.startswith("(") and trimmed.endswith(")")):
                                compiled.append(f"({trimmed})")
                            else:
                                compiled.append(trimmed)
                    return normalize_lyrics("\n".join(compiled))

                return normalize_lyrics("[intro]\n\n[theme a]\n\n[verse]\n\n[chorus]\n\n[solo]\n\n[breakdown]\n\n[theme b]\n\n[outro]")

            elif branch == "tags_only":
                active_blocks = self.instrumental_blocks if (self.instrumental_blocks and len(self.instrumental_blocks) > 0) else self.blocks
                if active_blocks and len(active_blocks) > 0:
                    tags = []
                    for b in active_blocks:
                        if not isinstance(b, dict):
                            continue
                        lbl = b.get("label") or b.get("type") or "verse"
                        clean_lbl = re.sub(r"[\[\]]", "", str(lbl)).strip() or "verse"
                        tags.append(f"[{clean_lbl}]")
                    return normalize_lyrics("\n\n".join(tags))

                active_raw = self.instrumental_lyrics if self.instrumental_lyrics.strip() else self.lyrics
                if active_raw and active_raw.strip():
                    tags = [m.group(0) for m in re.finditer(r"\[([^\]]+)\]", active_raw)]
                    if tags:
                        return normalize_lyrics("\n\n".join(tags))

                return normalize_lyrics("[intro]\n\n[theme a]\n\n[verse]\n\n[chorus]\n\n[solo]\n\n[breakdown]\n\n[theme b]\n\n[outro]")

        if self.blocks and len(self.blocks) > 0:
            compiled_blocks = []
            for b in self.blocks:
                if not isinstance(b, dict):
                    continue
                lbl = b.get("label") or b.get("type") or "verse"
                txt = (b.get("text") or "").strip()
                clean_lbl = re.sub(r"[\[\]]", "", str(lbl)).strip() or "verse"
                compiled_blocks.append(f"[{clean_lbl}]\n{txt}" if txt else f"[{clean_lbl}]")
            return normalize_lyrics("\n\n".join(compiled_blocks))

        if self.lyrics and self.lyrics.strip():
            return normalize_lyrics(self.lyrics)

        return ""

    def validate(self) -> None:
        if self.audio_duration <= 0.0 or self.audio_duration > 600.0:
            raise ValueError(f"Duration {self.audio_duration}s out of bounds (0.0 < t <= 600.0s).")
        if self.bpm is not None and self.bpm != 0 and (self.bpm < 30 or self.bpm > 300):
            raise ValueError(f"BPM {self.bpm} out of practical range (30-300 or 0 for unmetered).")
        if self.early_instrumental_solver not in SUPPORTED_SOLVERS:
            raise ValueError(f"Early instrumental solver '{self.early_instrumental_solver}' invalid. Must be one of: {SUPPORTED_SOLVERS}")
        if self.late_instrumental_solver not in SUPPORTED_SOLVERS:
            raise ValueError(f"Late instrumental solver '{self.late_instrumental_solver}' invalid. Must be one of: {SUPPORTED_SOLVERS}")
        if self.early_vocal_solver not in SUPPORTED_SOLVERS:
            raise ValueError(f"Early vocal solver '{self.early_vocal_solver}' invalid. Must be one of: {SUPPORTED_SOLVERS}")
        if self.late_vocal_solver not in SUPPORTED_SOLVERS:
            raise ValueError(f"Late vocal solver '{self.late_vocal_solver}' invalid. Must be one of: {SUPPORTED_SOLVERS}")
        if self.handoff_threshold < 0.0 or self.handoff_threshold > 1.0:
            raise ValueError(f"Handoff threshold {self.handoff_threshold} out of bounds (0.0 <= t* <= 1.0).")
        if self.num_inference_steps is not None and (self.num_inference_steps < 1 or self.num_inference_steps > 200):
            raise ValueError(f"Inference steps {self.num_inference_steps} out of bounds (1-200).")
        if self.instrumental_guidance_scale is not None and (self.instrumental_guidance_scale < 0.0 or self.instrumental_guidance_scale > 20.0):
            raise ValueError(f"Instrumental guidance scale {self.instrumental_guidance_scale} out of bounds (0.0-20.0).")
        if self.early_instrumental_cfg is not None and (self.early_instrumental_cfg < 0.0 or self.early_instrumental_cfg > 20.0):
            raise ValueError(f"Early instrumental guidance scale {self.early_instrumental_cfg} out of bounds (0.0-20.0).")
        if self.late_instrumental_cfg is not None and (self.late_instrumental_cfg < 0.0 or self.late_instrumental_cfg > 20.0):
            raise ValueError(f"Late instrumental guidance scale {self.late_instrumental_cfg} out of bounds (0.0-20.0).")
        if self.vocal_guidance_scale is not None and (self.vocal_guidance_scale < 0.0 or self.vocal_guidance_scale > 20.0):
            raise ValueError(f"Vocal guidance scale {self.vocal_guidance_scale} out of bounds (0.0-20.0).")
        if self.early_vocal_cfg is not None and (self.early_vocal_cfg < 0.0 or self.early_vocal_cfg > 20.0):
            raise ValueError(f"Early vocal guidance scale {self.early_vocal_cfg} out of bounds (0.0-20.0).")
        if self.late_vocal_cfg is not None and (self.late_vocal_cfg < 0.0 or self.late_vocal_cfg > 20.0):
            raise ValueError(f"Late vocal guidance scale {self.late_vocal_cfg} out of bounds (0.0-20.0).")
        if self.eta is not None and (self.eta < 0.0 or self.eta > 1.0):
            raise ValueError(f"Eta {self.eta} out of bounds (0.0 <= eta <= 1.0).")
        if self.s_noise is not None and (self.s_noise < 0.0 or self.s_noise > 5.0):
            raise ValueError(f"s_noise {self.s_noise} out of bounds (0.0 <= s_noise <= 5.0).")
        if self.vocoder_batch_size < 1 or self.vocoder_batch_size > 32:
            raise ValueError(f"Vocoder batch size {self.vocoder_batch_size} out of bounds (1-32).")
        if self.ar_guidance_scale is not None and (self.ar_guidance_scale < 0.0 or self.ar_guidance_scale > 10.0):
            raise ValueError(f"AR Guidance scale {self.ar_guidance_scale} out of bounds (0.0-10.0).")
        if self.temperature is not None and (self.temperature <= 0.0 or self.temperature > 3.0):
            raise ValueError(f"Temperature {self.temperature} out of bounds (0.0 < T <= 3.0).")
        if self.top_p is not None and (self.top_p <= 0.0 or self.top_p > 1.0):
            raise ValueError(f"Top-P {self.top_p} out of bounds (0.0 < p <= 1.0).")
        if self.top_k is not None and (self.top_k < 1 or self.top_k > 500):
            raise ValueError(f"Top-K {self.top_k} out of bounds (1-500).")

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
    early_instrumental_solver_used: str
    late_instrumental_solver_used: str
    early_vocal_solver_used: str
    late_vocal_solver_used: str
    handoff_threshold_used: float
    early_steps: int
    late_steps: int
    total_nfe_chunk: int
    instrumental_guidance_scale_used: float
    early_instrumental_cfg_used: float
    late_instrumental_cfg_used: float
    vocal_guidance_scale_used: float
    early_vocal_cfg_used: float
    late_vocal_cfg_used: float
    eta_used: float
    s_noise_used: float
    effective_prompt: str
    declick_applied: bool
    peak_linear: float
    peak_dbfs: float
    rms_dbfs: float
    crest_factor_db: float
    top_k_vector_used: List[int]
    instrumental_scheduler_used: Optional[str] = None
    vocal_scheduler_used: Optional[str] = None
    is_instrumental_used: bool = False
    companion_instrumental_used: bool = False
    instrumental_branch_used: str = "cues"