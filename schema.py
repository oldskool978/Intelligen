# schema.py
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Dict, Any, List

DEFAULT_LYRICS = """[verse]
Morning light filtering through the pine
Every quiet street is yours and mine
[chorus]
Softly the world begins to breathe
"""

SUPPORTED_SCHEDULERS = ["euler", "heun"]
SUPPORTED_NOISE_TOPOLOGIES = ["gaussian", "blue_noise", "perona_malik"]

@dataclass
class GenerationRequest:
    genre: str = "acoustic pop"
    bpm: int = 96
    key: str = "C major"
    mood: str = "Warm and intimate, building gently into the chorus."
    vocals: str = "soft female lead, close and breathy, light stacked harmonies in the chorus."
    arrangement: str = "fingerpicked guitar and soft piano; brushed drums and upright bass enter in the chorus."
    raw_prompt: Optional[str] = None
    lyrics: str = DEFAULT_LYRICS
    
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 50
    
    scheduler_type: str = "euler"
    num_inference_steps: int = 32
    guidance_scale: float = 4.5
    
    noise_topology: str = "gaussian"
    blue_noise_alpha: float = 0.75
    pm_iterations: int = 5
    pm_conductance: float = 0.15
    pm_lambda: float = 0.20
    
    audio_duration: float = 30.0
    seed: int = 42
    output_path: str = "output.wav"
    repo_id: str = "MiniMaxAI/MiniMax-Music3"
    device: str = "cuda"

    def compile_prompt(self) -> str:
        if self.raw_prompt and self.raw_prompt.strip():
            return self.raw_prompt.strip()
        
        segments = []
        if self.genre.strip():
            segments.append(f"Genre: {self.genre.strip()}.")
        if self.bpm is not None and self.bpm > 0:
            segments.append(f"BPM: {self.bpm}.")
        if self.key.strip():
            segments.append(f"Key: {self.key.strip()}.")
        if self.mood.strip():
            mood_str = self.mood.strip()
            if not mood_str.endswith("."):
                mood_str += "."
            segments.append(mood_str)
        if self.vocals.strip():
            segments.append(f"Vocals: {self.vocals.strip()}.")
        if self.arrangement.strip():
            segments.append(f"Arrangement: {self.arrangement.strip()}.")
            
        return " ".join(segments)

    def sanitize_lyrics(self) -> str:
        lines = [line.strip() for line in self.lyrics.strip().splitlines() if line.strip()]
        return "\n".join(lines)

    def validate(self) -> None:
        if self.audio_duration <= 0.0 or self.audio_duration > 47.5:
            raise ValueError(f"Duration {self.audio_duration}s out of bounds (0.0 < t <= 47.5s).")
        if self.bpm is not None and (self.bpm < 30 or self.bpm > 300):
            raise ValueError(f"BPM {self.bpm} out of practical range (30-300).")
        if self.scheduler_type not in SUPPORTED_SCHEDULERS:
            raise ValueError(f"Scheduler '{self.scheduler_type}' invalid. Must be one of: {SUPPORTED_SCHEDULERS}")
        if self.num_inference_steps < 1 or self.num_inference_steps > 200:
            raise ValueError(f"Inference steps {self.num_inference_steps} out of bounds (1-200).")
        if self.guidance_scale < 0.0 or self.guidance_scale > 20.0:
            raise ValueError(f"Guidance scale {self.guidance_scale} out of bounds (0.0-20.0).")
        if self.temperature <= 0.0 or self.temperature > 3.0:
            raise ValueError(f"Temperature {self.temperature} out of bounds (0.0 < T <= 3.0).")
        if self.top_p <= 0.0 or self.top_p > 1.0:
            raise ValueError(f"Top-P {self.top_p} out of bounds (0.0 < p <= 1.0).")
        if self.top_k < 1 or self.top_k > 500:
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

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GenerationRequest":
        valid_keys = cls.__dataclass_fields__.keys()
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)

    def save_preset(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load_preset(cls, path: Path) -> "GenerationRequest":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

@dataclass
class GenerationResponse:
    output_path: str
    sample_rate: int
    total_samples: int
    duration_seconds: float
    generation_time_seconds: float
    real_time_factor: float
    peak_linear: float
    peak_dbfs: float
    rms_dbfs: float
    scheduler_used: str
    nfe_count: int
    noise_topology_used: str
    effective_prompt: str