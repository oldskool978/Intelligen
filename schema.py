import re
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Dict, Any, List

from pipeline.prompt_compiler import clean_caption, normalize_lyrics

DEFAULT_LYRICS = """[intro]
(Smooth Rhodes chords, filtered 808 glide, ad-libs)
Yeah, listen
Midnight in the city, let the groove breathe
Oh, oh-woah, yeah

[verse 1]
Midnight riding under neon streetlights
Searching for the answers in the rearview mirror
Thought I had the blueprint solid in my mind
Now the silhouette of you is drawing nearer
Dashboard glowing with a steady slow pulse
Echoes of your whisper in the night air

[pre-chorus 1]
I try to fight it, but it's pulling me in
Every harmonic frequency starts spinning again
Tension rising from the bottom to top
Got that momentum and we never gon' stop

[chorus 1]
Got me caught up in the way that you move
Nobody else can lock right into the groove
Got my heart on the floor, baby, give me one more
Show me that rhythm, tell me what you wanna do
(Yeah, yeah, keep it right there)

[verse 2]
Two in the morning, baseline taking over
Sip of something smooth, leaning in a little closer
Sub-frequencies vibrating the floor
You give me everything, but I still want more
Syncopated touch, perfect timing on the beat
Fire in our eyes, generating pure heat

[pre-chorus 2]
I try to fight it, but it's pulling me in
Every harmonic frequency starts spinning again
Tension rising from the bottom to top
Got that momentum and we never gon' stop

[chorus 2]
Got me caught up in the way that you move
Nobody else can lock right into the groove
Got my heart on the floor, baby, give me one more
Show me that rhythm, tell me what you wanna do
(Yeah, yeah, right into the pocket)

[bridge]
Take it to the falsetto high, let the bass drop clean
Smoothest vibration that you've ever seen
Counterpoint melodies weaving around
Elevating the pressure, capturing the sound
Hold that note, let the energy soar
Take it to places that we never went before

[solo]
(Warm expressive nylon and electric guitar soloing over deep sub-bass and syncopated percussion)

[chorus 3]
Got me caught up in the way that you move
Nobody else can lock right into the groove
Got my heart on the floor, baby, give me one more
Show me that rhythm, tell me what you wanna do
(Oh-woah, give me one more time)

[outro]
Fade into the low-end frequency
Keep the drum pocket steady for me
Ad-libs drifting out into the night
Yeah, just like that
Fade to black"""

DEFAULT_PROMPT = (
    "Basic Attributes: bpm is 96. key is F, and scale is minor. Contemporary R&B / 2000s Pop R&B / Slow Jam Bounce. "
    "Mood: Sensual, passionate, smooth, confident, driving. "
    "Vocals: Silky male tenor lead vocal, dynamic chest-to-falsetto transitions, intricate melismatic ad-libs, stacked 4-part harmonies. "
    "Arrangement: Deep 808 sub-bass, crisp acoustic-electronic hybrid snare on 2 and 4, syncopated hi-hat rolls, warm Fender Rhodes electric piano chords, acoustic nylon guitar plucks."
)

SUPPORTED_SCHEDULERS = ["native", "euler", "heun"]
SUPPORTED_NOISE_TOPOLOGIES = ["gaussian", "blue_noise"]


@dataclass
class GenerationRequest:
    genre: str = "Contemporary R&B"
    subgenre: str = "2000s Pop R&B / Slow Jam Bounce"
    bpm: int = 96
    key: str = "F minor"
    mood: str = "Sensual, passionate, smooth, confident, driving."
    vocals: str = "Silky male tenor lead vocal, dynamic chest-to-falsetto transitions, intricate melismatic ad-libs, stacked 4-part harmonies."
    arrangement: str = "Deep 808 sub-bass, crisp acoustic-electronic hybrid snare on 2 and 4, syncopated hi-hat rolls, warm Fender Rhodes chords."
    raw_prompt: Optional[str] = None
    prompt: Optional[str] = None
    lyrics: str = DEFAULT_LYRICS
    temperature: Optional[float] = 0.91
    top_p: Optional[float] = 0.96
    top_k: Optional[int] = 44
    ar_guidance_scale: Optional[float] = 1.52
    scheduler_type: str = "heun"
    num_inference_steps: Optional[int] = 42
    guidance_scale: Optional[float] = 1.78
    noise_topology: str = "blue_noise"
    blue_noise_alpha: float = 0.75
    enable_pm_diffusion: bool = True
    pm_iterations: int = 5
    pm_conductance: float = 0.15
    pm_lambda: float = 0.20
    audio_duration: float = 300.0
    seed: int = 42
    output_path: str = "output.wav"
    repo_id: str = "MiniMaxAI/MiniMax-Music3"
    device: str = "cuda"
    apply_declick: bool = True
    cpu_offload: bool = False
    blocks: Optional[List[Dict[str, Any]]] = None

    def compile_prompt(self) -> str:
        candidate_prompt = self.prompt or self.raw_prompt
        if candidate_prompt and candidate_prompt.strip():
            return clean_caption(candidate_prompt.strip())

        key_clean = self.key.strip() if self.key else ""
        key_root = "F"
        scale_mode = "minor"
        if key_clean:
            key_match = re.match(r"^([A-G][b#]?)\s*(major|minor|m)?", key_clean, re.IGNORECASE)
            if key_match:
                key_root = key_match.group(1).upper()
                if len(key_root) > 1 and key_root[1] == "B":
                    key_root = key_root[0] + "b"
                mode_token = (key_match.group(2) or "").lower()
                scale_mode = "major" if mode_token == "major" else "minor"

        attr_parts = []
        if self.bpm and self.bpm > 0:
            attr_parts.append(f"bpm is {self.bpm}")
        if key_clean:
            attr_parts.append(f"key is {key_root}, and scale is {scale_mode}")
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

        return clean_caption(" ".join(segments))

    def sanitize_lyrics(self) -> str:
        if self.blocks and not (self.lyrics and self.lyrics.strip()):
            compiled_blocks = []
            for b in self.blocks:
                lbl = b.get("label") or b.get("type") or "verse"
                txt = (b.get("text") or "").strip()
                compiled_blocks.append(f"[{lbl.strip()}]\n{txt}" if txt else f"[{lbl.strip()}]")
            return normalize_lyrics("\n\n".join(compiled_blocks))
        return normalize_lyrics(self.lyrics)

    def validate(self) -> None:
        if self.audio_duration <= 0.0 or self.audio_duration > 600.0:
            raise ValueError(f"Duration {self.audio_duration}s out of bounds (0.0 < t <= 600.0s).")
        if self.bpm is not None and (self.bpm < 30 or self.bpm > 300):
            raise ValueError(f"BPM {self.bpm} out of practical range (30-300).")
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
    crest_factor_db: float
    scheduler_used: str
    noise_topology_used: str
    pm_diffusion_used: bool
    effective_prompt: str
    declick_applied: bool
    cpu_offload_active: bool
    peak_vram_gb: float