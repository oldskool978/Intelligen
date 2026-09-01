import re
from typing import Optional, Tuple
import torch
import torch.nn.functional as F

_IM_START = "<|im_start|>"
_IM_END = "<|im_end|>"
_CAPTION_START = "<|caption_start|>"
_CAPTION_END = "<|caption_end|>"
_LYRICS_START = "<|lyrics_start|>"
_LYRICS_END = "<|lyrics_end|>"
_AUDIO_START = "<|audio_start|>"
_AUDIO_END_TOKEN_ID = 151670
_AUDIO_CFG_TOKEN_ID = 151654
_AUDIO_CODE_OFFSET = 151675
_SEMANTIC_VOCAB_SIZE = 16384
_MAX_PROMPT_TOKENS = 5_000

_SPECIAL_TAG_RE = re.compile(r"<\|([^|]*)\|>")
_INLINE_TAG_RE = re.compile(r"\[([^\]]+)\]")


def clean_caption(caption: str) -> str:
    def _rewrite_special_tag(match: re.Match) -> str:
        inner = match.group(1).strip()
        parts = inner.split(None, 1)
        return f"{parts[0]} is {parts[1]}" if len(parts) == 2 else inner

    text = _SPECIAL_TAG_RE.sub(_rewrite_special_tag, caption)
    lines_out = []
    for line in text.splitlines():
        line = re.sub(r"^\s{0,3}#{1,6}\s+", "", line)
        line = re.sub(r"^\s*[*+-]\s+", "", line)
        line = re.sub(r"^\s*\*\s+", "", line)
        while "**" in line:
            updated = re.sub(r"\*\*([^*]+)\*\*", r"\1", line)
            if updated == line:
                break
            line = updated
        line = re.sub(r"(?<!\*)\*([^*\n]+)\*(?!\*)", r"\1", line)
        lines_out.append(line.rstrip())
    text = "\n".join(lines_out)
    text = re.sub(r"^\s*[-*_]{3,}\s*$", "", text, flags=re.MULTILINE)
    text = text.replace("  ", " ").replace("    ", " ")
    return re.sub(r"\n{2,}", "\n", text).strip()


def normalize_lyrics(lyrics: str) -> str:
    if not isinstance(lyrics, str) or not lyrics.strip():
        return "[start]\n[intro]\n[verse]\n[chorus]\n[outro]"

    raw_text = lyrics.replace("\r\n", "\n").replace("\r", "\n")
    raw_text = raw_text.replace(" ^ ", "\n")

    # Safely separate inline tags onto their own lines without losing lyrics
    def _expand_bracket(match: re.Match) -> str:
        content = match.group(1).strip()
        cleaned_content = re.sub(r"\s+", " ", content).lower()
        return f"\n[{cleaned_content}]\n"

    expanded = _INLINE_TAG_RE.sub(_expand_bracket, raw_text)

    # Clean empty lines and structure sections
    cleaned_lines = []
    for line in expanded.splitlines():
        line_clean = line.strip()
        if not line_clean:
            continue
        cleaned_lines.append(line_clean)

    text = "\n".join(cleaned_lines)
    # Ensure double newlines between tags/verses for clear tokenization boundaries
    text = re.sub(r"(\[[^\]]+\])", r"\n\1\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    if not text.startswith("[start]"):
        text = f"[start]\n{text}"
    return text


def build_text_ids(tokenizer, prompt: str, lyrics: str, device: torch.device) -> torch.Tensor:
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("`prompt` must be a non-empty string.")
    if not isinstance(lyrics, str) or not lyrics.strip():
        raise ValueError("`lyrics` must be a non-empty string.")

    cleaned_p = clean_caption(prompt)
    normalized_l = normalize_lyrics(lyrics)

    formatted_text = (
        f"{_IM_START}{_CAPTION_START}{cleaned_p}{_CAPTION_END}"
        f"{_LYRICS_START}{normalized_l}{_LYRICS_END}{_IM_END}{_AUDIO_START}"
    )

    input_ids = tokenizer(formatted_text, return_tensors="pt")["input_ids"]
    if input_ids.shape[1] > _MAX_PROMPT_TOKENS:
        raise ValueError(f"Assembled prompt exceeds {_MAX_PROMPT_TOKENS} tokens ({input_ids.shape[1]} tokens).")

    unconditional_ids = input_ids.clone()
    unconditional_ids[:, 1:-2] = _AUDIO_CFG_TOKEN_ID
    return torch.cat((input_ids, unconditional_ids), dim=0).to(device)