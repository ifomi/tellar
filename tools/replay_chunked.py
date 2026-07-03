"""Replay a saved WAV through the silero-vad chunked pipeline offline.

Imitates pipeline.py at the level that determines the final text:
chunking + rolling prompt + transcribe_chunk + leading/trailing strip.
No threading, no queue, no recorder — feed the WAV bytes through
ChunkingBufferVAD.push() in 1024-sample frames (matching PyAudio's
typical frame size) and run transcribe_chunk synchronously on each
emitted chunk.

The point is A/B testing prompt strategies on the user's saved
problem dictations without re-recording. Three modes:
  --order prefix       : current production layout (PUNCTUATION_PROMPT + rolling)
  --order suffix       : experimental fix (rolling + PUNCTUATION_PROMPT)
  --order rolling_only : rolling prompt only (no PUNCTUATION_PROMPT at all)

Usage:
  python tools/replay_chunked.py path/to/sample.wav [more.wav ...] [--order suffix]
"""

import argparse
import re
import sys
import wave
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tellar.chunking_vad import ChunkingBufferVAD, TARGET_RATE
from tellar.hallucinations import remove_hallucinations
from tellar.transcriber import (
    MODEL_NAME,
    PUNCTUATION_PROMPT,
    _release_mlx_scratch,
    _set_hf_offline,
    _with_vocabulary,
    clean_hallucinations,
)


_LEADING_CONTINUATION_RE = re.compile(r'^\s*[.…]{2,}\s*')
_TRAILING_CONTINUATION_RE = re.compile(r'\s*[.…]{2,}\s*$')

ROLLING_PROMPT_CHARS = 200
FRAME_SAMPLES = 1024  # PyAudio default frame size in real recording
SAMPLE_WIDTH = 2  # int16


def load_wav_pcm16(path: str) -> bytes:
    with wave.open(path, 'rb') as wf:
        if wf.getframerate() != 16000 or wf.getnchannels() != 1 or wf.getsampwidth() != 2:
            raise ValueError(
                f"{path}: expected 16kHz mono int16, got "
                f"{wf.getframerate()}Hz {wf.getnchannels()}ch {wf.getsampwidth()*8}bit"
            )
        return wf.readframes(wf.getnframes())


def build_prompt(initial_prompt: Optional[str], order: str) -> str:
    # Variant of PUNCTUATION_PROMPT without its trailing period. When
    # used as a suffix, the trailing "." cues the decoder to start a new
    # sentence — capitalizing the first word of the chunk even when it's
    # syntactically a continuation of the previous chunk. Stripping the
    # period leaves the punctuation pattern intact (commas, question
    # marks, internal periods) but doesn't force a fresh-sentence start.
    punct_open = PUNCTUATION_PROMPT.rstrip('. ').rstrip()

    if not initial_prompt:
        return _with_vocabulary(PUNCTUATION_PROMPT)
    if order == 'prefix':
        return PUNCTUATION_PROMPT + ' ' + initial_prompt
    if order == 'suffix':
        return initial_prompt + ' ' + PUNCTUATION_PROMPT
    if order == 'suffix_open':
        return initial_prompt + ' ' + punct_open
    if order == 'rolling_only':
        return initial_prompt
    raise ValueError(f"unknown order: {order}")


def transcribe_chunk_replay(
    audio: np.ndarray,
    initial_prompt: Optional[str],
    order: str,
) -> str:
    if len(audio) == 0:
        return ''
    _set_hf_offline(True)
    import mlx_whisper

    prompt = build_prompt(initial_prompt, order)
    result = mlx_whisper.transcribe(
        audio,
        path_or_hf_repo=MODEL_NAME,
        initial_prompt=prompt,
        no_speech_threshold=0.5,
        compression_ratio_threshold=2.4,
        temperature=0.0,
    )
    text = (result.get('text') or '').strip()
    _release_mlx_scratch()
    return remove_hallucinations(clean_hallucinations(text))


def replay(wav_path: str, order: str) -> List[Tuple[int, str, float, str]]:
    """Run the full chunking + transcribe sequence on a single WAV.
    Returns list of (idx, cut_reason, duration_s, chunk_text)."""
    pcm_bytes = load_wav_pcm16(wav_path)
    bytes_per_frame = FRAME_SAMPLES * SAMPLE_WIDTH

    buf = ChunkingBufferVAD(source_rate=16000)
    out: List[Tuple[int, str, float, str]] = []
    last_prompt: Optional[str] = None

    def handle_chunk(chunk_audio: np.ndarray, reason: str, is_full: bool):
        nonlocal last_prompt
        duration = len(chunk_audio) / TARGET_RATE
        text = transcribe_chunk_replay(chunk_audio, last_prompt, order)
        text = _LEADING_CONTINUATION_RE.sub('', text)
        if is_full:
            text = _TRAILING_CONTINUATION_RE.sub('', text)
        idx = len(out)
        out.append((idx, reason, duration, text))
        if text.strip():
            last_prompt = text[-ROLLING_PROMPT_CHARS:]
        else:
            last_prompt = None

    for i in range(0, len(pcm_bytes), bytes_per_frame):
        frame_bytes = pcm_bytes[i:i + bytes_per_frame]
        for chunk_audio, reason in buf.push(frame_bytes):
            handle_chunk(chunk_audio, reason, is_full=True)

    tail = buf.flush()
    if len(tail) > 0:
        handle_chunk(tail, 'tail', is_full=False)

    return out


def format_with_markers(chunks: List[Tuple[int, str, float, str]]) -> str:
    parts: List[str] = []
    for idx, reason, duration, text in chunks:
        if text:
            parts.append(text)
        if idx < len(chunks) - 1:
            parts.append(f"⟨✂{idx}: {duration:.2f}s {reason}⟩")
    return ' '.join(parts)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('wavs', nargs='+')
    parser.add_argument('--order', choices=['prefix', 'suffix', 'suffix_open', 'rolling_only'], default='prefix')
    args = parser.parse_args()

    for wav in args.wavs:
        print(f"\n{'=' * 88}")
        print(f"  {Path(wav).name}    [order={args.order}]")
        print('=' * 88)
        try:
            chunks = replay(wav, args.order)
        except Exception as e:
            print(f"FAILED: {e}")
            continue
        for idx, reason, duration, text in chunks:
            print(f"  ✂{idx} ({duration:5.2f}s, {reason:>9}): {text}")
        print()
        print('Final assembled:')
        print(format_with_markers(chunks))
        print()


if __name__ == '__main__':
    main()
