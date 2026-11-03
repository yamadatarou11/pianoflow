from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import math
import os
import unicodedata
from pathlib import Path


TRANSCRIPTION_VERSION = 12

FORBIDDEN_ASR_PHRASES = (
    "ご視聴ありがとうございました",
    "ご視聴ありがとうございます",
    "ご視聴いただきありがとうございました",
    "チャンネル登録よろしくお願いします",
)


def _remove_forbidden_asr_phrases(text: str) -> str:
    """Remove known Whisper outro hallucinations from every public ASR path."""
    cleaned = str(text or "")
    for phrase in FORBIDDEN_ASR_PHRASES:
        cleaned = cleaned.replace(phrase, "")
    return cleaned.strip(" \t\r\n。、！!？?")


def _katakana_to_hiragana(value: str) -> str:
    converted = []
    for character in value:
        code = ord(character)
        converted.append(chr(code - 0x60) if 0x30A1 <= code <= 0x30F6 else character)
    return "".join(converted)


def _to_hiragana(text: str) -> str:
    """Convert Japanese orthography to the pronunciation used for alignment."""
    try:
        from janome.tokenizer import Tokenizer

        parts = []
        for token in Tokenizer().tokenize(unicodedata.normalize("NFKC", text)):
            reading = str(getattr(token, "reading", "") or "")
            parts.append(token.surface if reading == "*" or not reading else reading)
        value = _katakana_to_hiragana("".join(parts))
    except Exception:
        value = _katakana_to_hiragana(unicodedata.normalize("NFKC", text))
    return "".join(character for character in value if character.isalnum() or "ぁ" <= character <= "ゖ" or character == "ー")


def _moras(value: str) -> list[str]:
    units: list[str] = []
    small = set("ゃゅょぁぃぅぇぉゎ")
    for character in value:
        if character in small and units:
            units[-1] += character
        elif character == "ー" and units:
            units[-1] += character
        else:
            units.append(character)
    return units


def _align_supplied_lyrics(result: dict, lyrics_text: str) -> dict:
    supplied_hiragana = _to_hiragana(lyrics_text)
    words = [dict(row) for row in result.get("words", [])]
    cursor = 0
    for word in words:
        word["hiragana"] = _to_hiragana(str(word.get("text", "")))
        word["hiraganaStart"] = cursor
        cursor += len(word["hiragana"])
        word["hiraganaEnd"] = cursor
    recognized_hiragana = "".join(str(word["hiragana"]) for word in words)
    matcher = difflib.SequenceMatcher(None, recognized_hiragana, supplied_hiragana, autojunk=False)
    recognized_to_reference: dict[int, int] = {}
    matched_characters = 0
    for tag, left_a, right_a, left_b, right_b in matcher.get_opcodes():
        if tag != "equal":
            continue
        for offset in range(right_a - left_a):
            recognized_to_reference[left_a + offset] = left_b + offset
            matched_characters += 1
    aligned_words: list[dict] = []
    for word in words:
        mapped = [recognized_to_reference[index] for index in range(int(word["hiraganaStart"]), int(word["hiraganaEnd"])) if index in recognized_to_reference]
        if mapped:
            reference_hiragana = supplied_hiragana[min(mapped):max(mapped) + 1]
            match_confidence = len(mapped) / max(1, len(str(word["hiragana"])), len(reference_hiragana))
        else:
            reference_hiragana = str(word["hiragana"])
            match_confidence = 0.0
        word["referenceHiragana"] = reference_hiragana
        word["lyricsMatchConfidence"] = round(float(match_confidence), 4)
        units = _moras(reference_hiragana)
        start, end = float(word.get("start", 0.0)), float(word.get("end", 0.0))
        span = max(0.001, end - start)
        for index, unit in enumerate(units):
            aligned_words.append({
                "start": round(start + span * index / max(1, len(units)), 4),
                "end": round(start + span * (index + 1) / max(1, len(units)), 4),
                "text": unit,
                "hiragana": unit,
                "confidence": round(float(max(0.12, float(word.get("confidence", 0.0)) * (0.45 + 0.55 * match_confidence))), 4),
                "lyricsMatchConfidence": round(float(match_confidence), 4),
                "source": "supplied-lyrics" if match_confidence >= 0.5 else "recognized-hiragana",
            })
    updated = dict(result)
    updated["words"] = words
    updated["textHiragana"] = recognized_hiragana
    updated["suppliedLyrics"] = lyrics_text
    updated["suppliedLyricsHiragana"] = supplied_hiragana
    similarity = float(matcher.ratio())
    # A supplied lyric must never force unrelated ASR text onto the audio.
    # Below this threshold, retain the observed word timestamps and let the
    # independent acoustic-boundary detector decide all note splits.
    updated["alignmentWords"] = aligned_words if similarity >= 0.45 else []
    updated["lyricsAlignmentAccepted"] = similarity >= 0.45
    updated["lyricsComparison"] = {
        "recognizedHiragana": recognized_hiragana,
        "suppliedHiragana": supplied_hiragana,
        "matchedCharacters": matched_characters,
        "recognizedCharacters": len(recognized_hiragana),
        "suppliedCharacters": len(supplied_hiragana),
        "similarity": round(similarity, 4),
        "method": "Janome reading normalization + ordered hiragana sequence alignment",
    }
    return updated


def _merge_japanese_words(text: str, raw_words: list[dict], segment_id: int) -> list[dict]:
    """Map Whisper's character-like Japanese tokens onto lexical words."""

    if not raw_words:
        return []
    try:
        from janome.tokenizer import Tokenizer

        surfaces = [token.surface.replace(" ", "") for token in Tokenizer().tokenize(text) if token.surface.strip()]
    except Exception:
        return raw_words
    compact_parts = [str(row["text"]).replace(" ", "") for row in raw_words]
    compact = "".join(compact_parts)
    spans: list[tuple[int, int, dict]] = []
    position = 0
    for part, row in zip(compact_parts, raw_words):
        spans.append((position, position + len(part), row))
        position += len(part)
    merged: list[dict] = []
    cursor = 0
    for surface in surfaces:
        start_char = compact.find(surface, cursor)
        if start_char < 0:
            continue
        end_char = start_char + len(surface)
        cursor = end_char
        covered = [row for left, right, row in spans if right > start_char and left < end_char]
        if not covered or all(unicodedata.category(char).startswith("P") for char in surface):
            continue
        weights = [max(0.001, float(row["end"]) - float(row["start"])) for row in covered]
        confidence = sum(float(row["confidence"]) * weight for row, weight in zip(covered, weights)) / sum(weights)
        merged.append({
            "start": round(float(covered[0]["start"]), 3),
            "end": round(float(covered[-1]["end"]), 3),
            "text": surface,
            "confidence": round(confidence, 4),
            "segmentId": segment_id,
        })
    return merged or raw_words


def _add_context_scores(words: list[dict], segments: list[dict]) -> list[dict]:
    """Score each lexical unit using decoder, segment and neighbouring-word context."""

    by_segment = {int(row["id"]): row for row in segments}
    scored: list[dict] = []
    for index, original in enumerate(words):
        row = dict(original)
        start, end = float(row["start"]), float(row["end"])
        duration = max(0.0, end - start)
        base = float(row.get("confidence", 0.0))
        segment = by_segment.get(int(row.get("segmentId", -1)), {})
        segment_confidence = float(segment.get("confidence", base))

        neighbour_scores: list[float] = []
        for neighbour_index in (index - 1, index + 1):
            if not 0 <= neighbour_index < len(words):
                continue
            neighbour = words[neighbour_index]
            if neighbour_index < index:
                gap = max(0.0, start - float(neighbour["end"]))
            else:
                gap = max(0.0, float(neighbour["start"]) - end)
            neighbour_scores.append(math.exp(-gap / 0.9))
        neighbour_context = sum(neighbour_scores) / len(neighbour_scores) if neighbour_scores else 0.55
        duration_score = 1.0 if 0.055 <= duration <= 1.8 else (0.55 if duration <= 2.8 else 0.2)
        repeated = any(
            str(words[other].get("text", "")) == str(row.get("text", ""))
            for other in (index - 1, index + 1)
            if 0 <= other < len(words)
        )
        repetition_score = 0.72 if repeated else 1.0
        context_score = (
            0.46 * base
            + 0.24 * segment_confidence
            + 0.18 * neighbour_context
            + 0.12 * duration_score
        ) * repetition_score
        row["contextConfidence"] = round(float(max(0.0, min(1.0, context_score))), 4)
        scored.append(row)
    return scored


def _transcribe_audio_only_kotoba(source: Path) -> dict:
    """Decode Japanese audio without any lyric, prompt, or expected phrase."""
    import librosa
    import numpy as np
    import soundfile as sf
    import torch
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

    model_name = "kotoba-tech/kotoba-whisper-v2.0"
    model_root = Path(__file__).resolve().parent / "models" / "transformers"
    model_root.mkdir(parents=True, exist_ok=True)
    use_cuda = bool(torch.cuda.is_available())
    dtype = torch.float16 if use_cuda else torch.float32
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_name,
        dtype=dtype,
        low_cpu_mem_usage=True,
        cache_dir=str(model_root),
        local_files_only=True,
    )
    processor = AutoProcessor.from_pretrained(model_name, cache_dir=str(model_root), local_files_only=True)
    recognizer = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        dtype=dtype,
        device=0 if use_cuda else -1,
    )
    audio, sample_rate = sf.read(str(source), dtype="float32", always_2d=False)
    if getattr(audio, "ndim", 1) > 1:
        audio = np.mean(audio, axis=1)
    if int(sample_rate) != 16000:
        audio = librosa.resample(np.asarray(audio, dtype=np.float32), orig_sr=int(sample_rate), target_sr=16000)
    decoded = recognizer(
        {"array": np.asarray(audio, dtype=np.float32), "sampling_rate": 16000},
        # Ten seconds retained fast syllables best in the audio-only singing
        # benchmark; longer windows duplicated phrases and invented words.
        chunk_length_s=10,
        batch_size=4 if use_cuda else 1,
        generate_kwargs={"language": "japanese", "task": "transcribe"},
    )
    text = _remove_forbidden_asr_phrases(str(decoded.get("text", "")))
    return {
        "available": bool(text),
        "model": model_name,
        "text": text,
        "textHiragana": _to_hiragana(text),
        "inputRule": "audio only; no supplied lyrics, prompt, or expected phrase",
    }


def transcribe(source: Path, output_dir: Path, model_name: str, lyrics_path: Path | None = None) -> dict:
    cache_path = output_dir / "transcription.json"
    lyrics_text = lyrics_path.read_text(encoding="utf-8").strip() if lyrics_path and lyrics_path.exists() else ""
    lyrics_hash = hashlib.sha256(lyrics_text.encode("utf-8")).hexdigest() if lyrics_text else ""
    if cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if (
                cached.get("version") == TRANSCRIPTION_VERSION
                and cached.get("model") == model_name
                and cached.get("sourceName") == source.name
                and cached.get("sourceSize") == source.stat().st_size
                and cached.get("lyricsHash", "") == lyrics_hash
            ):
                return cached
        except (OSError, ValueError):
            pass

    import ctranslate2
    from faster_whisper import WhisperModel

    output_dir.mkdir(parents=True, exist_ok=True)
    model_root = Path(__file__).resolve().parent / "models" / "faster-whisper"
    model_root.mkdir(parents=True, exist_ok=True)
    device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    try:
        model = WhisperModel(model_name, device=device, compute_type=compute_type, download_root=str(model_root))
    except Exception:
        device, compute_type = "cpu", "int8"
        model = WhisperModel(model_name, device=device, compute_type=compute_type, download_root=str(model_root))

    generated, info = model.transcribe(
        str(source),
        language="ja",
        task="transcribe",
        beam_size=8,
        best_of=8,
        patience=1.2,
        temperature=0.0,
        word_timestamps=True,
        vad_filter=False,
        # The decoder evaluates each word against the preceding phrase.  The
        # later acoustic gate prevents this language context from inventing notes.
        condition_on_previous_text=True,
        repetition_penalty=1.08,
        no_repeat_ngram_size=3,
        # Reference lyrics are never allowed into free transcription.
        initial_prompt=None,
    )
    segments = list(generated)
    words: list[dict] = []
    segment_rows: list[dict] = []
    for segment_id, segment in enumerate(segments):
        text = _remove_forbidden_asr_phrases(segment.text.strip())
        if not text:
            continue
        segment_confidence = float(max(0.0, min(1.0, math.exp(float(segment.avg_logprob)))))
        segment_rows.append({
            "id": segment_id,
            "start": round(float(segment.start), 3),
            "end": round(float(segment.end), 3),
            "text": text,
            "confidence": round(segment_confidence, 4),
            "noSpeechProbability": round(float(getattr(segment, "no_speech_prob", 0.0)), 4),
            "compressionRatio": round(float(getattr(segment, "compression_ratio", 0.0)), 4),
        })
        raw_segment_words: list[dict] = []
        if segment.words:
            for word in segment.words:
                token = word.word.strip()
                if not token or word.start is None or word.end is None:
                    continue
                raw_segment_words.append({
                    "start": round(float(word.start), 3),
                    "end": round(float(word.end), 3),
                    "text": token,
                    "confidence": round(float(max(0.0, min(1.0, word.probability))), 4),
                    "segmentId": segment_id,
                })
            words.extend(_merge_japanese_words(text, raw_segment_words, segment_id))
        elif text:
            words.append({
                "start": round(float(segment.start), 3),
                "end": round(float(segment.end), 3),
                "text": text,
                "confidence": round(float(max(0.0, min(1.0, 1.0 + segment.avg_logprob))), 4),
                "segmentId": segment_id,
            })
    words = _add_context_scores(words, segment_rows)
    result = {
        "available": True,
        "version": TRANSCRIPTION_VERSION,
        "model": model_name,
        "sourceName": source.name,
        "sourceSize": source.stat().st_size,
        "device": device,
        "language": getattr(info, "language", "ja"),
        "languageProbability": round(float(getattr(info, "language_probability", 0.0)), 4),
        "lyricsHash": lyrics_hash,
        "text": "".join(row["text"] for row in segment_rows),
        "segments": segment_rows,
        "words": words,
    }
    try:
        audio_only = _transcribe_audio_only_kotoba(source)
    except Exception as error:
        audio_only = {
            "available": bool(result["text"]),
            "model": model_name,
            "text": _remove_forbidden_asr_phrases(result["text"]),
            "textHiragana": _to_hiragana(_remove_forbidden_asr_phrases(result["text"])),
            "inputRule": "audio only; no supplied lyrics, prompt, or expected phrase",
            "fallback": True,
            "error": str(error)[-1200:],
        }
    if lyrics_text:
        truth_hiragana = _to_hiragana(lyrics_text)
        audio_only["similarity"] = round(float(difflib.SequenceMatcher(
            None, str(audio_only.get("textHiragana", "")), truth_hiragana, autojunk=False
        ).ratio()), 4)
        audio_only["evaluationRule"] = "reference lyrics used only after decoding"
    result["audioOnlyTranscription"] = audio_only
    if lyrics_text:
        result = _align_supplied_lyrics(result, lyrics_text)
        try:
            from lyrics_forced_aligner import align as force_align_lyrics

            forced = force_align_lyrics(source, lyrics_text, Path(__file__).resolve().parent / "models" / "torch")
            if forced.get("available"):
                rejected_free_transcription = {
                    "text": audio_only.get("text", ""),
                    "textHiragana": audio_only.get("textHiragana", ""),
                    "similarity": float(audio_only.get("similarity", 0.0)),
                    "reason": "free singing ASR disagreed with supplied lyrics and was excluded from display and boundaries",
                }
                result["forcedAlignment"] = {
                    key: value for key, value in forced.items()
                    if key not in {"words", "textMatch", "hiraganaMatchRate"}
                }
                result["alignmentWords"] = forced["words"]
                result["lyricsAlignmentAccepted"] = True
                result["lyricsAlignmentMethod"] = forced["method"]
                result["constrainedText"] = lyrics_text
                result["constrainedTextHiragana"] = forced["textHiragana"]
                result["transcriptionMode"] = "audio-only transcription plus separate supplied-lyrics forced alignment"
                result["asrLyricsRejected"] = float(result.get("lyricsComparison", {}).get("similarity", 0.0)) < 0.999
                result["rejectedFreeTranscription"] = rejected_free_transcription
                result["text"] = lyrics_text
                result["textHiragana"] = forced["textHiragana"]
                result["words"] = []
                result["segments"] = []
        except Exception as error:
            result["forcedAlignmentError"] = str(error)[-1200:]
    else:
        result["textHiragana"] = _to_hiragana(result["text"])
        for word in result["words"]:
            word["hiragana"] = _to_hiragana(str(word.get("text", "")))
    cache_path.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default=os.environ.get("PIANOFLOW_WHISPER_MODEL", "large-v3-turbo"))
    parser.add_argument("--lyrics-file", type=Path)
    args = parser.parse_args()
    result = transcribe(args.source.resolve(), args.output_dir.resolve(), args.model, args.lyrics_file.resolve() if args.lyrics_file else None)
    print(json.dumps(result, ensure_ascii=True))


if __name__ == "__main__":
    main()
