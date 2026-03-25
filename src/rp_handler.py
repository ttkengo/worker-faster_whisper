import base64
import tempfile
import os
import yt_dlp
from rp_schema import INPUT_VALIDATIONS
from runpod.serverless.utils import download_files_from_urls, rp_cleanup, rp_debugger
from runpod.serverless.utils.rp_validator import validate
import runpod
import predict

MODEL = predict.Predictor()
MODEL.setup()

def base64_to_tempfile(base64_file: str) -> str:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        temp_file.write(base64.b64decode(base64_file))
    return temp_file.name

def youtube_to_tempfile(youtube_url: str) -> str:
    tmp_dir = tempfile.mkdtemp()
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": os.path.join(tmp_dir, "audio.%(ext)s"),
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "wav",
        }],
        "quiet": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.extract_info(youtube_url, download=True)
    audio_path = os.path.join(tmp_dir, "audio.wav")
    return audio_path

def build_raw_and_processed(trans_segments, ja_segments):
    """
    RepeatTubeのraw/processed形式に変換
    raw: { parsed, allWords }
    processed: segments配列
    """
    # allWords：全単語リスト（単語タイムスタンプ付き）
    allWords = []
    word_to_seg = []  # 各単語がどのセグメントに属するか
    for seg_idx, seg in enumerate(trans_segments):
        if hasattr(seg, 'words') and seg.words:
            for w in seg.words:
                allWords.append({
                    "word": w.word.strip(),
                    "startMs": round(w.start * 1000),
                })
                word_to_seg.append(seg_idx)

    # parsed：文境界インデックス＋日本語訳
    parsed = []
    current_start = 0
    for seg_idx, seg in enumerate(trans_segments):
        # このセグメントに属する単語のインデックス範囲を計算
        indices = [i for i, s in enumerate(word_to_seg) if s == seg_idx]
        if not indices:
            continue
        end_idx = indices[-1]

        # 日本語訳
        translation = ""
        if seg_idx < len(ja_segments):
            translation = ja_segments[seg_idx].text.strip()

        parsed.append({
            "start": current_start,
            "end": end_idx,
            "translation": translation,
        })
        current_start = end_idx + 1

    # processed：RepeatTubeが直接使う形式
    processed = []
    for i, seg in enumerate(trans_segments):
        translation = ""
        if i < len(ja_segments):
            translation = ja_segments[i].text.strip()
        processed.append({
            "index": i,
            "start": round(seg.start, 3),
            "end": round(seg.end, 3),
            "text": seg.text.strip(),
            "translation": translation,
        })
    # endを次のstartに統一
    for i in range(len(processed) - 1):
        processed[i]["end"] = processed[i + 1]["start"]

    raw = {
        "parsed": parsed,
        "allWords": allWords,
    }
    return raw, processed

@rp_debugger.FunctionTimer
def run_whisper_job(job):
    job_input = job['input']

    with rp_debugger.LineTimer('validation_step'):
        input_validation = validate(job_input, INPUT_VALIDATIONS)
        if 'errors' in input_validation:
            return {"error": input_validation['errors']}
        job_input = input_validation['validated_input']

    if job_input.get('youtube_url', False):
        audio_input = youtube_to_tempfile(job_input['youtube_url'])
    elif job_input.get('audio', False) and not job_input.get('audio_base64', False):
        with rp_debugger.LineTimer('download_step'):
            audio_input = download_files_from_urls(job['id'], [job_input['audio']])[0]
    elif job_input.get('audio_base64', False):
        audio_input = base64_to_tempfile(job_input['audio_base64'])
    else:
        return {'error': 'Must provide youtube_url, audio, or audio_base64'}

    with rp_debugger.LineTimer('transcription_step'):
        # 英語文字起こし（単語タイムスタンプ付き）
        trans_result = MODEL.predict(
            audio=audio_input,
            model_name=job_input.get("model", "turbo"),
            transcription="plain_text",
            translation="plain_text",
            translate=False,
            language="en",
            temperature=job_input["temperature"],
            best_of=job_input["best_of"],
            beam_size=job_input["beam_size"],
            patience=job_input["patience"],
            length_penalty=job_input["length_penalty"],
            suppress_tokens=job_input.get("suppress_tokens", "-1"),
            initial_prompt=job_input["initial_prompt"],
            condition_on_previous_text=job_input["condition_on_previous_text"],
            temperature_increment_on_fallback=job_input["temperature_increment_on_fallback"],
            compression_ratio_threshold=job_input["compression_ratio_threshold"],
            logprob_threshold=job_input["logprob_threshold"],
            no_speech_threshold=job_input["no_speech_threshold"],
            enable_vad=job_input["enable_vad"],
            word_timestamps=True,
        )

    with rp_debugger.LineTimer('translation_step'):
        # 日本語翻訳
        ja_result = MODEL.predict(
            audio=audio_input,
            model_name=job_input.get("model", "turbo"),
            transcription="plain_text",
            translation="plain_text",
            translate=True,
            language="en",
            temperature=job_input["temperature"],
            best_of=job_input["best_of"],
            beam_size=job_input["beam_size"],
            patience=job_input["patience"],
            length_penalty=job_input["length_penalty"],
            suppress_tokens=job_input.get("suppress_tokens", "-1"),
            initial_prompt=job_input["initial_prompt"],
            condition_on_previous_text=job_input["condition_on_previous_text"],
            temperature_increment_on_fallback=job_input["temperature_increment_on_fallback"],
            compression_ratio_threshold=job_input["compression_ratio_threshold"],
            logprob_threshold=job_input["logprob_threshold"],
            no_speech_threshold=job_input["no_speech_threshold"],
            enable_vad=job_input["enable_vad"],
            word_timestamps=False,
        )

    with rp_debugger.LineTimer('build_step'):
        raw, processed = build_raw_and_processed(
            trans_result['_segments'],
            ja_result['_segments'],
        )

    with rp_debugger.LineTimer('cleanup_step'):
        rp_cleanup.clean(['input_objects'])

    return {
        "raw": raw,
        "processed": processed,
        "detected_language": trans_result["detected_language"],
    }

runpod.serverless.start({"handler": run_whisper_job})
