import base64
import tempfile
import os
import atexit
import json
import re
import shutil
import subprocess
import urllib.request
import soundfile as sf
import pyloudnorm as pyln
import numpy as np
import yt_dlp
from rp_schema import INPUT_VALIDATIONS
from runpod.serverless.utils import download_files_from_urls, rp_cleanup, rp_debugger
from runpod.serverless.utils.rp_validator import validate
import runpod
import predict

MODEL = predict.Predictor()
MODEL.setup()

# Node.jsパスを取得（js_runtimes設定に使用）
_node_path = shutil.which("node") or shutil.which("nodejs")

FIREBASE_API_KEY = os.environ.get("FIREBASE_API_KEY", "")
FIREBASE_PROJECT = os.environ.get("FIREBASE_PROJECT", "")

# YouTubeのcookiesをBase64環境変数から一時ファイルに展開
_cookies_file = None
_youtube_cookies_b64 = os.environ.get("YOUTUBE_COOKIES_B64", "")
if _youtube_cookies_b64:
    _cookies_tmp = tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode='wb')
    _cookies_tmp.write(base64.b64decode(_youtube_cookies_b64))
    _cookies_tmp.close()
    _cookies_file = _cookies_tmp.name
    atexit.register(os.unlink, _cookies_file)
    print(f"YouTube cookies loaded: {_cookies_file}")

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
        "js_runtimes": {"node": {"path": _node_path} if _node_path else {}},
    }
    if _cookies_file:
        ydl_opts["cookiefile"] = _cookies_file
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.extract_info(youtube_url, download=True)
    out_path = os.path.join(tmp_dir, "audio.wav")
    exists = os.path.exists(out_path)
    size   = os.path.getsize(out_path) if exists else 0
    print(f"[youtube_to_tempfile] exists={exists} size={size} bytes path={out_path}")
    if not exists or size == 0:
        print(f"[youtube_to_tempfile] WARNING: audio file missing or empty after download")
    return out_path

def measure_lufs(audio_path: str) -> float | None:
    """ffmpeg ebur128 フィルターで LUFS を計測"""
    try:
        result = subprocess.run(
            ["ffmpeg", "-nostats", "-i", audio_path,
             "-filter_complex", "ebur128", "-f", "null", "/dev/null"],
            capture_output=True, text=True
        )
        m = re.search(r'I:\s+([-\d.]+)\s+LUFS', result.stderr)
        if m:
            val = float(m.group(1))
            print(f"[measure_lufs] ffmpeg LUFS={val}")
            return val
        else:
            print(f"[measure_lufs] WARNING: LUFS pattern not found in ffmpeg output")
            print(f"[measure_lufs] ffmpeg stderr tail: {result.stderr[-500:]}")
    except Exception as e:
        print(f"[measure_lufs] ERROR: {e}")
    return None

def calculate_lufs(audio_path: str) -> float | None:
    """pyloudnorm (ITU-R BS.1770) で LUFS を計測 + 診断ログ"""
    try:
        file_size = os.path.getsize(audio_path) if os.path.exists(audio_path) else 0
        print(f"[calculate_lufs] path={audio_path} file_size={file_size} bytes")
        data, rate = sf.read(audio_path)
        duration_sec = len(data) / rate if rate > 0 else 0
        channels = 1 if data.ndim == 1 else data.shape[1]
        max_amp = float(np.max(np.abs(data)))
        rms = float(np.sqrt(np.mean(data ** 2)))
        print(f"[calculate_lufs] rate={rate}Hz duration={duration_sec:.2f}s channels={channels} max_amp={max_amp:.6f} rms={rms:.6f}")
        if max_amp < 1e-6:
            print(f"[calculate_lufs] WARNING: audio is essentially silent (max_amp < 1e-6)")
        if duration_sec < 0.4:
            print(f"[calculate_lufs] WARNING: duration {duration_sec:.3f}s < 0.4s (BS.1770 requires >= 400ms)")
        meter = pyln.Meter(rate)
        loudness = meter.integrated_loudness(data)
        print(f"[calculate_lufs] pyloudnorm LUFS={loudness:.2f}")
        return round(loudness, 2)
    except Exception as e:
        print(f"[calculate_lufs] ERROR: {e}")
        return None

def save_all_words_to_firebase(video_id: str, all_words: list, integrated_loudness=None):
    if not FIREBASE_API_KEY or not FIREBASE_PROJECT:
        print("Firebase config not set, skipping save")
        return
    mask = "updateMask.fieldPaths=allWords&updateMask.fieldPaths=fetchedAt"
    if integrated_loudness is not None:
        mask += "&updateMask.fieldPaths=integratedLoudness"
    url = (
        f"https://firestore.googleapis.com/v1/projects/{FIREBASE_PROJECT}"
        f"/databases/(default)/documents/transcripts/{video_id}"
        f"?{mask}"
        f"&key={FIREBASE_API_KEY}"
    )
    from datetime import datetime, timezone, timedelta
    fields = {
        "allWords":  {"stringValue": json.dumps(all_words)},
        "fetchedAt": {"stringValue": datetime.now(timezone(timedelta(hours=9))).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "+09:00"},
    }
    if integrated_loudness is not None:
        fields["integratedLoudness"] = {"doubleValue": integrated_loudness}
    payload = json.dumps({"fields": fields}).encode("utf-8")
    req = urllib.request.Request(url, data=payload, method="PATCH")
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req) as res:
            print(f"Firebase saved: {res.status}, LUFS={integrated_loudness}")
    except Exception as e:
        print(f"Firebase save error: {e}")

def extract_video_id(youtube_url: str) -> str:
    m = re.search(r'(?:v=|/embed/|/shorts/|youtu\.be/)([a-zA-Z0-9_-]{11})', youtube_url)
    return m.group(1) if m else None

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

    audio_size = os.path.getsize(audio_input) if os.path.exists(audio_input) else 0
    print(f"[run_whisper_job] audio_input={audio_input} size={audio_size} bytes")

    with rp_debugger.LineTimer('lufs_step'):
        lufs_ffmpeg = measure_lufs(audio_input)
        lufs_pyloud = calculate_lufs(audio_input)
        integrated_loudness = lufs_pyloud
        print(f"[run_whisper_job] LUFS summary: ffmpeg={lufs_ffmpeg} pyloudnorm={lufs_pyloud}")

    with rp_debugger.LineTimer('transcription_step'):
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

    with rp_debugger.LineTimer('build_words_step'):
        all_words = []
        for seg in trans_result['_segments']:
            if hasattr(seg, 'words') and seg.words:
                for w in seg.words:
                    all_words.append({
                        "word": w.word.strip(),
                        "startMs": round(w.start * 1000),
                        "endMs": round(w.end * 1000),
                    })
            else:
                raw_words = seg.text.strip().split()
                if not raw_words:
                    continue
                seg_start_ms = round(seg.start * 1000)
                seg_dur_ms = round((seg.end - seg.start) * 1000)
                for wi, w in enumerate(raw_words):
                    word_start_ms = seg_start_ms + round(seg_dur_ms * wi / len(raw_words))
                    word_end_ms = seg_start_ms + round(seg_dur_ms * (wi + 1) / len(raw_words))
                    all_words.append({
                        "word": w,
                        "startMs": word_start_ms,
                        "endMs": word_end_ms,
                    })

    with rp_debugger.LineTimer('firebase_save_step'):
        youtube_url = job_input.get('youtube_url', '')
        video_id = extract_video_id(youtube_url) if youtube_url else None
        if video_id:
            save_all_words_to_firebase(video_id, all_words, integrated_loudness)

    with rp_debugger.LineTimer('cleanup_step'):
        rp_cleanup.clean(['input_objects'])

    return {
        "allWords": all_words,
        "detected_language": trans_result["detected_language"],
        "word_count": len(all_words),
        "integratedLoudness": integrated_loudness,
    }

runpod.serverless.start({"handler": run_whisper_job})
