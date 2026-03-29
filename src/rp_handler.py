import base64
import tempfile
import os
import atexit
import json
import shutil
import subprocess
import urllib.request
import yt_dlp
from rp_schema import INPUT_VALIDATIONS
from runpod.serverless.utils import download_files_from_urls, rp_cleanup, rp_debugger
from runpod.serverless.utils.rp_validator import validate
import runpod
import predict

MODEL = predict.Predictor()
MODEL.setup()

# 診断情報をグローバルに収集（ジョブのエラーレスポンスに含める）
_diag = {}

_node_path = shutil.which("node") or shutil.which("nodejs")
try:
    _node_version = subprocess.check_output([_node_path or "node", "--version"], stderr=subprocess.DEVNULL).decode().strip() if _node_path else "not found"
except Exception as e:
    _node_version = f"error: {e}"
_diag["node_path"] = _node_path
_diag["node_version"] = _node_version
print(f"[diag] Node.js: {_node_path} {_node_version}")

try:
    import yt_dlp_ejs
    import os as _os
    pkg_dir = _os.path.dirname(yt_dlp_ejs.__file__)
    # 全ファイルを再帰的に列挙
    all_files = []
    for root, dirs, files in _os.walk(pkg_dir):
        for f in files:
            all_files.append(_os.path.relpath(_os.path.join(root, f), pkg_dir))
    _diag["ejs_dir"] = pkg_dir
    _diag["ejs_all_files"] = all_files
    # プラグインのスクリプトパス取得を試みる
    try:
        script_path = getattr(yt_dlp_ejs, 'SOLVER_SCRIPT', None) or getattr(yt_dlp_ejs, 'script_path', None)
        _diag["ejs_script_path_attr"] = str(script_path)
    except Exception as e2:
        _diag["ejs_script_path_attr"] = f"error: {e2}"
    # yt-dlp-ejsのentry_pointsを確認
    try:
        import importlib.metadata as _meta
        eps = list(_meta.entry_points(group='yt_dlp_plugins'))
        _diag["yt_dlp_plugins_eps"] = [str(e) for e in eps]
    except Exception as e3:
        _diag["yt_dlp_plugins_eps"] = f"error: {e3}"
except Exception as e:
    _diag["ejs_error"] = f"{type(e).__name__}: {e}"
print(f"[diag] ejs: {_diag}")

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
    node_runtime = f"node:{_node_path}" if _node_path else "node"
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": os.path.join(tmp_dir, "audio.%(ext)s"),
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "wav",
        }],
        "quiet": False,
        "verbose": True,
        "jsruntimes": [node_runtime],
        "extractor_args": {"youtube": {"player_client": ["mweb"]}},
    }
    if _cookies_file:
        ydl_opts["cookiefile"] = _cookies_file
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.extract_info(youtube_url, download=True)
    return os.path.join(tmp_dir, "audio.wav")

def save_all_words_to_firebase(video_id: str, all_words: list):
    if not FIREBASE_API_KEY or not FIREBASE_PROJECT:
        print("Firebase config not set, skipping save")
        return
    url = (
        f"https://firestore.googleapis.com/v1/projects/{FIREBASE_PROJECT}"
        f"/databases/(default)/documents/transcripts/{video_id}"
        f"?updateMask.fieldPaths=allWords&updateMask.fieldPaths=fetchedAt"
        f"&key={FIREBASE_API_KEY}"
    )
    from datetime import datetime, timezone, timedelta
    payload = json.dumps({
        "fields": {
            "allWords":  {"stringValue": json.dumps(all_words)},
            "fetchedAt": {"stringValue": datetime.now(timezone(timedelta(hours=9))).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "+09:00"},
        }
    }).encode("utf-8")
    req = urllib.request.Request(url, data=payload, method="PATCH")
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req) as res:
            print(f"Firebase saved: {res.status}")
    except Exception as e:
        print(f"Firebase save error: {e}")

def extract_video_id(youtube_url: str) -> str:
    import re
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
        try:
            audio_input = youtube_to_tempfile(job_input['youtube_url'])
        except Exception as e:
            return {"error": str(e), "diag": _diag}
    elif job_input.get('audio', False) and not job_input.get('audio_base64', False):
        with rp_debugger.LineTimer('download_step'):
            audio_input = download_files_from_urls(job['id'], [job_input['audio']])[0]
    elif job_input.get('audio_base64', False):
        audio_input = base64_to_tempfile(job_input['audio_base64'])
    else:
        return {'error': 'Must provide youtube_url, audio, or audio_base64'}

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
            save_all_words_to_firebase(video_id, all_words)

    with rp_debugger.LineTimer('cleanup_step'):
        rp_cleanup.clean(['input_objects'])

    return {
        "allWords": all_words,
        "detected_language": trans_result["detected_language"],
        "word_count": len(all_words),
    }

runpod.serverless.start({"handler": run_whisper_job})
