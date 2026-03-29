import base64
import tempfile
import os
import json
import urllib.request
import yt_dlp
from rp_schema import INPUT_VALIDATIONS
from runpod.serverless.utils import download_files_from_urls, rp_cleanup, rp_debugger
from runpod.serverless.utils.rp_validator import validate
import runpod
import predict

MODEL = predict.Predictor()
MODEL.setup()

FIREBASE_API_KEY = os.environ.get("FIREBASE_API_KEY", "")
FIREBASE_PROJECT = os.environ.get("FIREBASE_PROJECT", "")

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
        audio_input = youtube_to_tempfile(job_input['youtube_url'])
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
