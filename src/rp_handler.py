import base64
import tempfile
import os
import json
import urllib.request
import urllib.error
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

def segment_and_translate_with_openai(all_words, openai_api_key):
    word_list_text = ' '.join([str(i) + ':' + w['word'] for i, w in enumerate(all_words)])

    prompt = (
        "Below is a list of words from an English video (format: index:word).\n"
        "Detect natural sentence boundaries. Each sentence is typically 10-30 words.\n"
        "Always split at periods, question marks, or exclamation marks. Otherwise use context.\n"
        "Never combine multiple sentences into one segment.\n\n"
        "Word list:\n"
        + word_list_text
        + "\n\n"
        "Return ONLY the following JSON format:\n"
        '{"segments":[{"start":0,"end":8,"translation":"Japanese translation here"},{"start":9,"end":15,"translation":"Japanese translation here"}]}\n\n'
        "Rules:\n"
        "- start and end are word indices (inclusive, 0-based)\n"
        "- Every word must be included in exactly one segment\n"
        "- translation is a natural Japanese translation of words from start to end\n"
        "- Never put multiple sentences in one segment"
    )

    payload = json.dumps({
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }).encode("utf-8")

    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer " + openai_api_key,
        },
        method="POST",
    )

    with urllib.request.urlopen(req) as res:
        data = json.loads(res.read().decode("utf-8"))

    content = data["choices"][0]["message"]["content"]
    obj = json.loads(content)
    parsed = obj.get("segments") or list(obj.values())[0]
    if not isinstance(parsed, list):
        raise ValueError("OpenAI response is not a list")

    return parsed

def build_raw_and_processed(openai_parsed, all_words):
    processed = []
    for i, seg in enumerate(openai_parsed):
        start_idx = seg.get("start", 0)
        end_idx = min(seg.get("end", start_idx), len(all_words) - 1)
        start_word = all_words[start_idx] if start_idx < len(all_words) else None
        end_word = all_words[end_idx] if end_idx < len(all_words) else None
        if not start_word:
            continue
        start_ms = start_word["startMs"]
        end_ms = end_word["startMs"] + 500 if end_word else start_ms + 2000
        text = ' '.join([all_words[j]["word"] for j in range(start_idx, end_idx + 1)]).strip()
        processed.append({
            "index": i,
            "start": start_ms / 1000,
            "end": end_ms / 1000,
            "text": text,
            "translation": seg.get("translation", ""),
        })

    for i in range(len(processed) - 1):
        processed[i]["end"] = processed[i + 1]["start"]

    raw = {
        "parsed": openai_parsed,
        "allWords": all_words,
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

    openai_api_key = os.environ.get("OPENAI_API_KEY", "")
    if not openai_api_key:
        return {"error": "OPENAI_API_KEY is not set"}

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
                    })
            else:
                raw_words = seg.text.strip().split()
                if not raw_words:
                    continue
                seg_start_ms = round(seg.start * 1000)
                seg_dur_ms = round((seg.end - seg.start) * 1000)
                for wi, w in enumerate(raw_words):
                    all_words.append({
                        "word": w,
                        "startMs": seg_start_ms + round(seg_dur_ms * wi / len(raw_words)),
                    })

    with rp_debugger.LineTimer('openai_step'):
        openai_parsed = segment_and_translate_with_openai(all_words, openai_api_key)

    with rp_debugger.LineTimer('build_step'):
        raw, processed = build_raw_and_processed(openai_parsed, all_words)

    with rp_debugger.LineTimer('cleanup_step'):
        rp_cleanup.clean(['input_objects'])

    return {
        "raw": raw,
        "processed": processed,
        "detected_language": trans_result["detected_language"],
    }

runpod.serverless.start({"handler": run_whisper_job})
