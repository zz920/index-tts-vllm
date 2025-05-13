
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "7"

import asyncio
import io
import traceback
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from contextlib import asynccontextmanager
import uvicorn
import argparse
import json
import asyncio
import time
import numpy as np
import soundfile as sf

from indextts.infer_vllm import IndexTTS

tts = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global tts
    cfg_path = os.path.join(args.model_dir, "config.yaml")
    tts = IndexTTS(model_dir=args.model_dir, cfg_path=cfg_path)

    current_file_path = os.path.abspath(__file__)
    cur_dir = os.path.dirname(current_file_path)
    speaker_path = os.path.join(cur_dir, "assets/speaker.json")
    if os.path.exists(speaker_path):
        speaker_dict = json.load(open(speaker_path, 'r'))

        for speaker, audio_paths in speaker_dict.items():
            tts.registry_speaker(speaker, audio_paths)
    yield
    # Clean up the ML models and release the resources
    # ml_models.clear()

app = FastAPI(lifespan=lifespan)


@app.post("/tts_url", responses={
    200: {"content": {"application/octet-stream": {}}},
    500: {"content": {"application/json": {}}}
})
async def tts_api_url(request: Request):
    try:
        data = await request.json()
        text = data["text"]
        audio_paths = data["audio_paths"]

        global tts
        sr, wav = await tts.infer(audio_paths, text)
        
        with io.BytesIO() as wav_buffer:
            sf.write(wav_buffer, wav, sr, format='WAV')
            wav_bytes = wav_buffer.getvalue()

        return Response(content=wav_bytes, media_type="audio/wav")
    
    except Exception as ex:
        tb_str = ''.join(traceback.format_exception(type(ex), ex, ex.__traceback__))
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "error": str(tb_str)
            }
        )


@app.post("/tts", responses={
    200: {"content": {"application/octet-stream": {}}},
    500: {"content": {"application/json": {}}}
})
async def tts_api(request: Request):
    try:
        data = await request.json()
        text = data["text"]
        character = data["character"]

        global tts
        sr, wav = await tts.infer_with_ref_audio_embed(character, text)
        
        with io.BytesIO() as wav_buffer:
            sf.write(wav_buffer, wav, sr, format='WAV')
            wav_bytes = wav_buffer.getvalue()

        return Response(content=wav_bytes, media_type="audio/wav")
    
    except Exception as ex:
        tb_str = ''.join(traceback.format_exception(type(ex), ex, ex.__traceback__))
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "error": str(tb_str)
            }
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=11996)
    parser.add_argument("--model_dir", type=str, default="/path/to/IndexTeam/Index-TTS")
    args = parser.parse_args()

    uvicorn.run(app=app, host=args.host, port=args.port)