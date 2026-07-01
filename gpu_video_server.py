import os
import uuid
import soundfile as sf
import shutil
import subprocess
from contextlib import asynccontextmanager
from fastapi import FastAPI, Form, UploadFile, File
from fastapi.responses import FileResponse
import uvicorn
from voxcpm import VoxCPM

model_vox = None
OUTPUT_DIR = "/workspace/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_vox
    print("Carregando VoxCPM2...")
    model_vox = VoxCPM.from_pretrained("openbmb/VoxCPM2", load_denoiser=False)
    print("VoxCPM2 pronto!")
    yield

app = FastAPI(lifespan=lifespan)

@app.get("/health")
def health():
    return {"status": "ok", "voxcpm2": model_vox is not None}

@app.post("/upload_voice")
async def upload_voice(audio: UploadFile = File(...)):
    path = "/workspace/minha_voz.wav"
    with open(path, "wb") as f:
        shutil.copyfileobj(audio.file, f)
    return {"status": "ok", "path": path}

@app.post("/generate_voice")
async def generate_voice(text: str = Form(...)):
    job_id = str(uuid.uuid4())[:8]
    wav = model_vox.generate(text=text, cfg_value=2.0, inference_timesteps=10)
    path = f"{OUTPUT_DIR}/{job_id}_audio.wav"
    sf.write(path, wav, model_vox.tts_model.sample_rate)
    return FileResponse(path, media_type="audio/wav")

@app.post("/generate_voice_clone")
async def generate_voice_clone(text: str = Form(...)):
    job_id = str(uuid.uuid4())[:8]
    ref = "/workspace/minha_voz.wav"
    if os.path.exists(ref):
        wav = model_vox.generate(text=text, reference_wav_path=ref, cfg_value=2.0, inference_timesteps=10)
    else:
        wav = model_vox.generate(text=text, cfg_value=2.0, inference_timesteps=10)
    path = f"{OUTPUT_DIR}/{job_id}_audio.wav"
    sf.write(path, wav, model_vox.tts_model.sample_rate)
    return FileResponse(path, media_type="audio/wav")

@app.post("/generate_video")
async def generate_video(
    avatar: UploadFile = File(...),
    audio: UploadFile = File(...),
    product: UploadFile = File(None),
    scene_desc: str = Form("modern apartment living room"),
    outfit_desc: str = Form("casual smart outfit"),
):
    job_id = str(uuid.uuid4())[:8]
    avatar_path = f"{OUTPUT_DIR}/{job_id}_avatar.jpg"
    audio_path = f"{OUTPUT_DIR}/{job_id}_audio.wav"
    output_path = f"{OUTPUT_DIR}/{job_id}_final.mp4"

    with open(avatar_path, "wb") as f:
        shutil.copyfileobj(avatar.file, f)
    with open(audio_path, "wb") as f:
        shutil.copyfileobj(audio.file, f)

    audio_data, sr = sf.read(audio_path)
    duration = len(audio_data) / sr

    subprocess.run([
        "ffmpeg", "-y",
        "-loop", "1",
        "-i", avatar_path,
        "-i", audio_path,
        "-c:v", "libx264",
        "-tune", "stillimage",
        "-c:a", "aac",
        "-b:a", "192k",
        "-vf", "scale=720:1280:force_original_aspect_ratio=decrease,pad=720:1280:(ow-iw)/2:(oh-ih)/2",
        "-shortest",
        "-t", str(duration),
        output_path
    ], capture_output=True)

    if os.path.exists(output_path):
        return FileResponse(output_path, media_type="video/mp4", filename=f"{job_id}_final.mp4")
    raise Exception("Falha ao gerar video")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
