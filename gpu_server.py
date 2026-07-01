import os
import uuid
import soundfile as sf
from contextlib import asynccontextmanager
from fastapi import FastAPI, Form
from fastapi.responses import FileResponse
import uvicorn

model_vox = None
OUTPUT_DIR = "/workspace/outputs"
MODEL_CACHE = "/workspace/voxcpm2_model"
os.makedirs(OUTPUT_DIR, exist_ok=True)

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_vox
    from voxcpm import VoxCPM
    print("Carregando VoxCPM2...")
    model_vox = VoxCPM.from_pretrained("openbmb/VoxCPM2", load_denoiser=False)
    print("VoxCPM2 pronto!")
    yield

app = FastAPI(lifespan=lifespan)

@app.get("/health")
def health():
    return {"status": "ok", "voxcpm2": model_vox is not None}

@app.post("/generate_voice")
async def generate_voice(text: str = Form(...)):
    job_id = str(uuid.uuid4())[:8]
    wav = model_vox.generate(text=text, cfg_value=2.0, inference_timesteps=10)
    output_path = f"{OUTPUT_DIR}/{job_id}_audio.wav"
    sf.write(output_path, wav, model_vox.tts_model.sample_rate)
    return FileResponse(output_path, media_type="audio/wav", filename=f"{job_id}_audio.wav")

@app.post("/generate_voice_clone")
async def generate_voice_clone(text: str = Form(...), reference_audio: str = Form("/workspace/minha_voz.wav")):
    job_id = str(uuid.uuid4())[:8]
    wav = model_vox.generate(
        text=text,
        reference_wav_path=reference_audio,
        cfg_value=2.0,
        inference_timesteps=10,
    )
    output_path = f"{OUTPUT_DIR}/{job_id}_audio.wav"
    sf.write(output_path, wav, model_vox.tts_model.sample_rate)
    return FileResponse(output_path, media_type="audio/wav", filename=f"{job_id}_audio.wav")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

from fastapi import UploadFile, File
import shutil

@app.post("/upload_voice")
async def upload_voice(audio: UploadFile = File(...)):
    path = "/workspace/minha_voz.wav"
    with open(path, "wb") as f:
        shutil.copyfileobj(audio.file, f)
    return {"status": "ok", "path": path}
