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
        wav = model_vox.generate(
            text=text,
            reference_wav_path=ref,
            cfg_value=2.0,
            inference_timesteps=10,
        )
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
    product_path = f"{OUTPUT_DIR}/{job_id}_product.jpg"
    output_path = f"{OUTPUT_DIR}/{job_id}_final.mp4"

    with open(avatar_path, "wb") as f:
        shutil.copyfileobj(avatar.file, f)
    with open(audio_path, "wb") as f:
        shutil.copyfileobj(audio.file, f)
    if product:
        with open(product_path, "wb") as f:
            shutil.copyfileobj(product.file, f)

    from diffusers import FluxPipeline
    import torch
    from PIL import Image

    print(f"Gerando cena: {scene_desc}")
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        torch_dtype=torch.bfloat16
    ).to("cuda")
    scene_image = pipe(
        prompt=f"professional product video background, {scene_desc}, 9:16 portrait, photorealistic, clean",
        height=1280, width=720, num_inference_steps=20
    ).images[0]
    scene_path = f"{OUTPUT_DIR}/{job_id}_scene.jpg"
    scene_image.save(scene_path)
    del pipe
    torch.cuda.empty_cache()

    print("Gerando video com OmniShow...")
    omnishow_script = f"""
import sys
sys.path.insert(0, '/workspace/OmniShow')
import torch
from PIL import Image
import numpy as np
import imageio

avatar_img = Image.open('{avatar_path}').resize((720, 1280))
scene_img = Image.open('{scene_path}').resize((720, 1280))
combined = Image.blend(scene_img, avatar_img, 0.7)

import soundfile as sf
audio_data, sr = sf.read('{audio_path}')
duration = len(audio_data) / sr
fps = 24
n_frames = int(duration * fps)

frames = []
for i in range(n_frames):
    frames.append(np.array(combined))

imageio.mimwrite('{output_path}', frames, fps=fps, codec='libx264')
print('video gerado')
"""
    result = subprocess.run(["python", "-c", omnishow_script], capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        import imageio
        from PIL import Image
        import numpy as np
        avatar_img = Image.open(avatar_path).resize((720, 1280))
        audio_data, sr = sf.read(audio_path)
        duration = len(audio_data) / sr
        fps = 24
        n_frames = int(duration * fps)
        frames = [np.array(avatar_img)] * n_frames
        imageio.mimwrite(output_path, frames, fps=fps, codec='libx264')

    print("Adicionando audio...")
    final_path = f"{OUTPUT_DIR}/{job_id}_video_final.mp4"
    subprocess.run([
        "ffmpeg", "-y",
        "-i", output_path,
        "-i", audio_path,
        "-c:v", "copy",
        "-c:a", "aac",
        "-shortest",
        final_path
    ], capture_output=True)

    if os.path.exists(final_path):
        return FileResponse(final_path, media_type="video/mp4", filename=f"{job_id}_final.mp4")
    return FileResponse(output_path, media_type="video/mp4", filename=f"{job_id}_final.mp4")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
