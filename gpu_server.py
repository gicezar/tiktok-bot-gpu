"""
gpu_server.py v5.2
Stack: VoxCPM2 (voz) + OmniAvatar 1.3B (video animado com lip sync)
"""
import os, sys, uuid, shutil, subprocess, tempfile
from pathlib import Path

def ensure_ffmpeg():
    if shutil.which("ffmpeg"):
        print("[setup] ffmpeg OK")
        return
    print("[setup] Instalando ffmpeg...")
    subprocess.run(["apt-get", "update"], check=True)
    subprocess.run(["apt-get", "install", "-y", "ffmpeg"], check=True)
    print("[setup] ffmpeg instalado!")

ensure_ffmpeg()

DEPS = [
    "huggingface_hub", "imageio", "imageio-ffmpeg",
    "opencv-python-headless", "einops", "omegaconf",
    "transformers", "accelerate", "peft", "librosa", "torchvision==0.20.0"
]

subprocess.run([sys.executable, "-m", "pip", "install", "-q"] + DEPS, check=True)
subprocess.run(["/usr/bin/python", "-m", "pip", "install", "-q"] + DEPS, check=True)

import soundfile as sf
from contextlib import asynccontextmanager
from fastapi import FastAPI, Form, UploadFile, File
from fastapi.responses import FileResponse
import uvicorn

OUTPUT_DIR      = "/workspace/outputs"
MODELS_DIR      = "/workspace/models"
VOICE_REF       = "/workspace/minha_voz.wav"
OMNIAVATAR_REPO = "/workspace/OmniAvatar"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

model_vox = None

def setup_omniavatar():
    from huggingface_hub import snapshot_download

    if not os.path.exists(OMNIAVATAR_REPO):
        print("[setup] Clonando OmniAvatar...")
        subprocess.run(["git", "clone", "--depth=1",
            "https://github.com/Omni-Avatar/OmniAvatar", OMNIAVATAR_REPO], check=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "-q",
            "-r", f"{OMNIAVATAR_REPO}/requirements.txt"], check=True)
        subprocess.run(["/usr/bin/python", "-m", "pip", "install", "-q",
            "-r", f"{OMNIAVATAR_REPO}/requirements.txt"], check=True)

    wan_path  = f"{MODELS_DIR}/Wan2.1-T2V-1.3B"
    omni_path = f"{MODELS_DIR}/OmniAvatar-1.3B"
    w2v_path  = f"{MODELS_DIR}/wav2vec2-base-960h"

    if not os.path.exists(wan_path):
        print("[setup] Baixando Wan2.1-T2V-1.3B...")
        snapshot_download(repo_id="Wan-AI/Wan2.1-T2V-1.3B", local_dir=wan_path)
    if not os.path.exists(omni_path):
        print("[setup] Baixando OmniAvatar-1.3B...")
        snapshot_download(repo_id="OmniAvatar/OmniAvatar-1.3B", local_dir=omni_path)
    if not os.path.exists(w2v_path):
        print("[setup] Baixando wav2vec2...")
        snapshot_download(repo_id="facebook/wav2vec2-base-960h", local_dir=w2v_path)

    pretrained = f"{OMNIAVATAR_REPO}/pretrained_models"
    os.makedirs(pretrained, exist_ok=True)
    for name in ["Wan2.1-T2V-1.3B", "OmniAvatar-1.3B", "wav2vec2-base-960h"]:
        link = f"{pretrained}/{name}"
        src  = f"{MODELS_DIR}/{name}"
        if not os.path.exists(link) and os.path.exists(src):
            os.symlink(src, link)

    if OMNIAVATAR_REPO not in sys.path:
        sys.path.insert(0, OMNIAVATAR_REPO)

    print("[setup] OmniAvatar pronto!")

def run_omniavatar(avatar_path, audio_path, prompt, job_id):
    output_dir = f"{OUTPUT_DIR}/{job_id}_omni"
    final_path = f"{OUTPUT_DIR}/{job_id}_final.mp4"
    os.makedirs(output_dir, exist_ok=True)

    input_file = f"{OUTPUT_DIR}/{job_id}_input.txt"
    with open(input_file, "w") as f:
        f.write(f"{prompt}@@{avatar_path}@@{audio_path}\n")

    env = {
        **os.environ,
        "OUTPUT_DIR": output_dir,
        "PYTHONPATH": OMNIAVATAR_REPO + ":" + os.environ.get("PYTHONPATH", ""),
        "RANK": "0",
        "LOCAL_RANK": "0",
        "WORLD_SIZE": "1",
        "MASTER_ADDR": "localhost",
        "MASTER_PORT": "29500",
    }

    cmd = [
        "/usr/bin/python", f"{OMNIAVATAR_REPO}/scripts/inference.py",
        "--config", f"{OMNIAVATAR_REPO}/configs/inference_1.3B.yaml",
        "--input_file", input_file,
        "--hp=num_steps=20,guidance_scale=4.5,audio_scale=3,tea_cache_l1_thresh=0.14",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, cwd=OMNIAVATAR_REPO, env=env)
    print(f"[omniavatar] stdout:\n{result.stdout[-3000:]}")
    print(f"[omniavatar] stderr:\n{result.stderr[-3000:]}")

    if result.returncode != 0:
        raise RuntimeError(f"OmniAvatar falhou com código {result.returncode}")

    mp4_files = list(Path(output_dir).glob("**/*.mp4"))
    for extra in [f"{OMNIAVATAR_REPO}/outputs", f"{OMNIAVATAR_REPO}/results"]:
        if os.path.exists(extra):
            mp4_files += list(Path(extra).glob("**/*.mp4"))

    if not mp4_files:
        raise RuntimeError("OmniAvatar nao gerou nenhum video")

    raw_video = str(sorted(mp4_files, key=os.path.getmtime)[-1])
    subprocess.run([
        "ffmpeg", "-y", "-i", raw_video,
        "-vf", "scale=1080:1080,pad=1080:1920:(ow-iw)/2:300:black,setsar=1",
        "-c:v", "libx264", "-preset", "fast",
        "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart", final_path
    ], check=True)

    shutil.rmtree(output_dir, ignore_errors=True)
    os.unlink(input_file)
    return final_path

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_vox
    print("[startup] Carregando VoxCPM2...")
    from voxcpm import VoxCPM
    model_vox = VoxCPM.from_pretrained("openbmb/VoxCPM2", load_denoiser=False)
    print("[startup] VoxCPM2 pronto!")
    setup_omniavatar()
    yield

app = FastAPI(lifespan=lifespan)

@app.get("/health")
def health():
    return {"status": "ok", "version": "5.2",
            "voxcpm2": model_vox is not None,
            "omniavatar": os.path.exists(f"{MODELS_DIR}/OmniAvatar-1.3B"),
            "voice_ref": os.path.exists(VOICE_REF)}

@app.post("/upload_voice")
async def upload_voice(audio: UploadFile = File(...)):
    with open(VOICE_REF, "wb") as f:
        shutil.copyfileobj(audio.file, f)
    return {"status": "ok"}

@app.post("/generate_voice_clone")
async def generate_voice_clone(text: str = Form(...)):
    job_id = str(uuid.uuid4())[:8]
    output_path = f"{OUTPUT_DIR}/{job_id}_audio.wav"
    if os.path.exists(VOICE_REF):
        wav = model_vox.generate(text=text, reference_wav_path=VOICE_REF,
                                  cfg_value=2.0, inference_timesteps=10)
    else:
        wav = model_vox.generate(text=text, cfg_value=2.0, inference_timesteps=10)
    sf.write(output_path, wav, model_vox.tts_model.sample_rate)
    return FileResponse(output_path, media_type="audio/wav")

@app.post("/generate_video")
async def generate_video(
    avatar: UploadFile = File(...),
    audio: UploadFile = File(...),
    product: UploadFile = File(None),
    scene_desc: str = Form("modern apartment"),
    outfit_desc: str = Form("casual smart outfit"),
):
    job_id = str(uuid.uuid4())[:8]
    avatar_path = f"{OUTPUT_DIR}/{job_id}_avatar.jpg"
    audio_path  = f"{OUTPUT_DIR}/{job_id}_audio.wav"

    with open(avatar_path, "wb") as f:
        shutil.copyfileobj(avatar.file, f)
    with open(audio_path, "wb") as f:
        shutil.copyfileobj(audio.file, f)

    prompt = (f"A person in {outfit_desc} speaking directly to camera, "
              f"presenting a product with natural gestures, in a {scene_desc}.")

    try:
        import asyncio
        loop = asyncio.get_event_loop()
        final_path = await loop.run_in_executor(
            None, run_omniavatar, avatar_path, audio_path, prompt, job_id)
        return FileResponse(final_path, media_type="video/mp4",
                            filename=f"{job_id}_final.mp4")
    except Exception as e:
        print(f"[error] {e}")
        raise
    finally:
        for f in [avatar_path, audio_path]:
            try: os.unlink(f)
            except: pass

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
