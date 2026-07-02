"""
gpu_server.py v4.0 — TikTok Bot GPU Worker
Stack: VoxCPM2 (clone de voz) + OmniAvatar 1.3B (corpo animado + lip sync)

DEPLOY SEM CACHE — Start Command do Pod RunPod:
  wget -q -O /workspace/gpu_server.py \
    https://raw.githubusercontent.com/gicezar/tiktok-bot-gpu/main/gpu_server.py \
  && python /workspace/gpu_server.py

GPU recomendada: RTX A4000 (16GB) ou RTX 4090 (24GB)
Modelos ficam em /workspace/models (Network Volume — baixam 1x só):
  - Wan2.1-T2V-1.3B: ~3GB
  - OmniAvatar-1.3B: ~1GB  
  - wav2vec2-base-960h: ~400MB
  Total: ~5GB no Network Volume

Custo por vídeo de 30s na RTX A4000 (~$0.17/hr):
  - ~8 min de geração = ~$0.02 por vídeo
"""

import os, uuid, shutil, subprocess, sys, tempfile
import soundfile as sf
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, Form, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
import uvicorn

# ── Paths ──────────────────────────────────────────────────────────────────────
OUTPUT_DIR      = "/workspace/outputs"
MODELS_DIR      = "/workspace/models"
VOICE_REF       = "/workspace/minha_voz.wav"
OMNIAVATAR_REPO = "/workspace/OmniAvatar"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# ── Modelo global ──────────────────────────────────────────────────────────────
model_vox = None

# ── Setup OmniAvatar (roda 1x, salva no Network Volume) ───────────────────────
def setup_omniavatar():
    """Clona OmniAvatar e baixa modelos se ainda não existirem."""
    
    wan_model  = f"{MODELS_DIR}/Wan2.1-T2V-1.3B"
    omni_model = f"{MODELS_DIR}/OmniAvatar-1.3B"
    wav2vec    = f"{MODELS_DIR}/wav2vec2-base-960h"

    # 1. Clonar repo
    if not os.path.exists(OMNIAVATAR_REPO):
        print("[setup] Clonando OmniAvatar...")
        subprocess.run([
            "git", "clone", "--depth=1",
            "https://github.com/Omni-Avatar/OmniAvatar",
            OMNIAVATAR_REPO
        ], check=True)
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-q",
            "torch==2.4.0", "torchvision==0.19.0", "torchaudio==2.4.0",
            "--index-url", "https://download.pytorch.org/whl/cu124"
        ], check=True)
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-q",
            "-r", f"{OMNIAVATAR_REPO}/requirements.txt"
        ], check=True)
        print("[setup] OmniAvatar instalado.")

    # 2. Baixar modelos (só na 1a vez, ficam no Network Volume)
    if not os.path.exists(wan_model):
        print("[setup] Baixando Wan2.1-T2V-1.3B (~3GB)...")
        subprocess.run([
            "huggingface-cli", "download", "Wan-AI/Wan2.1-T2V-1.3B",
            "--local-dir", wan_model
        ], check=True)

    if not os.path.exists(omni_model):
        print("[setup] Baixando OmniAvatar-1.3B (~1GB)...")
        subprocess.run([
            "huggingface-cli", "download", "OmniAvatar/OmniAvatar-1.3B",
            "--local-dir", omni_model
        ], check=True)

    if not os.path.exists(wav2vec):
        print("[setup] Baixando wav2vec2 (~400MB)...")
        subprocess.run([
            "huggingface-cli", "download", "facebook/wav2vec2-base-960h",
            "--local-dir", wav2vec
        ], check=True)

    # 3. Criar symlink para que os scripts do OmniAvatar encontrem os modelos
    pretrained = f"{OMNIAVATAR_REPO}/pretrained_models"
    if not os.path.exists(pretrained):
        os.makedirs(pretrained, exist_ok=True)
    
    for name in ["Wan2.1-T2V-1.3B", "OmniAvatar-1.3B", "wav2vec2-base-960h"]:
        link = f"{pretrained}/{name}"
        src  = f"{MODELS_DIR}/{name}"
        if not os.path.exists(link) and os.path.exists(src):
            os.symlink(src, link)

    print("[setup] OmniAvatar pronto!")

def run_omniavatar(avatar_path: str, audio_path: str, prompt: str, job_id: str) -> str:
    """
    Gera vídeo animado com OmniAvatar 1.3B.
    Usa CPU offloading para caber em GPUs com 8-16GB de VRAM.
    Retorna path do vídeo gerado.
    """
    output_dir  = f"{OUTPUT_DIR}/{job_id}_omni"
    output_path = f"{OUTPUT_DIR}/{job_id}_omni.mp4"
    os.makedirs(output_dir, exist_ok=True)

    # Cria arquivo de input no formato que o OmniAvatar espera
    input_file = f"{OUTPUT_DIR}/{job_id}_input.txt"
    with open(input_file, "w") as f:
        f.write(f"{prompt}@@{avatar_path}@@{audio_path}\n")

    cmd = [
        "torchrun", "--standalone", "--nproc_per_node=1",
        f"{OMNIAVATAR_REPO}/scripts/inference.py",
        "--config", f"{OMNIAVATAR_REPO}/configs/inference_1.3B.yaml",
        "--input_file", input_file,
        "--hp=num_steps=20,guidance_scale=4.5,audio_scale=3,"
        "num_persistent_param_in_dit=0,"  # CPU offload — cabe em 8GB VRAM
        "tea_cache_l1_thresh=0.14",        # TeaCache — acelera ~30%
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=OMNIAVATAR_REPO,
        env={**os.environ, "OUTPUT_DIR": output_dir}
    )

    if result.returncode != 0:
        print(f"[omniavatar] stderr: {result.stderr[-800:]}")
        raise RuntimeError(f"OmniAvatar falhou: {result.stderr[-200:]}")

    # Encontra o vídeo gerado (OmniAvatar salva com nome baseado no job)
    mp4_files = list(Path(output_dir).glob("*.mp4"))
    if not mp4_files:
        raise RuntimeError("OmniAvatar não gerou nenhum vídeo")

    raw_video = str(mp4_files[0])

    # Redimensiona para formato TikTok 1080x1920
    compose_tiktok(raw_video, output_path)

    # Limpeza
    try:
        shutil.rmtree(output_dir)
        os.unlink(input_file)
    except Exception:
        pass

    return output_path

def compose_tiktok(input_path: str, output_path: str):
    """
    Redimensiona vídeo para formato TikTok vertical 1080x1920.
    OmniAvatar gera em 480p (832x480) — aqui convertemos para vertical.
    """
    subprocess.run([
        "ffmpeg", "-y",
        "-i", input_path,
        "-vf", (
            "scale=1080:1080:force_original_aspect_ratio=decrease,"
            "pad=1080:1920:(ow-iw)/2:300:black,"
            "setsar=1"
        ),
        "-c:v", "libx264", "-preset", "fast",
        "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart",
        output_path
    ], check=True, capture_output=True)

# ── Startup ────────────────────────────────────────────────────────────────────
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

# ── Endpoints ──────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "ok",
        "version": "4.0",
        "voxcpm2": model_vox is not None,
        "omniavatar": os.path.exists(f"{MODELS_DIR}/OmniAvatar-1.3B"),
        "wan_base": os.path.exists(f"{MODELS_DIR}/Wan2.1-T2V-1.3B"),
        "voice_ref": os.path.exists(VOICE_REF),
    }

@app.post("/upload_voice")
async def upload_voice(audio: UploadFile = File(...)):
    with open(VOICE_REF, "wb") as f:
        shutil.copyfileobj(audio.file, f)
    return {"status": "ok", "path": VOICE_REF}

@app.post("/generate_voice_clone")
async def generate_voice_clone(text: str = Form(...)):
    job_id = str(uuid.uuid4())[:8]
    output_path = f"{OUTPUT_DIR}/{job_id}_audio.wav"

    if os.path.exists(VOICE_REF):
        wav = model_vox.generate(
            text=text,
            reference_wav_path=VOICE_REF,
            cfg_value=2.0,
            inference_timesteps=10,
        )
    else:
        print("[voice] AVISO: sem voz de referência, usando voz padrão")
        wav = model_vox.generate(text=text, cfg_value=2.0, inference_timesteps=10)

    sf.write(output_path, wav, model_vox.tts_model.sample_rate)
    return FileResponse(output_path, media_type="audio/wav", filename=f"{job_id}_audio.wav")

@app.post("/generate_video")
async def generate_video(
    avatar: UploadFile  = File(...),
    audio: UploadFile   = File(...),
    product: UploadFile = File(None),
    scene_desc: str     = Form("modern apartment living room, bright and clean"),
    outfit_desc: str    = Form("casual smart outfit"),
):
    """
    Pipeline completo:
    1. Recebe avatar (jpg) + audio (wav)
    2. OmniAvatar 1.3B anima o avatar com o áudio (corpo + lip sync)
    3. ffmpeg redimensiona para 1080x1920 (TikTok)
    4. Retorna MP4 final
    """
    job_id = str(uuid.uuid4())[:8]
    avatar_path = f"{OUTPUT_DIR}/{job_id}_avatar.jpg"
    audio_path  = f"{OUTPUT_DIR}/{job_id}_audio.wav"
    final_video = f"{OUTPUT_DIR}/{job_id}_final.mp4"

    # Salva arquivos recebidos
    with open(avatar_path, "wb") as f:
        shutil.copyfileobj(avatar.file, f)
    with open(audio_path, "wb") as f:
        shutil.copyfileobj(audio.file, f)

    # Monta prompt descritivo para o OmniAvatar
    prompt = (
        f"A person in {outfit_desc} is speaking directly to the camera, "
        f"presenting a product with natural gestures and expressions, "
        f"in a {scene_desc}."
    )

    try:
        print(f"[{job_id}] Gerando vídeo com OmniAvatar...")
        final_video = await _run_in_executor(
            run_omniavatar, avatar_path, audio_path, prompt, job_id
        )
        print(f"[{job_id}] Vídeo pronto: {final_video}")
        return FileResponse(final_video, media_type="video/mp4",
                            filename=f"{job_id}_final.mp4")

    except Exception as e:
        print(f"[{job_id}] Erro no OmniAvatar: {e}")
        raise

    finally:
        for f in [avatar_path, audio_path]:
            try: os.unlink(f)
            except: pass

async def _run_in_executor(func, *args):
    """Roda função bloqueante em thread separada para não travar o FastAPI."""
    import asyncio
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, func, *args)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
