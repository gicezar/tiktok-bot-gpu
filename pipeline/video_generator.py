import httpx
import os
from config.settings import OUTPUT_DIR

async def upload_voice_reference(gpu_url: str, voice_path: str):
    async with httpx.AsyncClient(timeout=60) as client:
        with open(voice_path, "rb") as f:
            response = await client.post(
                f"{gpu_url}/upload_voice",
                files={"audio": f}
            )
        return response.json()

async def generate_voice_clone(text: str, job_id: str, gpu_url: str) -> str:
    output_path = f"{OUTPUT_DIR}/{job_id}_audio.wav"
    async with httpx.AsyncClient(timeout=180) as client:
        response = await client.post(
            f"{gpu_url}/generate_voice_clone",
            data={"text": text}
        )
        if response.status_code == 200:
            with open(output_path, "wb") as f:
                f.write(response.content)
            return output_path
        raise ValueError(f"Erro voz: {response.status_code}")

async def generate_video(job_id: str, gpu_url: str, avatar_path: str, product_image_path: str, audio_path: str, scene_desc: str, outfit_desc: str) -> str:
    output_path = f"{OUTPUT_DIR}/{job_id}_video.mp4"
    async with httpx.AsyncClient(timeout=600) as client:
        files = {}
        if os.path.exists(avatar_path):
            files["avatar"] = open(avatar_path, "rb")
        if product_image_path and os.path.exists(product_image_path):
            files["product"] = open(product_image_path, "rb")
        if os.path.exists(audio_path):
            files["audio"] = open(audio_path, "rb")
        response = await client.post(
            f"{gpu_url}/generate_video",
            files=files,
            data={"scene_desc": scene_desc, "outfit_desc": outfit_desc}
        )
        if response.status_code == 200:
            with open(output_path, "wb") as f:
                f.write(response.content)
            return output_path
        raise ValueError(f"Erro video: {response.status_code}")
