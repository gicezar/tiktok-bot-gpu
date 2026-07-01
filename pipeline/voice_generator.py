import httpx
import os
from config.settings import OUTPUT_DIR

async def generate_voice(text: str, job_id: str, gpu_url: str) -> str:
    output_path = f"{OUTPUT_DIR}/{job_id}_audio.wav"
    async with httpx.AsyncClient(timeout=120) as client:
        response = await client.post(
            f"{gpu_url}/generate_voice",
            data={"text": text},
        )
        if response.status_code != 200:
            raise ValueError(f"Erro ao gerar voz: {response.status_code} {response.text}")
        with open(output_path, "wb") as f:
            f.write(response.content)
    return output_path
