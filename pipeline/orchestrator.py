import asyncio
import uuid
import os
from dataclasses import dataclass, field
from typing import Optional, Callable, Awaitable
from enum import Enum
from datetime import datetime
from pipeline.scraper import ProductData, scrape_tiktok_product, build_product_from_manual, is_tiktok_url
from pipeline.script_generator import VideoScript, generate_video_script
from pipeline.video_generator import generate_voice_clone, generate_video, upload_voice_reference

class JobStatus(str, Enum):
    QUEUED = "queued"
    SCRAPING = "scraping"
    GENERATING_SCRIPT = "generating_script"
    GENERATING_VOICE = "generating_voice"
    GENERATING_VIDEO = "generating_video"
    DONE = "done"
    FAILED = "failed"

STATUS_MESSAGES = {
    JobStatus.QUEUED:            "⏳ Na fila...",
    JobStatus.SCRAPING:          "🔍 Buscando dados do produto...",
    JobStatus.GENERATING_SCRIPT: "✍️ Criando roteiro de venda...",
    JobStatus.GENERATING_VOICE:  "🎙️ Gerando sua voz...",
    JobStatus.GENERATING_VIDEO:  "🎥 Gerando o vídeo (~5 min)...",
    JobStatus.DONE:              "✅ Pronto!",
    JobStatus.FAILED:            "❌ Falhou. Tente novamente.",
}

@dataclass
class VideoJob:
    job_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    user_id: int = 0
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    input_url: Optional[str] = None
    input_title: Optional[str] = None
    input_image_bytes: Optional[bytes] = None
    product: Optional[ProductData] = None
    script: Optional[VideoScript] = None
    audio_path: Optional[str] = None
    video_final_path: Optional[str] = None
    status: JobStatus = JobStatus.QUEUED
    error: Optional[str] = None

ProgressCallback = Callable[[str, str], Awaitable[None]]

class Orchestrator:
    def __init__(self, api_key: str, gpu_url: str):
        self.api_key = api_key
        self.gpu_url = gpu_url
        self.jobs: dict[str, VideoJob] = {}
        self._callbacks: dict[str, ProgressCallback] = {}
        self.voice_uploaded = False
        self.avatar_path = os.getenv("AVATAR_BASE_IMAGE", "./assets/avatar_base.jpg")
        self.voice_path = "./assets/minha_voz.wav"

    async def _update(self, job: VideoJob, status: JobStatus):
        job.status = status
        cb = self._callbacks.get(job.job_id)
        if cb:
            await cb(job.job_id, STATUS_MESSAGES[status])

    async def create_and_run(self, user_id: int, input_text: str, image_bytes=None, progress_callback=None) -> VideoJob:
        job = VideoJob(user_id=user_id)
        self.jobs[job.job_id] = job
        if progress_callback:
            self._callbacks[job.job_id] = progress_callback
        if is_tiktok_url(input_text):
            job.input_url = input_text
        else:
            job.input_title = input_text
            job.input_image_bytes = image_bytes
        asyncio.create_task(self._run(job))
        return job

    async def _run(self, job: VideoJob):
        try:
            await self._update(job, JobStatus.SCRAPING)
            if job.input_url:
                product = await scrape_tiktok_product(job.input_url)
                if not product:
                    raise ValueError("Não consegui extrair dados do link.")
            else:
                product = await build_product_from_manual(
                    title=job.input_title or "Produto",
                    description=None,
                    image_bytes=job.input_image_bytes,
                )
            job.product = product

            await self._update(job, JobStatus.GENERATING_SCRIPT)
            job.script = await generate_video_script(product, self.api_key)

            await self._update(job, JobStatus.GENERATING_VOICE)
            if not self.voice_uploaded and os.path.exists(self.voice_path):
                await upload_voice_reference(self.gpu_url, self.voice_path)
                self.voice_uploaded = True
            job.audio_path = await generate_voice_clone(
                text=job.script.narration,
                job_id=job.job_id,
                gpu_url=self.gpu_url,
            )

            await self._update(job, JobStatus.GENERATING_VIDEO)
            product_image_path = f"./outputs/{job.job_id}_product.jpg"
            if job.product.image_bytes:
                with open(product_image_path, "wb") as f:
                    f.write(job.product.image_bytes)
            job.video_final_path = await generate_video(
                job_id=job.job_id,
                gpu_url=self.gpu_url,
                avatar_path=self.avatar_path,
                product_image_path=product_image_path if os.path.exists(product_image_path) else "",
                audio_path=job.audio_path,
                scene_desc=job.script.scene_description,
                outfit_desc=job.script.outfit_description,
            )
            await self._update(job, JobStatus.DONE)

        except Exception as e:
            job.error = str(e)
            job.status = JobStatus.FAILED
            cb = self._callbacks.get(job.job_id)
            if cb:
                await cb(job.job_id, f"❌ Erro: {str(e)[:200]}")
            print(f"[orchestrator] job {job.job_id} falhou: {e}")
