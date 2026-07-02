import asyncio
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from telegram.constants import ParseMode
from config.settings import TELEGRAM_BOT_TOKEN, TELEGRAM_ALLOWED_USER_ID, ANTHROPIC_API_KEY
from pipeline.orchestrator import Orchestrator, VideoJob, JobStatus
from pipeline.scraper import is_tiktok_url

GPU_SERVER_URL = os.getenv("GPU_SERVER_URL", "https://evs5lzl36qe3ph-8000.proxy.runpod.net")
orchestrator = Orchestrator(api_key=ANTHROPIC_API_KEY, gpu_url=GPU_SERVER_URL)

def only_allowed(func):
    async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE):
        if TELEGRAM_ALLOWED_USER_ID and update.effective_user.id != TELEGRAM_ALLOWED_USER_ID:
            await update.message.reply_text("⛔ Acesso negado.")
            return
        return await func(update, context)
    return wrapper

@only_allowed
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 *Bot TikTok Shop ativo!*\n\n"
        "Como usar:\n\n"
        "📎 Manda o *link do produto* do TikTok Shop\n\n"
        "📸 Manda a *foto do produto* com o título na legenda\n\n"
        "📝 Manda só o *nome do produto* em texto\n\n"
        "Vou gerar o vídeo completo! 🎬",
        parse_mode=ParseMode.MARKDOWN
    )

@only_allowed
async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text.strip()
    if is_tiktok_url(text):
        msg = await update.message.reply_text("🔗 Link recebido! Iniciando pipeline...")
    else:
        msg = await update.message.reply_text(f"📝 Produto: *{text}*\nIniciando...", parse_mode=ParseMode.MARKDOWN)
    await _start_job(update, text, image_bytes=None, status_msg=msg)

@only_allowed
async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    caption = (update.message.caption or "").strip()
    if not caption:
        await update.message.reply_text("📸 Manda novamente com o *título* na legenda.", parse_mode=ParseMode.MARKDOWN)
        return
    msg = await update.message.reply_text(f"📸 Produto: *{caption}*\nIniciando...", parse_mode=ParseMode.MARKDOWN)
    photo = update.message.photo[-1]
    file = await context.bot.get_file(photo.file_id)
    image_bytes = bytes(await file.download_as_bytearray())
    await _start_job(update, caption, image_bytes=image_bytes, status_msg=msg)

async def _start_job(update: Update, input_text: str, image_bytes, status_msg):
    async def on_progress(job_id: str, message: str):
        try:
            await status_msg.edit_text(message)
        except:
            pass

    job = await orchestrator.create_and_run(
        user_id=update.effective_user.id,
        input_text=input_text,
        image_bytes=image_bytes,
        progress_callback=on_progress,
    )
    asyncio.create_task(_wait_and_deliver(update, job, status_msg))

async def _wait_and_deliver(update: Update, job: VideoJob, status_msg):
    # Aguarda até 20 minutos (240 x 5s) — OmniAvatar pode levar ~8 min
    for _ in range(240):
        await asyncio.sleep(5)

        if job.status == JobStatus.DONE:
            caption = f"✅ *Vídeo pronto!*\n📦 {job.product.title if job.product else ''}"
            if job.script:
                caption += f"\n\n📋 _{job.script.narration[:300]}_"

            # Entrega o VÍDEO se foi gerado
            if job.video_final_path and os.path.exists(job.video_final_path):
                with open(job.video_final_path, "rb") as f:
                    await update.message.reply_video(
                        video=f,
                        caption=caption,
                        parse_mode=ParseMode.MARKDOWN,
                        supports_streaming=True,
                    )
                # Limpa o arquivo após enviar
                try:
                    os.unlink(job.video_final_path)
                except Exception:
                    pass

            # Fallback: entrega só o áudio se vídeo não existir
            elif job.audio_path and os.path.exists(job.audio_path):
                await update.message.reply_audio(
                    audio=open(job.audio_path, "rb"),
                    caption=caption + "\n\n⚠️ _Vídeo não gerado, enviando áudio._",
                    parse_mode=ParseMode.MARKDOWN,
                )
            else:
                await update.message.reply_text(caption, parse_mode=ParseMode.MARKDOWN)
            return

        if job.status == JobStatus.FAILED:
            await update.message.reply_text(
                f"❌ Erro: `{job.error}`",
                parse_mode=ParseMode.MARKDOWN
            )
            return

    await update.message.reply_text("⏱️ Timeout — o vídeo demorou mais que 20 min. Tente novamente.")

def run_bot():
    if not TELEGRAM_BOT_TOKEN:
        raise ValueError("TELEGRAM_BOT_TOKEN não configurado")
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    print("🤖 Bot iniciado!")
    app.run_polling(drop_pending_updates=True)
