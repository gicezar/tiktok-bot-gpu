import json
import re
from dataclasses import dataclass
import anthropic

@dataclass
class VideoScript:
    narration: str
    outfit_description: str
    scene_description: str
    video_style: str
    duration_estimate: int

async def generate_video_script(product, api_key: str) -> VideoScript:
    client = anthropic.Anthropic(api_key=api_key)

    prompt = (
        "Voce e um especialista em videos virais para TikTok Shop Brasil.\n"
        "Crie um roteiro de venda persuasivo e natural em portugues brasileiro.\n"
        "Estilo direto, conversacional, como um amigo recomendando um produto.\n\n"
        "Produto: " + product.title + "\n"
        "Preco: " + (product.price or "nao informado") + "\n"
        "Descricao: " + (product.description or "nao informada") + "\n\n"
        "Retorne APENAS este JSON sem markdown:\n"
        "{\n"
        '  "narration": "texto da fala em portugues brasileiro, 80-120 palavras, hook forte no inicio, beneficio claro no meio, CTA no final tipo corre la ou link na bio",\n'
        '  "outfit_description": "descricao em ingles da roupa ideal para esse produto",\n'
        '  "scene_description": "descricao em ingles do cenario ideal",\n'
        '  "video_style": "uma palavra: energetico calmo urgente inspirador ou divertido",\n'
        '  "duration_estimate": 25\n'
        "}"
    )

    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}]
    )

    text = message.content[0].text.strip()
    json_match = re.search(r'\{.*\}', text, re.DOTALL)
    if not json_match:
        raise ValueError("Claude nao retornou JSON valido: " + text[:200])
    data = json.loads(json_match.group())
    return VideoScript(
        narration=data["narration"],
        outfit_description=data["outfit_description"],
        scene_description=data["scene_description"],
        video_style=data["video_style"],
        duration_estimate=int(data.get("duration_estimate", 25)),
    )
