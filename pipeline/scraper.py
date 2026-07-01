import asyncio
import httpx
from playwright.async_api import async_playwright
from dataclasses import dataclass
from typing import Optional
import re

@dataclass
class ProductData:
    title: str
    price: Optional[str]
    description: Optional[str]
    image_url: Optional[str]
    image_bytes: Optional[bytes]
    source: str

async def scrape_tiktok_product(url: str) -> Optional[ProductData]:
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (iPhone; CPU iPhone OS 16_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.0 Mobile/15E148 Safari/604.1"
        )
        page = await context.new_page()
        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=20000)
            await page.wait_for_timeout(3000)
            title = await _extract_text(page, ["h1.product-title","[data-testid='product-title']","h1",".title"])
            price = await _extract_text(page, [".product-price","[data-testid='price']",".price","span.currency"])
            description = await _extract_text(page, [".product-description","[data-testid='description']",".desc"])
            image_url = await _extract_image(page, [".product-image img","[data-testid='product-image'] img",".swiper-slide img","img.product-img"])
            image_bytes = None
            if image_url:
                image_bytes = await _download_image(image_url)
            await browser.close()
            if not title:
                return None
            return ProductData(title=title, price=price, description=description, image_url=image_url, image_bytes=image_bytes, source="url")
        except Exception as e:
            await browser.close()
            print(f"[scraper] erro: {e}")
            return None

async def build_product_from_manual(title: str, description: Optional[str], image_bytes: Optional[bytes]) -> ProductData:
    return ProductData(title=title, price=None, description=description, image_url=None, image_bytes=image_bytes, source="manual")

async def _extract_text(page, selectors):
    for sel in selectors:
        try:
            el = await page.query_selector(sel)
            if el:
                text = (await el.inner_text()).strip()
                if text:
                    return text
        except:
            continue
    return None

async def _extract_image(page, selectors):
    for sel in selectors:
        try:
            el = await page.query_selector(sel)
            if el:
                src = await el.get_attribute("src")
                if src and src.startswith("http"):
                    return src
        except:
            continue
    return None

async def _download_image(url):
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(url)
            if r.status_code == 200:
                return r.content
    except:
        pass
    return None

def is_tiktok_url(text: str) -> bool:
    return bool(re.search(r"tiktok\.com", text, re.IGNORECASE))
