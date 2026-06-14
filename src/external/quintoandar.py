"""
Scraper de imóveis para Quinto Andar.

Estratégias (em ordem de tentativa):
  1. __NEXT_DATA__ — JSON injetado pelo Next.js no HTML (SSR)
  2. Varredura de <script> tags com JSON — alguns portais embebem dados em
     blocos application/json ou scripts sem type
  3. JSON-LD schema.org/ItemList — fallback padrão

URLs:
  Aluguel → https://www.quintoandar.com.br/alugar/imovel/{cidade}/
  Venda   → https://www.quintoandar.com.br/comprar/imovel/{cidade}/
  + Bairro → .../alugar/imovel/{cidade}/{bairro}/
"""
from __future__ import annotations

import json
import re
import sys
import time
import unicodedata

import requests


def _log(*args, **kwargs):
    print(*args, **kwargs, file=sys.stderr, flush=True)

_TIMEOUT = 25
_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)
_HEADERS = {
    "User-Agent": _UA,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "pt-BR,pt;q=0.9,en-US;q=0.8",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
}

_TIPO_MAP = {
    "APARTMENT": "apartment",
    "HOUSE":     "house",
    "LAND":      "land",
    "COMMERCIAL":"commercial",
    "STUDIO":    "apartment",
    "CONDO":     "apartment",
    "DUPLEX":    "house",
    "KITNET":    "apartment",
}

# Chaves que indicam um objeto de imóvel dentro de JSON arbitrário
_LISTING_KEYS = {"id", "rent", "salePrice", "price", "bedrooms", "area",
                 "neighborhood", "coverImageUrl", "listingId", "totalCost"}


# ─── helpers ──────────────────────────────────────────────────────────────

def _slug(text: str) -> str:
    nfkd = unicodedata.normalize("NFKD", text)
    sem_acento = "".join(c for c in nfkd if not unicodedata.combining(c))
    slug = re.sub(r"[^a-zA-Z0-9 ]", "", sem_acento).lower().strip()
    return re.sub(r"\s+", "-", slug)


def _url_qa(cidade_slug: str, bairro_slug: str, negocio: str) -> str:
    base = "alugar" if negocio == "RENTAL" else "comprar"
    if bairro_slug:
        return f"https://www.quintoandar.com.br/{base}/imovel/{cidade_slug}/{bairro_slug}/"
    return f"https://www.quintoandar.com.br/{base}/imovel/{cidade_slug}/"


def _fetch_html(url: str) -> str | None:
    try:
        r = requests.get(url, headers=_HEADERS, timeout=_TIMEOUT)
        _log(f"[QA] HTTP {r.status_code} — {url}")
        if r.status_code != 200:
            return None
        html = r.content.decode("utf-8", errors="replace")
        _log(f"[QA] HTML recebido: {len(html):,} bytes")
        return html
    except Exception as e:
        _log(f"[QA] Erro de conexão: {e}")
        return None


# ─── extração ─────────────────────────────────────────────────────────────

def _extrair_next_data(html: str) -> dict:
    m = re.search(
        r'<script[^>]+id=["\']__NEXT_DATA__["\'][^>]*>([\s\S]*?)</script>',
        html, re.IGNORECASE,
    )
    if not m:
        _log("[QA] __NEXT_DATA__ não encontrado na página.")
        return {}
    try:
        data = json.loads(m.group(1))
        _log(f"[QA] __NEXT_DATA__ encontrado — chaves raiz: {list(data.keys())[:8]}")
        return data
    except Exception as e:
        _log(f"[QA] Falha ao parsear __NEXT_DATA__: {e}")
        return {}


def _parece_listing(obj: object) -> bool:
    """Retorna True se o dict tem pelo menos 2 chaves típicas de imóvel."""
    if not isinstance(obj, dict):
        return False
    return len(_LISTING_KEYS & set(obj.keys())) >= 2


def _buscar_listas_recursivo(obj: object, depth: int = 0) -> list[list[dict]]:
    """Varre recursivamente obj em busca de listas de imóveis."""
    if depth > 8:
        return []
    found: list[list[dict]] = []
    if isinstance(obj, list) and len(obj) >= 1:
        if _parece_listing(obj[0]):
            found.append(obj)
        else:
            for item in obj[:5]:
                found.extend(_buscar_listas_recursivo(item, depth + 1))
    elif isinstance(obj, dict):
        for v in obj.values():
            found.extend(_buscar_listas_recursivo(v, depth + 1))
    return found


def _extrair_listings_next(data: dict) -> list[dict]:
    if not data:
        return []
    page_props = data.get("props", {}).get("pageProps", {})

    # Tentativa direta nas chaves mais comuns
    for key in ("homes", "listings", "listingItems", "results", "data",
                 "properties", "imoveis", "items", "searchResults"):
        val = page_props.get(key)
        if isinstance(val, list) and val and _parece_listing(val[0]):
            _log(f"[QA] Listings encontrados em pageProps.{key}: {len(val)} itens")
            return val

    # Busca recursiva em pageProps
    listas = _buscar_listas_recursivo(page_props)
    if listas:
        melhor = max(listas, key=len)
        _log(f"[QA] Listings encontrados via busca recursiva: {len(melhor)} itens")
        return melhor

    _log(f"[QA] Nenhuma lista de imóveis em __NEXT_DATA__. Chaves pageProps: {list(page_props.keys())[:10]}")
    return []


def _extrair_json_scripts(html: str) -> list[dict]:
    """Varre todas as tags <script> buscando JSON que contenha listings."""
    pattern = re.compile(
        r'<script[^>]*>([\s\S]{50,200000}?)</script>', re.IGNORECASE
    )
    results: list[dict] = []
    for m in pattern.finditer(html):
        text = m.group(1).strip()
        if not text.startswith("{") and not text.startswith("["):
            continue
        try:
            data = json.loads(text)
        except Exception:
            continue
        listas = _buscar_listas_recursivo(data)
        for lista in listas:
            results.extend(lista)
        if results:
            _log(f"[QA] Listings encontrados em <script> tag: {len(results)} itens")
            break
    return results


def _extrair_jsonld(html: str) -> list[dict]:
    results = []
    pattern = r'<script[^>]*type="application/ld\+json"[^>]*>([\s\S]*?)</script>'
    for block in re.findall(pattern, html, re.IGNORECASE):
        try:
            data = json.loads(block.replace("", ""))
            if data.get("@type") == "ItemList":
                for e in data.get("itemListElement", []):
                    item = e.get("item") or e
                    if isinstance(item, dict):
                        results.append(item)
        except Exception:
            continue
    if results:
        _log(f"[QA] Listings encontrados via JSON-LD: {len(results)} itens")
    return results


# ─── normalização ─────────────────────────────────────────────────────────

def _normalizar(raw: dict, cidade_ref: str, estado_ref: str) -> dict | None:
    if not raw or not isinstance(raw, dict):
        return None

    item_id = str(raw.get("id") or raw.get("listingId") or raw.get("@id") or "")

    href = raw.get("href") or raw.get("listingUrl") or raw.get("url") or ""
    if href and not href.startswith("http"):
        href = "https://www.quintoandar.com.br" + href

    tipo_raw = (raw.get("type") or raw.get("propertyType") or
                raw.get("@type") or "APARTMENT").upper()
    tipo_raw = re.sub(r"[^A-Z]", "", tipo_raw)
    tipo = _TIPO_MAP.get(tipo_raw, "apartment")

    preco = 0.0
    for key in ("rent", "salePrice", "price", "totalCost", "monthlyRent",
                "totalRent", "offers"):
        val = raw.get(key)
        if isinstance(val, dict):
            val = val.get("price") or val.get("value") or 0
        if val:
            try:
                preco = float(val)
                if preco > 0:
                    break
            except (ValueError, TypeError):
                pass

    area = 0.0
    for key in ("area", "totalArea", "usableArea", "livingArea", "floorSize"):
        val = raw.get(key)
        if isinstance(val, dict):
            val = val.get("value") or 0
        if val:
            try:
                area = float(val)
                if area > 0:
                    break
            except (ValueError, TypeError):
                pass
    area = max(area, 18.0)

    quartos   = int(raw.get("bedrooms") or raw.get("beds") or
                    raw.get("numberOfBedrooms") or 0)
    banheiros = int(raw.get("bathrooms") or raw.get("baths") or
                    raw.get("numberOfBathroomsTotal") or 0)
    vagas     = int(raw.get("parkingSpots") or raw.get("parkingSpaces") or
                    raw.get("garage") or 0)

    addr = raw.get("address") or {}
    if isinstance(addr, str):
        addr = {}

    bairro = (
        raw.get("neighborhood") or raw.get("region") or
        addr.get("neighborhood") or addr.get("suburb") or ""
    ).strip()
    cidade = (raw.get("city") or addr.get("city") or
              addr.get("addressLocality") or cidade_ref).strip()
    estado = (raw.get("state") or addr.get("state") or
              addr.get("addressRegion") or estado_ref).strip()

    thumb = ""
    for key in ("coverImageUrl", "imageUrl", "photo", "image",
                "thumbnail", "coverPhoto"):
        val = raw.get(key)
        if isinstance(val, str) and val.startswith("http"):
            thumb = val
            break
        if isinstance(val, list) and val:
            candidate = val[0]
            if isinstance(candidate, str) and candidate.startswith("http"):
                thumb = candidate
                break

    negocio_raw = (raw.get("businessType") or raw.get("listingType") or "").upper()
    if "RENT" in negocio_raw or "ALUG" in negocio_raw:
        negocio = "RENTAL"
    elif "SALE" in negocio_raw or "VEND" in negocio_raw or "BUY" in negocio_raw:
        negocio = "SALE"
    else:
        negocio = "RENTAL" if preco < 20_000 else "SALE"

    condo = 0.0
    for key in ("condoFee", "condominium", "adminFee", "condominiumFee"):
        val = raw.get(key)
        if val:
            try:
                condo = float(val)
                if condo > 0:
                    break
            except (ValueError, TypeError):
                pass

    lat = lon = None
    for lat_key, lon_key in [("lat", "lng"), ("latitude", "longitude"),
                              ("lat", "lon"), ("geoLat", "geoLon")]:
        if raw.get(lat_key) and raw.get(lon_key):
            try:
                lat = float(raw[lat_key])
                lon = float(raw[lon_key])
                break
            except (ValueError, TypeError):
                pass

    titulo = (raw.get("title") or raw.get("name") or
              f"Imóvel {quartos}q — {bairro or cidade}").strip()

    return {
        "id":            item_id,
        "titulo":        titulo,
        "descricao":     str(raw.get("description") or "")[:500],
        "preco":         preco,
        "thumbnail":     thumb,
        "link":          href,
        "cidade":        cidade.lower().strip(),
        "estado":        estado,
        "bairro":        bairro,
        "endereco":      str(addr.get("street") or addr.get("streetAddress") or "")[:100],
        "tipo":          tipo,
        "quartos":       quartos,
        "banheiros":     banheiros,
        "area":          area,
        "vagas_garagem": vagas,
        "suites":        0,
        "condominio":    condo,
        "iptu":          0.0,
        "latitude":      lat,
        "longitude":     lon,
        "negocio":       negocio,
        "fonte":         "quintoandar",
    }


# ─── public API ───────────────────────────────────────────────────────────

def buscar(
    cidade: str = "Guarulhos",
    estado: str = "São Paulo",
    bairro: str = "",
    tipos: list[str] | None = None,
    negocio: str = "SALE",
    limit: int = 24,
    quartos_min: int = 0,
    preco_min: float = 0,
    preco_max: float = 0,
    area_min: float = 0,
    **_kw,
) -> list[dict]:
    """Busca imóveis no Quinto Andar com múltiplas estratégias de extração."""
    cidade_slug = _slug(cidade)
    bairro_slug = _slug(bairro) if bairro else ""

    url = _url_qa(cidade_slug, bairro_slug, negocio)
    _log(f"\n[QA] === Iniciando busca: cidade={cidade!r} bairro={bairro!r} negocio={negocio!r}")
    _log(f"[QA] URL: {url}")

    html = _fetch_html(url)
    if not html:
        return []

    # ── Estratégia 1: __NEXT_DATA__ ──
    next_data = _extrair_next_data(html)
    raw_items = _extrair_listings_next(next_data)

    # ── Estratégia 2: <script> tags com JSON ──
    if not raw_items:
        _log("[QA] Tentando extração via script tags...")
        raw_items = _extrair_json_scripts(html)

    # ── Estratégia 3: JSON-LD schema.org ──
    if not raw_items:
        _log("[QA] Tentando extração via JSON-LD...")
        raw_items = _extrair_jsonld(html)

    if not raw_items:
        _log(f"[QA] Nenhum dado extraído da página. "
              f"O site pode estar bloqueando scrapers ou usando renderização 100% client-side.")
        return []

    _log(f"[QA] Total de itens brutos: {len(raw_items)}")

    results: list[dict] = []
    for raw in raw_items:
        if len(results) >= limit:
            break
        n = _normalizar(raw, cidade, estado)
        if not n:
            continue
        if quartos_min > 0 and n["quartos"] > 0 and n["quartos"] < quartos_min:
            continue
        if preco_max > 0 and n["preco"] > 0 and n["preco"] > preco_max:
            continue
        if preco_min > 0 and n["preco"] > 0 and n["preco"] < preco_min:
            continue
        if area_min > 0 and n["area"] > 0 and n["area"] < area_min:
            continue
        results.append(n)

    _log(f"[QA] Resultados após filtros: {len(results)}")
    time.sleep(0.3)
    return results[:limit]
