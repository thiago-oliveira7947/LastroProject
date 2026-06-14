"""
Scraper de imóveis para ZAP Imóveis e Viva Real.

Estratégia: extrai o bloco JSON-LD (schema.org ItemList) embutido nas páginas
de busca dos portais. Sem API key — acesso público idêntico ao de qualquer
browser. Os dados são os mesmos usados pelo Google para indexação.

Fontes:
  ZAP   → https://www.zapimoveis.com.br/{negocio}/{tipo}/{uf}+{cidade}/
  Viva  → https://www.vivareal.com.br/{negocio}/{uf}/{cidade}/{tipo}/
"""
from __future__ import annotations

import re
import time
import unicodedata

import requests

_TIMEOUT = 20
_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)

# ─── estado → sigla ───────────────────────────────────────────────────────
_UF: dict[str, str] = {
    "São Paulo": "sp", "Minas Gerais": "mg", "Rio de Janeiro": "rj",
    "Paraná": "pr", "Rio Grande do Sul": "rs", "Santa Catarina": "sc",
    "Bahia": "ba", "Pernambuco": "pe", "Ceará": "ce",
    "Goiás": "go", "Distrito Federal": "df", "Espírito Santo": "es",
    "Mato Grosso do Sul": "ms", "Mato Grosso": "mt", "Pará": "pa",
    "Amazonas": "am", "Maranhão": "ma", "Piauí": "pi",
    "Rio Grande do Norte": "rn", "Alagoas": "al", "Paraíba": "pb",
    "Sergipe": "se", "Rondônia": "ro", "Tocantins": "to",
    "Amapá": "ap", "Acre": "ac", "Roraima": "rr",
}

# ─── tipo em PT → slug do portal ──────────────────────────────────────────
_TIPO_ZAP: dict[str, str] = {
    "apartment": "apartamentos",
    "house":     "casas",
    "land":      "terrenos",
    "commercial":"comercial",
}
_TIPO_VIVA: dict[str, str] = {
    "apartment": "apartamento_residencial",
    "house":     "casa_residencial",
    "land":      "terreno_condominio",
    "commercial":"sala_comercial",
}

_NEGOCIO_ZAP:  dict[str, str] = {"SALE": "venda",   "RENTAL": "aluguel"}
_NEGOCIO_VIVA: dict[str, str] = {"SALE": "venda",   "RENTAL": "aluguel"}


# ─── helpers ──────────────────────────────────────────────────────────────

def _slug(text: str) -> str:
    """Remove acentos e converte para slug (letras/números/hífens)."""
    nfkd = unicodedata.normalize("NFKD", text)
    sem_acento = "".join(c for c in nfkd if not unicodedata.combining(c))
    slug = re.sub(r"[^a-zA-Z0-9 ]", "", sem_acento).lower().strip()
    return re.sub(r"\s+", "-", slug)


def _parse_int_from_name(name: str, pattern: str) -> int:
    m = re.search(pattern, name, re.IGNORECASE)
    return int(m.group(1)) if m else 0


def _bairro_from_name(name: str, cidade: str) -> str:
    """Extrai bairro do título do anúncio: '... em Bairro Nome, Cidade'."""
    m = re.search(r"\bem\s+(.+?),\s*" + re.escape(cidade), name, re.IGNORECASE)
    if m:
        return m.group(1).strip().title()
    return ""


def _clean_text(text: str) -> str:
    """Remove caracteres de substituição Unicode (U+FFFD) e normaliza o texto."""
    # U+FFFD = replacement character (encoding mismatch residue)
    clean = text.replace("�", "").replace("�", "")
    # Restaura m² (frequentemente corrompido no JSON-LD do ZAP)
    clean = re.sub(r"(\d+)\s*m\s*,", r"\1m², ", clean)
    clean = re.sub(r"(\d+)\s*m\s+", r"\1m² ", clean)
    return clean.strip()


def _fetch_html(url: str) -> str | None:
    try:
        r = requests.get(
            url,
            headers={
                "User-Agent": _UA,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "pt-BR,pt;q=0.9",
                # Sem Accept-Encoding: requests usa gzip/deflate nativamente
                # br (Brotli) requer pacote externo — evitamos para simplificar
            },
            timeout=_TIMEOUT,
        )
        if r.status_code != 200:
            return None
        return r.content.decode("utf-8", errors="replace")
    except Exception:
        return None


def _extrair_items_ld(html: str) -> list[dict]:
    """Extrai a lista de imóveis do bloco JSON-LD schema.org/ItemList."""
    import json
    pattern = r'<script[^>]*type="application/ld\+json"[^>]*>([\s\S]*?)</script>'
    for block_text in re.findall(pattern, html, re.IGNORECASE):
        try:
            block_text_clean = block_text.replace("�", "")
            data = json.loads(block_text_clean)
            if data.get("@type") == "ItemList":
                return data.get("itemListElement", [])
        except Exception:
            continue
    return []


def _normalizar_ld(entry: dict, portal: str) -> dict | None:
    """Converte um item ListItem/schema.org → formato interno do Lastro."""
    item = entry.get("item") or entry
    if not item:
        return None

    item_type  = item.get("@type", "Apartment")
    url        = item.get("url") or (item.get("offers") or {}).get("url") or ""
    offers     = item.get("offers") or {}
    addr       = item.get("address") or {}
    floor_size = item.get("floorSize") or {}

    # Preço
    try:
        preco = float(offers.get("price") or 0)
    except (ValueError, TypeError):
        preco = 0.0

    # Condomínio
    extra = offers.get("additionalProperty") or {}
    try:
        condo = float(extra.get("value") or 0) if extra.get("name") == "Condominium Fee" else 0.0
    except (ValueError, TypeError):
        condo = 0.0

    # Área
    try:
        area = float(floor_size.get("value") or 0)
    except (ValueError, TypeError):
        area = 0.0
    area = max(area, 18.0)

    # Tipo
    tipo_map = {"Apartment": "apartment", "House": "house",
                "LandParcel": "land", "Place": "commercial"}
    tipo = tipo_map.get(item_type, "apartment")

    name      = _clean_text(item.get("name") or "")
    desc_raw  = _clean_text(item.get("description") or "")

    quartos   = int(item.get("numberOfBedrooms") or 0)
    banheiros = int(item.get("numberOfBathroomsTotal") or 0)

    # Vagas — não está no schema.org diretamente; extrai do título
    vagas = _parse_int_from_name(name, r"(\d+)\s*vaga")

    # Foto principal
    images = item.get("image") or []
    thumb  = images[0] if images else ""

    cidade_raw = addr.get("addressLocality") or ""
    estado_uf  = addr.get("addressRegion") or ""
    bairro     = _bairro_from_name(name, cidade_raw)

    # Estado por extenso (preferimos o nome completo para o modelo)
    _UF_INV = {v: k for k, v in _UF.items()}
    estado = _UF_INV.get(estado_uf.lower(), estado_uf)

    return {
        "id":            item.get("@id") or "",
        "titulo":        name,
        "descricao":     desc_raw[:500],
        "preco":         preco,
        "thumbnail":     thumb,
        "link":          url,
        "cidade":        cidade_raw.lower().strip(),
        "estado":        estado,
        "bairro":        bairro,
        "endereco":      _clean_text(addr.get("streetAddress") or ""),
        "tipo":          tipo,
        "quartos":       quartos,
        "banheiros":     banheiros,
        "area":          area,
        "vagas_garagem": vagas,
        "suites":        0,
        "condominio":    condo,
        "iptu":          0.0,
        "latitude":      None,
        "longitude":     None,
        "negocio":       "RENTAL" if "aluguel" in url else "SALE",
        "fonte":         portal,
    }


# ─── URL builders ─────────────────────────────────────────────────────────

def _url_zap(
    cidade: str,
    estado: str,
    tipo: str = "apartment",
    negocio: str = "SALE",
    pagina: int = 1,
    bairro: str = "",
) -> str:
    uf    = _UF.get(estado, "sp")
    tipo_slug  = _TIPO_ZAP.get(tipo, "imoveis")
    neg_slug   = _NEGOCIO_ZAP.get(negocio, "venda")
    cidade_slug = _slug(cidade)
    base   = f"https://www.zapimoveis.com.br/{neg_slug}/{tipo_slug}/{uf}+{cidade_slug}/"
    if bairro:
        base += f"?bairros={_slug(bairro)}"
        if pagina > 1:
            base += f"&pagina={pagina}"
    elif pagina > 1:
        base += f"?pagina={pagina}"
    return base


def _url_viva(
    cidade: str,
    estado: str,
    tipo: str = "apartment",
    negocio: str = "SALE",
    pagina: int = 1,
    bairro: str = "",
) -> str:
    uf    = _UF.get(estado, "sp")
    tipo_slug  = _TIPO_VIVA.get(tipo, "apartamento_residencial")
    neg_slug   = _NEGOCIO_VIVA.get(negocio, "venda")
    cidade_slug = _slug(cidade)
    base   = f"https://www.vivareal.com.br/{neg_slug}/{uf}/{cidade_slug}/{tipo_slug}/"
    if pagina > 1:
        base += f"?pagina={pagina}"
    return base


# ─── public API ───────────────────────────────────────────────────────────

def buscar(
    cidade: str = "Guarulhos",
    estado: str = "São Paulo",
    bairro: str = "",
    tipos: list[str] | None = None,
    negocio: str = "SALE",
    limit: int = 24,
    pagina: int = 1,
    portal: str = "zap",
    # parâmetros de filtro ignorados pelo scraper mas aceitos para compatibilidade
    quartos_min: int = 0,
    preco_min: float = 0,
    preco_max: float = 0,
    area_min: float = 0,
) -> list[dict]:
    """
    Busca imóveis via JSON-LD do ZAP Imóveis ou Viva Real.
    Sem API key — dados públicos idênticos ao que o Google indexa.
    """
    tipos_use = tipos or ["apartment", "house"]
    results: list[dict] = []

    url_fn = _url_zap if portal == "zap" else _url_viva

    for tipo in tipos_use:
        if len(results) >= limit:
            break
        url = url_fn(
            cidade=cidade, estado=estado, tipo=tipo,
            negocio=negocio, pagina=pagina, bairro=bairro,
        )
        html = _fetch_html(url)
        if not html:
            time.sleep(0.5)
            continue

        entries = _extrair_items_ld(html)
        for entry in entries:
            n = _normalizar_ld(entry, portal)
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

        time.sleep(0.3)  # respeita rate limit

    return results[:limit]


def buscar_zap(cidade: str, estado: str, **kw) -> list[dict]:
    return buscar(cidade=cidade, estado=estado, portal="zap", **kw)


def buscar_vivareal(cidade: str, estado: str, **kw) -> list[dict]:
    return buscar(cidade=cidade, estado=estado, portal="vivareal", **kw)
