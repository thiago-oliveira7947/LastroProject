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
import sys
import time
import unicodedata

import requests


def _log(*args, **kwargs):
    print(*args, **kwargs, file=sys.stderr, flush=True)

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

# Zonas de São Paulo — (slug_portal, lat_centro, lon_centro)
_ZONAS_SP: dict[str, tuple[str, float, float]] = {
    "zona norte":    ("zona-norte",  -23.4876, -46.6350),
    "zona sul":      ("zona-sul",    -23.6557, -46.6654),
    "zona leste":    ("zona-leste",  -23.5462, -46.4801),
    "zona oeste":    ("zona-oeste",  -23.5674, -46.7297),
    "zona central":  ("se",          -23.5489, -46.6388),
    "zona centro":   ("se",          -23.5489, -46.6388),
    "centro":        ("se",          -23.5489, -46.6388),
    "zona nordeste": ("zona-norte",  -23.4876, -46.6350),
    "zona sudeste":  ("zona-sul",    -23.6557, -46.6654),
    "zona sudoeste": ("zona-oeste",  -23.5674, -46.7297),
    "zona noroeste": ("zona-norte",  -23.4876, -46.6350),
}

_ZONA_CENTROS: dict[str, tuple[float, float]] = {
    "zona-norte":  (-23.4876, -46.6350),
    "zona-sul":    (-23.6557, -46.6654),
    "zona-leste":  (-23.5462, -46.4801),
    "zona-oeste":  (-23.5674, -46.7297),
}


# ─── helpers ──────────────────────────────────────────────────────────────

def detectar_zona_nome(texto: str) -> tuple[str, float, float] | None:
    """Detecta zona de SP pelo nome. Retorna (slug_zona, lat, lon) ou None."""
    import unicodedata as _ud
    nfkd = _ud.normalize("NFKD", texto)
    norm = "".join(c for c in nfkd if not _ud.combining(c)).lower().replace("-", " ")
    for nome, info in _ZONAS_SP.items():
        if nome in norm:
            return info
    return None


def zona_por_coords(lat: float, lon: float) -> str:
    """Retorna o slug da zona de SP mais próxima das coordenadas."""
    return min(
        _ZONA_CENTROS,
        key=lambda z: (lat - _ZONA_CENTROS[z][0]) ** 2 + (lon - _ZONA_CENTROS[z][1]) ** 2,
    )


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
    """Extrai bairro do título do anúncio: '... em Bairro, Cidade'."""
    m = re.search(r"\bem\s+(.+?),\s*" + re.escape(cidade), name, re.IGNORECASE)
    if m:
        return m.group(1).strip().title()
    return ""


# Palavras que aparecem nos slugs ZAP/Viva antes do bairro (features do imóvel).
_FEATURE_WORDS = {
    # transação / tipo de imóvel
    "venda", "aluguel", "compra",
    "apartamento", "apartamentos", "casa", "casas", "terreno", "terrenos",
    "comercial", "comerciais", "flat", "studio", "cobertura", "coberturas",
    "galpao", "loja", "lojas", "sala", "salas", "sobrado", "duplex", "triplex",
    # cômodos / vagas (nunca fazem parte de nome de bairro)
    "quarto", "quartos", "dormitorio", "dormitorios", "suite", "suites", "dorm",
    "banheiro", "banheiros", "lavabo", "lavabos",
    "vaga", "vagas", "garagem", "garagens",
    # estado de mobília
    "mobiliado", "mobiliada", "semimobiliado", "semimobiliada",
    # conectores ambíguos (na URL geralmente precedem features, não bairro)
    "com", "sem",
    # amenidades (certamente features, não bairros)
    "piscina", "churrasqueira", "sacada", "varanda", "quintal", "gourmet",
    "elevador", "condominio", "garden",
    # estados/características do imóvel — removidos "novo/nova/alto/area"
    # pois são muito comuns em nomes de bairro (Vila Nova X, Alto da X)
    "reformado", "reformada", "renovado", "renovada",
    "andar", "andares",
}


def _bairro_from_url(url: str, cidade: str = "") -> str:
    """Extrai bairro da URL do anúncio individual ZAP/Viva Real.

    Estrutura do slug:
      /imovel/{negocio}-{tipo}-{features}-{BAIRRO}-zona-{dir}-{cidade}-{uf}-{N}m2-id-{id}/

    Âncoras usadas:
    1. Antes de 'zona-{direção}' (São Paulo e capitais)
    2. Antes de '{cidade_slug}-{uf}-{N}m2' (cidades sem zona)
    """
    m = re.search(r"/imovel/([^/?#]+)", url)
    if not m:
        return ""

    slug = m.group(1).lower().rstrip("/")

    def _extrair(before: str) -> str:
        parts = before.rstrip("-").split("-")
        bairro_parts: list[str] = []
        for part in reversed(parts):
            if not part:
                continue
            if re.search(r"\d", part):      # dígito → feature (quartos, área)
                break
            if part in _FEATURE_WORDS:      # palavra reservada → feature
                break
            bairro_parts.insert(0, part)
        return " ".join(bairro_parts).title() if bairro_parts else ""

    # Âncora 1: zona-{direção} (mais confiável, específica de SP)
    zona_m = re.search(r"-(zona-(?:sul|norte|leste|oeste|centro|central))-", slug)
    if zona_m:
        bairro = _extrair(slug[: zona_m.start()])
        if bairro:
            return bairro

    # Âncora 2: {cidade_slug}-{uf}-{digits}m2 (cidades sem zona)
    if cidade:
        cidade_slug = _slug(cidade)
        city_m = re.search(rf"-{re.escape(cidade_slug)}-[a-z]{{2}}-\d+m2", slug)
        if city_m:
            bairro = _extrair(slug[: city_m.start()])
            if bairro:
                return bairro

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

    vagas = _parse_int_from_name(name, r"(\d+)\s*vaga")

    images = item.get("image") or []
    thumb  = images[0] if images else ""

    cidade_raw = addr.get("addressLocality") or ""
    estado_uf  = addr.get("addressRegion") or ""
    # URL tem precedência: o slug é gerado pelo portal (mais confiável que o título)
    bairro = _bairro_from_url(url, cidade_raw) or _bairro_from_name(name, cidade_raw)

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
    zona_slug: str = "",
) -> str:
    uf          = _UF.get(estado, "sp")
    tipo_slug   = _TIPO_ZAP.get(tipo, "imoveis")
    neg_slug    = _NEGOCIO_ZAP.get(negocio, "venda")
    cidade_slug = _slug(cidade)
    if zona_slug:
        # Zona: sp+sao-paulo+zona-oeste  (ZAP usa + como separador)
        base = (f"https://www.zapimoveis.com.br/{neg_slug}/{tipo_slug}/"
                f"{uf}+{cidade_slug}+{zona_slug}/")
    elif bairro:
        base = (f"https://www.zapimoveis.com.br/{neg_slug}/{tipo_slug}/"
                f"{uf}+{cidade_slug}/{_slug(bairro)}/")
    else:
        base = f"https://www.zapimoveis.com.br/{neg_slug}/{tipo_slug}/{uf}+{cidade_slug}/"
    if pagina > 1:
        base += f"?pagina={pagina}"
    return base


def _url_viva(
    cidade: str,
    estado: str,
    tipo: str = "apartment",
    negocio: str = "SALE",
    pagina: int = 1,
    bairro: str = "",
    zona_slug: str = "",
) -> str:
    uf          = _UF.get(estado, "sp")
    tipo_slug   = _TIPO_VIVA.get(tipo, "apartamento_residencial")
    neg_slug    = _NEGOCIO_VIVA.get(negocio, "venda")
    cidade_slug = _slug(cidade)
    # Zona e bairro usam a mesma posição na URL do Viva Real
    seg = zona_slug if zona_slug else (_slug(bairro) if bairro else "")
    if seg:
        base = (f"https://www.vivareal.com.br/{neg_slug}/{uf}/"
                f"{cidade_slug}/{seg}/{tipo_slug}/")
    else:
        base = f"https://www.vivareal.com.br/{neg_slug}/{uf}/{cidade_slug}/{tipo_slug}/"
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
    quartos_min: int = 0,
    preco_min: float = 0,
    preco_max: float = 0,
    area_min: float = 0,
    zona_slug: str = "",
    lat_busca: float | None = None,
    lon_busca: float | None = None,
    **_kw,
) -> list[dict]:
    """Busca imóveis via JSON-LD do ZAP Imóveis ou Viva Real (busca por cidade)."""
    tipos_use = tipos or ["apartment", "house"]
    results: list[dict] = []
    url_fn = _url_zap if portal == "zap" else _url_viva

    for tipo in tipos_use:
        if len(results) >= limit:
            break
        url = url_fn(cidade=cidade, estado=estado, tipo=tipo, negocio=negocio,
                     pagina=pagina, bairro=bairro, zona_slug=zona_slug)
        _log(f"[ZAP/{portal.upper()}] Buscando URL: {url}")
        html = _fetch_html(url)
        _log(f"[ZAP/{portal.upper()}] HTML recebido: {'sim' if html else 'NÃO (bloqueado ou erro)'} ({len(html) if html else 0} bytes)")
        entries = _extrair_items_ld(html) if html else []
        _log(f"[ZAP/{portal.upper()}] Entradas JSON-LD extraídas: {len(entries)}")

        if bairro:
            bairro_slug = _slug(bairro)
            entradas_do_bairro = [
                e for e in entries[:6]
                if bairro_slug in ((e.get("item") or e).get("url") or "").lower()
            ]
            _log(f"[ZAP/{portal.upper()}] Entradas com slug '{bairro_slug}': {len(entradas_do_bairro)}/{min(6,len(entries))}")
            if entries:
                sample_urls = [((e.get("item") or e).get("url") or "")[:80] for e in entries[:3]]
                _log(f"[ZAP/{portal.upper()}] URLs amostra: {sample_urls}")

            if not entradas_do_bairro:
                # Fallback 1: zona (mais preciso que cidade completa)
                # Infere a zona pelas coordenadas do bairro quando não fornecida
                zona_fb = zona_slug
                if not zona_fb and lat_busca is not None and lon_busca is not None:
                    if "sao paulo" in _slug(cidade).replace("-", " "):
                        zona_fb = zona_por_coords(lat_busca, lon_busca)
                        _log(f"[ZAP/{portal.upper()}] Zona inferida por coords: {zona_fb}")

                usou_zona = False
                if zona_fb:
                    url_z = url_fn(cidade=cidade, estado=estado, tipo=tipo,
                                   negocio=negocio, pagina=pagina, zona_slug=zona_fb)
                    if url_z != url:
                        _log(f"[ZAP/{portal.upper()}] Fallback zona: {url_z}")
                        time.sleep(0.3)
                        html_z = _fetch_html(url_z)
                        _log(f"[ZAP/{portal.upper()}] HTML zona: {'sim' if html_z else 'NÃO'} ({len(html_z) if html_z else 0} bytes)")
                        entries_z = _extrair_items_ld(html_z) if html_z else []
                        _log(f"[ZAP/{portal.upper()}] Entradas zona: {len(entries_z)}")
                        if entries_z:
                            entries = entries_z
                            usou_zona = True

                if not usou_zona:
                    # Fallback 2: cidade completa
                    url_cidade = url_fn(cidade=cidade, estado=estado, tipo=tipo,
                                        negocio=negocio, pagina=pagina)
                    _log(f"[ZAP/{portal.upper()}] Fallback cidade: {url_cidade}")
                    if url_cidade != url:
                        time.sleep(0.3)
                        html2 = _fetch_html(url_cidade)
                        _log(f"[ZAP/{portal.upper()}] HTML cidade: {'sim' if html2 else 'NÃO'} ({len(html2) if html2 else 0} bytes)")
                        entries = _extrair_items_ld(html2) if html2 else entries
                        _log(f"[ZAP/{portal.upper()}] Entradas cidade: {len(entries)}")

        if not entries:
            _log(f"[ZAP/{portal.upper()}] Nenhuma entrada encontrada para tipo={tipo!r}, pulando.")
            time.sleep(0.5)
            continue

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

        time.sleep(0.3)

    return results[:limit]


def buscar_zap(cidade: str, estado: str, **kw) -> list[dict]:
    return buscar(cidade=cidade, estado=estado, portal="zap", **kw)


def buscar_vivareal(cidade: str, estado: str, **kw) -> list[dict]:
    return buscar(cidade=cidade, estado=estado, portal="vivareal", **kw)
