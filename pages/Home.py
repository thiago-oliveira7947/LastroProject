"""
Lastro — Busca e Previsão de Preços de Imóveis (dados reais)
Subir: streamlit run app.py
"""
from __future__ import annotations

import base64
import hashlib
import html as _html
import math
import sys
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed


def _log(*args, **kwargs):
    """Print para stderr com flush imediato — visível mesmo com stdout capturado pelo Streamlit."""
    print(*args, **kwargs, file=sys.stderr, flush=True)

import requests as _requests

import folium
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium

load_dotenv()

from src import config
from src.external import zapimoveis
from src.models.predict import carregar_metadata, carregar_modelo
from src.search.engine import filtrar_por_poi, haversine_km
from src.search.geocoder import geocode_com_fallback, geocode_detalhado
from src.search.overpass import buscar_pois_localizacoes
from src.search.query_parser import parse as parse_query

# ══════════════════════════════════════════════════════════════════════════
# CACHE DE THUMBNAILS (persiste entre re-runs via st.cache_resource)
# ══════════════════════════════════════════════════════════════════════════
# NOTA: variável de módulo (_IMG_CACHE = {}) é zerada a cada re-run do Streamlit.
# @st.cache_resource retorna o MESMO objeto em todos os re-runs/sessões.

_REFERER_MAP = {
    "zapimoveis":   "https://www.zapimoveis.com.br/",
    "vivareal":     "https://www.vivareal.com.br/",
}

def _referer(url: str) -> str:
    for key, ref in _REFERER_MAP.items():
        if key in url:
            return ref
    return ""

@st.cache_resource
def _img_cache() -> dict:
    """Singleton persistente entre re-runs — nunca zerado pelo Streamlit."""
    return {}


@st.cache_data(ttl=3600, show_spinner=False)
def _buscar_pois_cached(lat: float, lon: float, raio_m: int, categorias_key: tuple) -> dict:
    """Busca POIs no Overpass com cache de 1 hora por localização+categorias."""
    return buscar_pois_localizacoes(lat, lon, raio_m, list(categorias_key))

def _fetch_img_b64(url: str) -> str:
    """Baixa thumbnail com Referer correto, retorna data URI ou '' se falhar."""
    if not url or not url.startswith("http"):
        return ""
    cache = _img_cache()
    if url in cache:
        return cache[url]
    try:
        hdrs = {"User-Agent": "Mozilla/5.0 Chrome/124.0"}
        ref = _referer(url)
        if ref:
            hdrs["Referer"] = ref
        r = _requests.get(url, headers=hdrs, timeout=5)
        if r.status_code == 200:
            ct = r.headers.get("Content-Type", "image/webp")
            uri = f"data:{ct};base64,{base64.b64encode(r.content).decode()}"
            cache[url] = uri
            return uri
    except Exception:
        pass
    cache[url] = ""
    return ""

def _prefetch_thumbnails(items: list[dict], max_imgs: int = 48) -> None:
    """Baixa thumbnails em paralelo — popula _img_cache()."""
    cache = _img_cache()
    urls = [
        i.get("thumbnail", "")
        for i in items
        if i.get("thumbnail", "").startswith("http")
        and i.get("thumbnail", "") not in cache
    ]
    urls = list(dict.fromkeys(urls))[:max_imgs]
    if not urls:
        return
    with ThreadPoolExecutor(max_workers=10) as ex:
        for _ in as_completed({ex.submit(_fetch_img_b64, u): u for u in urls}):
            pass


# ══════════════════════════════════════════════════════════════════════════
# CONSTANTES
# ══════════════════════════════════════════════════════════════════════════
DEFAULT_LAT, DEFAULT_LON = -23.4628, -46.5333

TIPOS_PT: dict[str, str] = {
    "apartment": "Apartamento",
    "house":     "Casa",
    "commercial":"Comercial",
    "land":      "Terreno",
}
TIPO_EMOJI: dict[str, str] = {
    "apartment": "🏢", "house": "🏠", "commercial": "🏪", "land": "🌿",
}

_CIDADES_BR: dict[str, tuple[str, str]] = {
    "guarulhos":        ("Guarulhos", "São Paulo"),
    "sao paulo":        ("São Paulo", "São Paulo"),
    "osasco":           ("Osasco", "São Paulo"),
    "santo andre":      ("Santo André", "São Paulo"),
    "sao bernardo":     ("São Bernardo do Campo", "São Paulo"),
    "sao caetano":      ("São Caetano do Sul", "São Paulo"),
    "mogi das cruzes":  ("Mogi das Cruzes", "São Paulo"),
    "diadema":          ("Diadema", "São Paulo"),
    "maua":             ("Mauá", "São Paulo"),
    "campinas":         ("Campinas", "São Paulo"),
    "ribeirao preto":   ("Ribeirão Preto", "São Paulo"),
    "sorocaba":         ("Sorocaba", "São Paulo"),
    "belo horizonte":   ("Belo Horizonte", "Minas Gerais"),
    "curitiba":         ("Curitiba", "Paraná"),
    "porto alegre":     ("Porto Alegre", "Rio Grande do Sul"),
    "rio de janeiro":   ("Rio de Janeiro", "Rio de Janeiro"),
    "brasilia":         ("Brasília", "Distrito Federal"),
    "fortaleza":        ("Fortaleza", "Ceará"),
    "salvador":         ("Salvador", "Bahia"),
    "recife":           ("Recife", "Pernambuco"),
    "manaus":           ("Manaus", "Amazonas"),
    "goiania":          ("Goiânia", "Goiás"),
    "florianopolis":    ("Florianópolis", "Santa Catarina"),
    "vitoria":          ("Vitória", "Espírito Santo"),
    "natal":            ("Natal", "Rio Grande do Norte"),
    "maceio":           ("Maceió", "Alagoas"),
    "joao pessoa":      ("João Pessoa", "Paraíba"),
    "aracaju":          ("Aracaju", "Sergipe"),
    "campo grande":     ("Campo Grande", "Mato Grosso do Sul"),
    "cuiaba":           ("Cuiabá", "Mato Grosso"),
    "belem":            ("Belém", "Pará"),
    "macapa":           ("Macapá", "Amapá"),
    "porto velho":      ("Porto Velho", "Rondônia"),
    "boa vista":        ("Boa Vista", "Roraima"),
    "palmas":           ("Palmas", "Tocantins"),
    "rio branco":       ("Rio Branco", "Acre"),
}


def _slug(text: str) -> str:
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c)).lower()


def _extrair_cidade(location_text: str) -> tuple[str, str]:
    t = _slug(location_text)
    for key, val in _CIDADES_BR.items():
        if key in t:
            return val
    return "Guarulhos", "São Paulo"


def _extrair_bairro(location_text: str, cidade: str) -> str:
    t = location_text.strip()
    cidade_slug = _slug(cidade)
    t_slug = _slug(t)
    if cidade_slug in t_slug:
        idx = t_slug.find(cidade_slug)
        t = (t[:idx] + t[idx + len(cidade_slug):]).strip(" ,")
    return t.strip().title() if t.strip() else ""


def _bairro_match(item_bairro: str, target: str) -> bool:
    """Retorna True se o bairro do item corresponde ao bairro alvo da busca.

    Normaliza acentos e capitalização. Usa correspondência por palavras
    (todas as palavras significativas do alvo devem aparecer no bairro do item),
    o que tolera abreviações e sufixos como 'Jardim Maia' ↔ 'Jardim Maia Guarulhos'.
    Itens sem bairro são sempre incluídos (não há dado para filtrar).
    """
    if not target:
        return True
    if not item_bairro:
        return True  # sem dado de bairro → inclui (raio_km decide pela distância)
    t = _slug(target)
    b = _slug(item_bairro)
    if t in b or b in t:
        return True
    words = [w for w in t.split() if len(w) > 2]
    return bool(words) and all(w in b for w in words)


# ══════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════
# CACHE
# ══════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def _modelo():
    return carregar_modelo()


@st.cache_data(ttl=86400, show_spinner=False)
def _geocode_cidade(cidade: str, estado: str) -> tuple[float, float]:
    from src.search.geocoder import geocode
    r = geocode(f"{cidade}, {estado}, Brasil")
    return r or (DEFAULT_LAT, DEFAULT_LON)


# ══════════════════════════════════════════════════════════════════════════
# PREVISÃO IA
# ══════════════════════════════════════════════════════════════════════════

def _preparar_features(item: dict) -> dict:
    lat = item.get("latitude")
    lon = item.get("longitude")
    if not lat or not lon:
        lat, lon = _geocode_cidade(
            item.get("cidade") or "guarulhos",
            item.get("estado") or "São Paulo",
        )
    return {
        "tipo":          item.get("tipo", "apartment"),
        "estado":        item.get("estado") or "São Paulo",
        "cidade":        _slug(item.get("cidade") or "guarulhos"),
        "quartos":       max(int(item.get("quartos") or 0), 0),
        "banheiros":     max(int(item.get("banheiros") or 0), 0),
        "area":          max(float(item.get("area") or 0), 18.0),
        "vagas_garagem": max(int(item.get("vagas_garagem") or 0), 0),
        "condominio":    float(item.get("condominio") or 0),
        "iptu":          float(item.get("iptu") or 0),
        "latitude":      float(lat),
        "longitude":     float(lon),
    }


def _prever(items: list[dict]) -> list[dict]:
    if not items:
        return []
    modelo = _modelo()
    feats = [_preparar_features(i) for i in items]
    df_feats = pd.DataFrame(feats)[config.FEATURES]
    preds = modelo.predict(df_feats).round(0).astype(int)
    return [{**item, "preco_previsto": int(pred)} for item, pred in zip(items, preds)]


def _eval(preco: float, previsto: float):
    ratio = previsto / max(preco, 1)
    if ratio > 1.10:
        return "good",   "bg-good",   "ev-good",   "✅ Bom negócio"
    if ratio < 0.90:
        return "pricey", "bg-pricey", "ev-pricey", "⚠️ Caro"
    return   "fair",  "bg-fair",   "ev-fair",   "🔶 Preço justo"


# ══════════════════════════════════════════════════════════════════════════
# CARD HTML
# ══════════════════════════════════════════════════════════════════════════

_SRC_LABEL: dict[str, tuple[str, str]] = {
    "zap":      ("ZAP Imóveis", "src-zap"),
    "vivareal": ("Viva Real",   "src-vivareal"),
}


def _card_html(item: dict) -> str:
    tipo      = item.get("tipo", "apartment")
    tipo_pt   = TIPOS_PT.get(tipo, "Imóvel")
    preco     = float(item.get("preco") or 0)
    previsto  = int(item.get("preco_previsto") or preco)
    fonte     = item.get("fonte") or "zap"
    negocio   = item.get("negocio") or "SALE"
    quartos   = int(item.get("quartos") or 0)
    banheiros = int(item.get("banheiros") or 0)
    area      = float(item.get("area") or 0)
    vagas     = int(item.get("vagas_garagem") or 0)
    suites    = int(item.get("suites") or 0)

    # Escapar todos os campos de texto que vêm de APIs externas
    titulo    = _html.escape((item.get("titulo") or "").strip())
    if not titulo:
        titulo = _html.escape(f"{tipo_pt} em {str(item.get('cidade','')).title()}")
    bairro    = _html.escape(str(item.get("bairro") or "").title())
    cidade    = _html.escape(str(item.get("cidade") or "").title())
    estado    = _html.escape(item.get("estado") or "")
    endereco  = _html.escape((item.get("endereco") or "").strip())
    descricao = _html.escape((item.get("descricao") or "").strip())
    link      = _html.escape((item.get("link") or ""), quote=True)

    local = bairro if bairro else cidade
    if bairro and cidade:
        local = f"{bairro}, {cidade}"
    if estado:
        local += f" — {estado}"

    _, badge_cls, eval_cls, eval_txt = _eval(preco, previsto)

    # Imagem — prefetch server-side como data URI evita bloqueio de Referer pelo CDN
    thumb = (item.get("thumbnail") or "").strip()
    if thumb.startswith("http"):
        # _img_cache() persiste entre re-runs (cache_resource); fallback = URL com no-referrer
        thumb_src = _img_cache().get(thumb) or thumb
        img_html = (
            f'<img src="{thumb_src}" alt="{titulo}" loading="eager" referrerpolicy="no-referrer" '
            f'onerror="this.style.display=\'none\';this.nextElementSibling.style.display=\'flex\'">'
            f'<div class="prop-img-ph" style="display:none">{TIPO_EMOJI.get(tipo,"🏠")}</div>'
        )
    else:
        img_html = f'<div class="prop-img-ph">{TIPO_EMOJI.get(tipo, "🏠")}</div>'

    src_lbl, src_cls = _SRC_LABEL.get(fonte, ("Imóvel", "src-zap"))

    feats = ""
    if quartos:   feats += f'<span class="prop-feat">🛏 {quartos}q</span>'
    if suites:    feats += f'<span class="prop-feat">✨ {suites} suíte{"s" if suites>1 else ""}</span>'
    if banheiros: feats += f'<span class="prop-feat">🚿 {banheiros}b</span>'
    if area:      feats += f'<span class="prop-feat">📐 {area:.0f}m²</span>'
    if vagas:     feats += f'<span class="prop-feat">🚗 {vagas}v</span>'

    preco_txt    = f"R$ {preco:,.0f}" if preco > 0 else "Sob consulta"
    prev_txt     = f"R$ {previsto:,.0f}"
    neg_cls      = "neg-sale" if negocio == "SALE" else "neg-rental"
    neg_lbl      = "Venda" if negocio == "SALE" else "Aluguel"
    price_pill   = f'<div class="price-pill">{preco_txt}</div>' if preco else ""

    link_html = (
        f'<a href="{link}" target="_blank" class="prop-link">Ver anúncio completo →</a>'
        if link else
        '<span class="prop-link prop-link-dis">Sem link</span>'
    )
    desc_html = f'<p class="prop-desc">{descricao}</p>' if descricao else ""
    addr_html = f'<p class="prop-local">&#127968; {endereco}</p>' if endereco else ""

    raw = (
        f'<div class="prop-card">'
        f'<div class="prop-img">'
        f'{img_html}'
        f'<div class="badge-ai {badge_cls}">{eval_txt}</div>'
        f'<div class="badge-src {src_cls}">{src_lbl}</div>'
        f'{price_pill}'
        f'<div class="negocio-pill {neg_cls}">{neg_lbl}</div>'
        f'</div>'
        f'<div class="prop-body">'
        f'<div class="prop-tipo">{tipo_pt}</div>'
        f'<p class="prop-titulo">{titulo}</p>'
        f'<p class="prop-local">&#128205; {local}</p>'
        f'{addr_html}'
        f'{desc_html}'
        f'<div class="prop-feats">{feats or "<span class=\"prop-feat\">Sem detalhes</span>"}</div>'
        f'<div class="prop-prices">'
        f'<div><div class="p-listed-v">{preco_txt}</div><div class="p-listed-l">anunciado</div></div>'
        f'<div><div class="p-ai-v">{prev_txt}</div><div class="p-ai-l">&#129302; IA prev&#234;</div></div>'
        f'</div>'
        f'<div class="prop-eval {eval_cls}">{eval_txt}</div>'
        f'{link_html}'
        f'</div>'
        f'</div>'
    )
    return raw


def _render_grid(items: list[dict], cols: int = 3):
    if not items:
        return
    rows = [items[i:i+cols] for i in range(0, len(items), cols)]
    for row in rows:
        columns = st.columns(cols)
        for col, item in zip(columns, row):
            with col:
                st.html(_card_html(item))


def _criar_mapa(items: list[dict], lat_c: float, lon_c: float) -> folium.Map:
    m = folium.Map(location=[lat_c, lon_c], zoom_start=13, tiles="CartoDB dark_matter")
    folium.Marker(
        [lat_c, lon_c],
        icon=folium.Icon(color="blue", icon="map-marker", prefix="fa"),
        tooltip="📍 Local buscado",
    ).add_to(m)
    cluster = MarkerCluster(
        options={"maxClusterRadius": 45, "disableClusteringAtZoom": 16}
    ).add_to(m)
    for item in items:
        lat = item.get("latitude")
        lon = item.get("longitude")
        if not lat or not lon:
            continue
        preco = float(item.get("preco") or 0)
        prev  = int(item.get("preco_previsto") or preco)
        ev_type, _, _, ev_txt = _eval(preco, prev)
        cor = {"good": "green", "fair": "orange", "pricey": "red"}.get(ev_type, "gray")
        tipo_pt  = TIPOS_PT.get(item.get("tipo", "apartment"), "Imóvel")
        link     = _html.escape((item.get("link") or ""), quote=True)
        bairro_p = _html.escape(str(item.get("bairro", "")).title())
        cidade_p = _html.escape(str(item.get("cidade", "")).title())
        link_tag = f'<a href="{link}" target="_blank">Ver anúncio →</a>' if link else ""
        popup = (
            f'<div style="min-width:220px;font-family:sans-serif;font-size:13px">'
            f'<b>{tipo_pt} · {bairro_p}, {cidade_p}</b><br>'
            f'🛏 {int(item.get("quartos",0))}q '
            f'🚿 {int(item.get("banheiros",0))}b '
            f'📐 {float(item.get("area",0)):.0f}m²<br>'
            f'💰 R$ {preco:,.0f} &nbsp; 🤖 R$ {prev:,.0f}<br>'
            f'{ev_txt}<br>{link_tag}</div>'
        )
        folium.Marker(
            [lat, lon],
            icon=folium.Icon(color=cor, icon="home", prefix="fa"),
            popup=folium.Popup(popup, max_width=280),
            tooltip=f"{tipo_pt} R$ {preco:,.0f}",
        ).add_to(cluster)
    return m


# ══════════════════════════════════════════════════════════════════════════
# GEOCODIFICAÇÃO DE IMÓVEIS (atribui lat/lon para marcadores no mapa)
# ══════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=86400, show_spinner=False)
def _geocode_bairro_cached(bairro: str, cidade: str, estado: str) -> tuple[float, float] | None:
    """Geocodifica bairro+cidade — resultado em cache por 24h."""
    from src.search.geocoder import geocode
    partes = [p for p in [bairro, cidade, estado] if p.strip()]
    if not partes:
        return None
    return geocode(", ".join(partes) + ", Brasil")


def _atribuir_coords(
    items: list[dict],
    fallback_lat: float,
    fallback_lon: float,
    max_novas_queries: int = 20,
) -> list[dict]:
    """Atribui latitude/longitude a itens sem coordenadas.

    Prioridade de coordenadas:
    1. Bairro + cidade  (geocodificado)
    2. Cidade apenas    (quando bairro não disponível ou geocoding falhou)
    3. Fallback genérico da busca  (último recurso)

    Jitter determinístico ~±200 m evita sobreposição de pins.
    """
    geo_cache: dict[tuple, tuple[float, float]] = {}
    novas = 0
    result = []

    def _geocode_key(key: tuple) -> tuple[float, float] | None:
        nonlocal novas
        if key in geo_cache:
            return geo_cache[key]
        if novas >= max_novas_queries:
            return None
        coords = _geocode_bairro_cached(*key)
        novas += 1
        if coords:
            geo_cache[key] = coords
        return coords

    for item in items:
        if item.get("latitude") and item.get("longitude"):
            result.append(item)
            continue

        bairro = (item.get("bairro") or "").strip().title()
        cidade = (item.get("cidade") or "").strip().title()
        estado = (item.get("estado") or "").strip()

        # Tenta bairro+cidade; se falhar, tenta só cidade; se falhar, usa fallback
        coords = None
        if bairro:
            coords = _geocode_key((bairro, cidade, estado))
        if not coords:
            coords = _geocode_key(("", cidade, estado))
        if not coords:
            coords = (fallback_lat, fallback_lon)

        geo_cache[(bairro, cidade, estado)] = coords

        lat_base, lon_base = coords
        h = int(hashlib.md5((item.get("id") or item.get("titulo") or "").encode()).hexdigest()[:8], 16)
        lat_jit = ((h & 0xFFFF) - 32768) * 0.002 / 32768
        lon_jit = (((h >> 16) & 0xFFFF) - 32768) * 0.002 / 32768

        result.append({**item, "latitude": lat_base + lat_jit, "longitude": lon_base + lon_jit})

    return result


# ══════════════════════════════════════════════════════════════════════════
# FILTRO PÓS-BUSCA POR BAIRRO
# ══════════════════════════════════════════════════════════════════════════



# ══════════════════════════════════════════════════════════════════════════
# BUSCA PARALELA NAS APIs REAIS
# ══════════════════════════════════════════════════════════════════════════

def _buscar_tudo(
    cidade: str, estado: str,
    tipos: list[str], negocio: str,
    quartos_min: int, preco_min: float, preco_max: float, area_min: float,
    query_raw: str = "",
    limit_por_fonte: int = 60,
) -> tuple[list[dict], list[dict]]:
    kw = dict(
        cidade=cidade, estado=estado, bairro="",
        tipos=tipos if tipos else ["apartment", "house"],
        negocio=negocio,
        quartos_min=quartos_min,
        preco_min=preco_min,
        preco_max=preco_max if preco_max < 4_900_000 else 0,
        area_min=area_min if area_min > 30 else 0,
        limit=limit_por_fonte,
    )

    def _zap():  return zapimoveis.buscar_zap(**kw)
    def _viva(): return zapimoveis.buscar_vivareal(**kw)

    results: dict[str, list[dict]] = {}
    fns = {"zap": _zap, "viva": _viva}

    with ThreadPoolExecutor(max_workers=4) as ex:
        futures = {ex.submit(fn): key for key, fn in fns.items()}
        for future in as_completed(futures):
            key = futures[future]
            try:
                results[key] = future.result() or []
            except Exception as _exc:
                _log(f"[BUSCAR] Erro na fonte '{key}': {type(_exc).__name__}: {_exc}")
                results[key] = []

    return (
        results.get("zap", []),
        results.get("viva", []),
    )


# ══════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════════════
def _init():
    defaults = {
        "geocoded":       (DEFAULT_LAT, DEFAULT_LON),
        "geocoded_label": "Guarulhos, SP",
        "poi_locs":       {},
        "qp":             None,
        "buscou":         False,
        "bairro_busca":   "",
        "zap_items":  [],
        "viva_items": [],
        "api_status": {},
        "aviso_bairro":   "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init()

# ══════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🏠 Lastro Imóveis")
    st.caption("Dados reais · IA de precificação")

    st.divider()
    st.markdown("### Filtros")

    negocio_sel = st.radio("Negócio", ["Venda", "Aluguel"], horizontal=True)
    negocio_api = "SALE" if negocio_sel == "Venda" else "RENTAL"

    tipos_sel = st.multiselect(
        "Tipo de imóvel", list(TIPOS_PT.keys()),
        default=["apartment", "house"],
        format_func=lambda x: TIPOS_PT[x],
    )
    quartos_r   = st.slider("Quartos",     0, 6, (0, 6))
    banheiros_r = st.slider("Banheiros",   0, 6, (0, 6))
    area_r      = st.slider("Área (m²)",  20, 600, (20, 600))
    if negocio_sel == "Aluguel":
        preco_r = st.slider("Aluguel/mês (R$)", 300, 30_000,
                            (300, 30_000), step=100, format="R$ %d",
                            help="Valor mensal do aluguel")
    else:
        preco_r = st.slider("Preço (R$)", 50_000, 5_000_000,
                            (50_000, 5_000_000), step=50_000, format="R$ %d")
    vagas_r     = st.slider("Vagas",       0, 6, (0, 4))
    # Aplica intenção de raio (definida pelo handler de busca quando detecta bairro)
    # antes do widget ser criado — única janela válida para alterar a chave do slider.
    if "_raio_next" in st.session_state:
        st.session_state["raio_km_slider"] = st.session_state["_raio_next"]
        del st.session_state["_raio_next"]
    raio_km     = st.slider("Raio de busca (km)", 1, 50, 15,
                            key="raio_km_slider",
                            help="Filtra imóveis pela distância do centro da busca. Bairros específicos: 3–5 km")

    st.divider()
    st.markdown("### 📍 Próximo a")
    dist_poi = st.slider("Dist. máxima (m)", 100, 2000, 500, step=100)
    c1, c2 = st.columns(2)
    with c1:
        p_super    = st.checkbox("🛒 Supermercado")
        p_escola   = st.checkbox("🏫 Escola")
        p_hospital = st.checkbox("🏥 Hospital")
    with c2:
        p_rest   = st.checkbox("🍽️ Restaurante")
        p_parque = st.checkbox("🌳 Parque")
        p_metro  = st.checkbox("🚇 Metrô/Trem")

    pois_sidebar = [c for c, s in [
        ("supermercado", p_super), ("escola", p_escola), ("hospital", p_hospital),
        ("restaurante", p_rest),   ("parque", p_parque), ("metro", p_metro),
    ] if s]

    try:
        meta = carregar_metadata()
        m_info = meta.get("metricas", {})
        with st.expander("📊 Modelo de IA"):
            st.metric("R²",   f"{m_info.get('r2', 0):.3f}")
            st.metric("MAPE", f"{m_info.get('mape_pct', 0):.1f}%")
            st.metric("Base", f"{m_info.get('n_treino', 0):,} imóveis")
    except Exception:
        pass

# ══════════════════════════════════════════════════════════════════════════
# HERO + BUSCA
# ══════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero">
  <div class="hero-title">Encontre seu imóvel ideal</div>
  <div class="hero-sub">ZAP Imóveis · Viva Real · Previsão de preço por IA</div>
</div>""", unsafe_allow_html=True)

with st.form("busca_form", clear_on_submit=False, border=False):
    col_q, col_btn = st.columns([5, 1])
    with col_q:
        query = st.text_input(
            "Buscar", label_visibility="collapsed",
            placeholder="Ex: apartamento 3 quartos Jardim Maia Guarulhos perto de escola",
            key="search_input",
        )
    with col_btn:
        buscar_btn = st.form_submit_button("🔍 Buscar", type="primary", use_container_width=True)

with st.expander("💡 Como buscar", expanded=False):
    st.markdown("""
| O que digitar | Exemplos |
|---|---|
| Tipo | `apartamento`, `casa`, `terreno`, `comercial` |
| Quartos | `3 quartos`, `2 dormitórios`, `1 suíte` |
| Localização | `Guarulhos`, `São Paulo`, `Belo Horizonte`, `Curitiba` |
| Proximidade | `perto de escola`, `próximo ao metrô`, `perto de mercado` |

Altere **Venda / Aluguel** e demais filtros na barra lateral antes de buscar.
    """)

st.page_link("pages/Analise.py", label="📈 Ver análise do mercado")

# ══════════════════════════════════════════════════════════════════════════
# EXECUTAR BUSCA
# ══════════════════════════════════════════════════════════════════════════
if buscar_btn and query.strip():
    qp = parse_query(query)
    st.session_state.qp = qp
    st.session_state.buscou = True

    loc_text = qp.location_text or query

    with st.spinner("Pesquisando imóveis..."):
        _log(f"\n{'='*60}")
        _log(f"[APP] Nova busca: query={query!r}  loc_text={loc_text!r}")
        geo = geocode_detalhado(loc_text)

        lat    = geo.get("lat", DEFAULT_LAT)
        lon    = geo.get("lon", DEFAULT_LON)
        cidade = geo.get("cidade") or ""
        estado = geo.get("estado") or ""
        bairro = geo.get("bairro") or ""

        _log(f"[APP] Geocoding resultado: cidade={cidade!r} estado={estado!r} lat={lat} lon={lon}")

        if not cidade:
            cidade, estado = _extrair_cidade(loc_text)
            _log(f"[APP] Cidade extraída do texto: cidade={cidade!r} estado={estado!r}")

        _log(f"[APP] Buscando por cidade={cidade!r} estado={estado!r}")

        st.session_state.geocoded       = (lat, lon)
        st.session_state.bairro_busca   = ""
        st.session_state.geocoded_label = cidade or loc_text.title()

        tipos_eff = qp.tipo_hint or tipos_sel or list(TIPOS_PT.keys())
        _log(f"[APP] tipos_eff={tipos_eff}  negocio={negocio_api!r}")

        pois_eff = list(set(pois_sidebar + (qp.poi_hints or [])))
        if pois_eff:
            _buscar_pois_cached(lat, lon, int(raio_km * 1000 + 3000), tuple(sorted(pois_eff)))
        st.session_state.poi_locs = {}

        zap_r, viva_r = _buscar_tudo(
            cidade=cidade, estado=estado,
            tipos=tipos_eff,
            negocio=negocio_api,
            quartos_min=qp.quartos_hint or quartos_r[0],
            preco_min=float(preco_r[0]),
            preco_max=float(preco_r[1]),
            area_min=float(area_r[0]),
            query_raw=query,
        )
        _log(f"[APP] Resultados brutos — ZAP: {len(zap_r)}  Viva: {len(viva_r)}")
        _log(f"{'='*60}\n")

    st.session_state.zap_items  = zap_r
    st.session_state.viva_items = viva_r
    st.session_state.api_status = {
        "ZAP Imóveis": len(zap_r) > 0,
        "Viva Real":   len(viva_r) > 0,
    }

    total = sum(map(len, [zap_r, viva_r]))
    if total:
        st.toast(f"✅ {total} imóveis reais encontrados!", icon="🏠")
        todos_items = zap_r + viva_r
        with st.spinner("🖼️ Carregando fotos dos imóveis..."):
            _prefetch_thumbnails(todos_items, max_imgs=60)
        todos_geo = _atribuir_coords(todos_items, lat, lon)
        n = len(zap_r)
        st.session_state.zap_items  = todos_geo[:n]
        st.session_state.viva_items = todos_geo[n:]
    else:
        st.toast("APIs responderam 0 resultados — tente outra cidade.", icon="⚠️")

# ══════════════════════════════════════════════════════════════════════════
# WELCOME STATE
# ══════════════════════════════════════════════════════════════════════════
if not st.session_state.buscou:
    st.markdown("""
    <div class="welcome">
      <div class="welcome-icon">🏠</div>
      <div class="welcome-t">Busque imóveis reais agora</div>
      <div class="welcome-s">
        Digite uma cidade, bairro ou endereço acima.<br>
        O Lastro vai buscar em tempo real no <b>ZAP Imóveis</b> e <b>Viva Real</b>,
        e usar IA para avaliar se o preço está
        <b style="color:#34d399">bom</b>,
        <b style="color:#fbbf24">justo</b> ou
        <b style="color:#f87171">caro</b>.
      </div>
    </div>""", unsafe_allow_html=True)
    st.stop()

lat_c, lon_c = st.session_state.geocoded
qp = st.session_state.qp

# ══════════════════════════════════════════════════════════════════════════
# FILTROS PÓS-BUSCA E PREDIÇÃO
# ══════════════════════════════════════════════════════════════════════════

def _filtrar(items: list[dict]) -> list[dict]:
    q_min, q_max = quartos_r
    b_min, b_max = banheiros_r
    lat_c, lon_c = st.session_state.geocoded
    check_raio   = st.session_state.buscou
    out = []
    n_tipo = n_quartos = n_banheiros = n_area = n_preco = n_vagas = n_negocio = n_raio = 0
    for i in items:
        if tipos_sel and i.get("tipo") not in tipos_sel:
            n_tipo += 1; continue
        if not (q_min <= int(i.get("quartos") or 0) <= q_max):
            n_quartos += 1; continue
        if not (b_min <= int(i.get("banheiros") or 0) <= b_max):
            n_banheiros += 1; continue
        a = float(i.get("area") or 0)
        if a > 0 and not (area_r[0] <= a <= area_r[1]):
            n_area += 1; continue
        p = float(i.get("preco") or 0)
        if p > 0 and not (preco_r[0] <= p <= preco_r[1]):
            n_preco += 1; continue
        v = int(i.get("vagas_garagem") or 0)
        if vagas_r[0] > 0 and not (vagas_r[0] <= v <= vagas_r[1]):
            n_vagas += 1; continue
        neg = i.get("negocio") or "SALE"
        if negocio_api == "SALE" and neg == "RENTAL":
            n_negocio += 1; continue
        if negocio_api == "RENTAL" and neg == "SALE":
            n_negocio += 1; continue
        if check_raio:
            item_lat = i.get("latitude")
            item_lon = i.get("longitude")
            if item_lat is not None and item_lon is not None:
                if haversine_km(lat_c, lon_c, item_lat, item_lon) > raio_km:
                    n_raio += 1; continue
        out.append(i)
    if items:
        _log(f"[FILTRAR] Entrada: {len(items)} → Saída: {len(out)} | "
             f"tipo={n_tipo} quartos={n_quartos} banheiros={n_banheiros} "
             f"area={n_area} preco={n_preco} vagas={n_vagas} "
             f"negocio={n_negocio} raio={n_raio} "
             f"(raio_km={raio_km}, check_raio={check_raio})")
    return out


def _filtrar_poi(items: list[dict]) -> list[dict]:
    pois_eff = list(set(pois_sidebar + (qp.poi_hints if qp else [])))
    if not pois_eff or not items or not st.session_state.buscou:
        return items
    lat_c, lon_c = st.session_state.geocoded
    raio_m = int(raio_km * 1000 + 3000)
    poi_locs = _buscar_pois_cached(lat_c, lon_c, raio_m, tuple(sorted(pois_eff)))
    if not poi_locs:
        return items
    df = pd.DataFrame(items)
    tem = df[df["latitude"].notna() & df["longitude"].notna()]
    if tem.empty:
        return items
    filtrado = filtrar_por_poi(tem, poi_locs, pois_eff, dist_poi)
    return filtrado.to_dict("records")


zap_f      = _prever(_filtrar_poi(_filtrar(st.session_state.zap_items)))
viva_f     = _prever(_filtrar_poi(_filtrar(st.session_state.viva_items)))
all_items  = zap_f + viva_f
total_res  = len(all_items)

# ══════════════════════════════════════════════════════════════════════════
# MÉTRICAS
# ══════════════════════════════════════════════════════════════════════════
st.divider()

status = st.session_state.api_status
if status:
    chips = "".join(
        f'<span class="api-chip {"chip-ok" if ok else "chip-fail"}">'
        f'{"✓" if ok else "✗"} {name}</span>'
        for name, ok in status.items()
    )
    st.markdown(f'<div class="api-status">{chips}</div>', unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
precos_reais  = [i.get("preco", 0) for i in all_items if i.get("preco", 0) > 0]
precos_previs = [i.get("preco_previsto", 0) for i in all_items if i.get("preco_previsto", 0) > 0]
c1.metric("🏘️ Imóveis reais", f"{total_res:,}")
c2.metric("💰 Preço médio",
          f"R$ {sum(precos_reais)/max(len(precos_reais),1):,.0f}" if precos_reais else "—")
c3.metric("🤖 Previsão IA média",
          f"R$ {sum(precos_previs)/max(len(precos_previs),1):,.0f}" if precos_previs else "—")
c4.metric("📍 Buscado em", st.session_state.geocoded_label)

if total_res > 0:
    _cta_col, _btn_col = st.columns([5, 1])
    with _cta_col:
        st.markdown(f"""
        <div class="ana-cta">
          <div class="ana-cta-icon">📈</div>
          <div class="ana-cta-body">
            <div class="ana-cta-t">{total_res} imóveis prontos para análise</div>
            <div class="ana-cta-s">Histogramas de preço · Área × Preço · Preço/m² por tipo · Top bairros · Real vs Previsão IA</div>
          </div>
        </div>""", unsafe_allow_html=True)
    with _btn_col:
        st.write("")
        st.write("")
        st.page_link("pages/Analise.py", label="📈 Abrir Análise", use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════
st.divider()
tab1, tab2, tab3 = st.tabs(["🏘️ Imóveis", "🗺️ Mapa", "📊 Tabela"])

with tab1:
    if not all_items:
        st.markdown("""
        <div class="no-res">
          <div class="no-res-icon">🔍</div>
          <div class="no-res-t">Nenhum imóvel real encontrado</div>
          <div class="no-res-s">
            Tente uma cidade maior (São Paulo, Belo Horizonte, Curitiba, Rio de Janeiro),
            remova filtros de quartos/preço ou alterne Venda ↔ Aluguel.
          </div>
        </div>""", unsafe_allow_html=True)
    else:
        if zap_f:
            st.markdown(f"""
            <div class="sec-hdr">
              <span class="sec-hdr-t">ZAP Imóveis</span>
              <span class="sec-hdr-c">{len(zap_f)} anúncios</span>
              <span class="sec-badge src-zap-badge">zapimoveis.com.br</span>
            </div>""", unsafe_allow_html=True)
            _render_grid(zap_f[:60])

        if viva_f:
            if zap_f:
                st.divider()
            st.markdown(f"""
            <div class="sec-hdr">
              <span class="sec-hdr-t">Viva Real</span>
              <span class="sec-hdr-c">{len(viva_f)} anúncios</span>
              <span class="sec-badge src-vivareal-badge">vivareal.com.br</span>
            </div>""", unsafe_allow_html=True)
            _render_grid(viva_f[:60])


with tab2:
    mapa_items = [i for i in all_items if i.get("latitude") and i.get("longitude")]
    st.markdown(
        f"**{len(mapa_items)} imóveis no mapa** · "
        "Clique no marcador para ver detalhes e acessar o link do anúncio"
    )
    if mapa_items:
        mapa = _criar_mapa(mapa_items[:250], lat_c, lon_c)
        st_folium(mapa, height=600, use_container_width=True)
        st.caption(
            "🟢 Bom negócio · 🟠 Preço justo · 🔴 Caro · 🔵 Local buscado · "
            "Coordenadas aproximadas por bairro — Mapa: CartoDB Dark Matter"
        )
    else:
        st.info(
            "Faça uma busca para ver os imóveis no mapa.",
            icon="🗺️",
        )

with tab3:
    if not all_items:
        st.info("Faça uma busca para ver a tabela.")
    else:
        cols_show = [
            "titulo", "tipo", "bairro", "cidade", "quartos", "banheiros",
            "area", "vagas_garagem", "preco", "preco_previsto", "negocio", "fonte", "link",
        ]
        rename = {
            "titulo": "Título", "tipo": "Tipo", "bairro": "Bairro", "cidade": "Cidade",
            "quartos": "Q", "banheiros": "B", "area": "Área m²", "vagas_garagem": "Vagas",
            "preco": "Preço", "preco_previsto": "Previsão IA",
            "negocio": "Negócio", "fonte": "Fonte", "link": "Link",
        }
        df_tab = pd.DataFrame(all_items)
        if "tipo" in df_tab.columns:
            df_tab["tipo"] = df_tab["tipo"].map(TIPOS_PT).fillna(df_tab["tipo"])
        if "negocio" in df_tab.columns:
            df_tab["negocio"] = df_tab["negocio"].apply(
                lambda x: "Venda" if x == "SALE" else ("Aluguel" if x == "RENTAL" else x)
            )
        cols_exist = [c for c in cols_show if c in df_tab.columns]
        st.dataframe(
            df_tab[cols_exist].rename(columns=rename).style.format({
                "Preço":       "R$ {:,.0f}",
                "Previsão IA": "R$ {:,.0f}",
                "Área m²":     "{:.0f}",
            }),
            use_container_width=True, height=440,
        )
        csv = df_tab.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Baixar CSV",
            csv, "lastro_imoveis.csv", "text/csv",
            use_container_width=True,
        )
