"""
Lastro — Busca e Previsão de Preços de Imóveis (dados reais)
Subir: streamlit run app.py
"""
from __future__ import annotations

import base64
import hashlib
import html as _html
import math
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests as _requests

import folium
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium

load_dotenv()

from src import config
from src.external import mercadolivre as ml_api
from src.external import zapimoveis
from src.models.predict import carregar_metadata, carregar_modelo
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
    "mercadolibre": "https://www.mercadolivre.com.br/",
    "mlstatic":     "https://www.mercadolivre.com.br/",
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


# ══════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Lastro — Imóveis",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════
# CSS
# ══════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:ital,opsz,wght@0,14..32,300;0,14..32,400;0,14..32,500;0,14..32,600;0,14..32,700;0,14..32,800&display=swap');

html,body,[class*="css"],.stApp{font-family:'Inter',system-ui,sans-serif!important}
#MainMenu,footer,.stDeployButton{visibility:hidden;display:none}
.stApp{background:#07070f!important}

[data-testid="stSidebar"]{background:#0b0b18!important;border-right:1px solid rgba(255,255,255,.06)!important}
[data-testid="stSidebar"] label,[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span{color:#94a3b8!important}

.stTextInput>div>div>input{
  background:#111127!important;border:1.5px solid rgba(255,255,255,.1)!important;
  border-radius:14px!important;color:#f1f5f9!important;font-size:16px!important;
  padding:14px 18px!important;height:auto!important;transition:border .2s,box-shadow .2s!important}
.stTextInput>div>div>input:focus{
  border-color:#3b82f6!important;box-shadow:0 0 0 4px rgba(59,130,246,.18)!important;outline:none!important}
.stTextInput>div>div>input::placeholder{color:#334155!important}

[data-testid="stBaseButton-primary"]{
  background:linear-gradient(135deg,#3b82f6,#6366f1)!important;border:none!important;
  border-radius:14px!important;color:#fff!important;font-weight:700!important;
  font-size:15px!important;padding:13px 24px!important;
  box-shadow:0 4px 20px rgba(59,130,246,.35)!important;transition:all .2s!important}
[data-testid="stBaseButton-primary"]:hover{
  opacity:.9!important;transform:translateY(-2px)!important;
  box-shadow:0 8px 30px rgba(59,130,246,.5)!important}

[data-testid="stBaseButton-secondary"]{
  background:rgba(255,255,255,.05)!important;border:1px solid rgba(255,255,255,.1)!important;
  border-radius:10px!important;color:#94a3b8!important;transition:all .18s!important}
[data-testid="stBaseButton-secondary"]:hover{
  background:rgba(255,255,255,.09)!important;color:#e2e8f0!important}

[data-testid="metric-container"]{
  background:#0f1120!important;border:1px solid rgba(255,255,255,.07)!important;
  border-radius:14px!important;padding:18px 22px!important}
[data-testid="stMetricValue"]{color:#f1f5f9!important;font-weight:800!important}
[data-testid="stMetricLabel"]{color:#475569!important}

.stTabs [data-baseweb="tab-list"]{
  background:#0d0d1f!important;border-radius:14px!important;padding:5px!important;
  border:1px solid rgba(255,255,255,.07)!important;gap:4px!important}
.stTabs [data-baseweb="tab"]{
  color:#475569!important;font-weight:600!important;border-radius:10px!important;
  padding:9px 20px!important;transition:all .2s!important}
.stTabs [aria-selected="true"]{background:rgba(59,130,246,.18)!important;color:#60a5fa!important}

[data-testid="stExpander"]{background:#0d0d1f!important;
  border:1px solid rgba(255,255,255,.07)!important;border-radius:12px!important}
[data-baseweb="tag"]{background:rgba(59,130,246,.18)!important;color:#60a5fa!important;
  border:1px solid rgba(59,130,246,.3)!important}
[data-baseweb="select"] [data-baseweb="select-control"]{background:#111127!important;
  border:1px solid rgba(255,255,255,.1)!important;border-radius:10px!important}

hr{border-color:rgba(255,255,255,.06)!important}
::-webkit-scrollbar{width:5px;height:5px}
::-webkit-scrollbar-track{background:#07070f}
::-webkit-scrollbar-thumb{background:#1e2035;border-radius:3px}

/* ══ HERO ══ */
.hero{text-align:center;padding:36px 0 20px}
.hero-title{font-size:40px;font-weight:800;
  background:linear-gradient(135deg,#f1f5f9 0%,#60a5fa 45%,#a78bfa 100%);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;
  line-height:1.15;margin:0 0 10px}
.hero-sub{font-size:15px;color:#475569;margin:0}

/* ══ PROPERTY CARD ══ */
.prop-card{
  background:#0d0d1f;border:1px solid rgba(255,255,255,.07);border-radius:18px;
  overflow:hidden;transition:all .28s cubic-bezier(.4,0,.2,1);
  display:flex;flex-direction:column;height:100%;margin-bottom:4px}
.prop-card:hover{
  border-color:rgba(59,130,246,.45);
  box-shadow:0 16px 50px rgba(0,0,0,.6),0 0 0 1px rgba(59,130,246,.2);
  transform:translateY(-5px)}

.prop-img{position:relative;height:210px;overflow:hidden;flex-shrink:0;background:#0a0a1a}
.prop-img img{width:100%;height:100%;object-fit:cover;display:block;transition:transform .45s ease}
.prop-card:hover .prop-img img{transform:scale(1.07)}
.prop-img-ph{width:100%;height:100%;display:flex;align-items:center;justify-content:center;
  font-size:56px;background:linear-gradient(135deg,#141428,#0f1f3d)}

.badge-ai{position:absolute;top:12px;left:12px;padding:5px 12px;border-radius:20px;
  font-size:11px;font-weight:700;letter-spacing:.5px;
  backdrop-filter:blur(10px);-webkit-backdrop-filter:blur(10px)}
.bg-good  {background:rgba(16,185,129,.22);color:#34d399;border:1px solid rgba(52,211,153,.3)}
.bg-fair  {background:rgba(245,158,11,.22);color:#fbbf24;border:1px solid rgba(251,191,36,.3)}
.bg-pricey{background:rgba(239,68,68,.22);color:#f87171;border:1px solid rgba(248,113,113,.3)}

.badge-src{position:absolute;top:12px;right:12px;padding:4px 9px;border-radius:7px;
  font-size:10px;font-weight:700;letter-spacing:.6px}
.src-zap     {background:rgba(255,93,0,.18);color:#fb923c;border:1px solid rgba(251,146,60,.25)}
.src-vivareal{background:rgba(124,58,237,.18);color:#a78bfa;border:1px solid rgba(167,139,250,.25)}
.src-ml      {background:rgba(255,196,0,.18);color:#fbbf24;border:1px solid rgba(251,191,36,.25)}

.price-pill{position:absolute;bottom:12px;right:12px;
  background:rgba(0,0,0,.78);backdrop-filter:blur(12px);
  padding:5px 14px;border-radius:20px;color:#fff;font-weight:800;font-size:13px;white-space:nowrap}
.negocio-pill{position:absolute;bottom:12px;left:12px;
  padding:4px 10px;border-radius:20px;font-size:10px;font-weight:700;letter-spacing:.5px;
  backdrop-filter:blur(10px)}
.neg-sale  {background:rgba(59,130,246,.22);color:#60a5fa;border:1px solid rgba(96,165,250,.3)}
.neg-rental{background:rgba(16,185,129,.22);color:#34d399;border:1px solid rgba(52,211,153,.3)}

.prop-body{padding:15px 16px 16px;flex:1;display:flex;flex-direction:column;gap:8px}
.prop-tipo{display:inline-flex;align-items:center;font-size:10px;font-weight:700;
  letter-spacing:1.1px;text-transform:uppercase;color:#60a5fa;
  background:rgba(59,130,246,.12);padding:3px 9px;border-radius:5px;width:fit-content}
.prop-titulo{font-size:13.5px;font-weight:600;color:#e2e8f0;line-height:1.5;margin:0;
  display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden}
.prop-local{font-size:12px;color:#475569;margin:0;display:flex;align-items:center;gap:4px}
.prop-desc{font-size:12px;color:#334155;line-height:1.55;margin:0;
  display:-webkit-box;-webkit-line-clamp:3;-webkit-box-orient:vertical;overflow:hidden}
.prop-feats{display:flex;flex-wrap:wrap;gap:5px}
.prop-feat{font-size:12px;color:#94a3b8;background:rgba(255,255,255,.05);
  padding:4px 9px;border-radius:8px;white-space:nowrap}

.prop-prices{display:flex;justify-content:space-between;align-items:center;
  padding:11px 13px;background:rgba(255,255,255,.03);
  border:1px solid rgba(255,255,255,.05);border-radius:11px;margin-top:auto}
.p-listed-v{font-size:19px;font-weight:800;color:#f1f5f9;line-height:1.2}
.p-listed-l{font-size:10px;color:#334155;font-weight:500;margin-top:2px}
.p-ai-v{font-size:14px;font-weight:700;color:#60a5fa;text-align:right;line-height:1.2}
.p-ai-l{font-size:10px;color:#334155;font-weight:500;text-align:right;margin-top:2px}

.prop-eval{display:flex;align-items:center;justify-content:center;
  font-size:12px;font-weight:700;padding:7px;border-radius:9px;gap:5px}
.ev-good  {background:rgba(16,185,129,.1);color:#34d399;border:1px solid rgba(52,211,153,.15)}
.ev-fair  {background:rgba(245,158,11,.1);color:#fbbf24;border:1px solid rgba(251,191,36,.15)}
.ev-pricey{background:rgba(239,68,68,.1);color:#f87171;border:1px solid rgba(248,113,113,.15)}

.prop-link{display:flex;align-items:center;justify-content:center;gap:6px;padding:10px;
  background:linear-gradient(135deg,rgba(59,130,246,.12),rgba(99,102,241,.12));
  border:1px solid rgba(59,130,246,.22);border-radius:11px;
  color:#60a5fa!important;text-decoration:none!important;font-size:13px;font-weight:600;
  transition:all .2s}
.prop-link:hover{background:linear-gradient(135deg,rgba(59,130,246,.22),rgba(99,102,241,.22));
  border-color:rgba(59,130,246,.4);color:#93c5fd!important}
.prop-link-dis{opacity:.3;cursor:default;pointer-events:none}

/* ══ SECTION HEADER ══ */
.sec-hdr{display:flex;align-items:center;gap:10px;margin:22px 0 14px;
  padding-bottom:11px;border-bottom:1px solid rgba(255,255,255,.06)}
.sec-hdr-t{font-size:18px;font-weight:700;color:#e2e8f0;margin:0}
.sec-hdr-c{font-size:12px;color:#475569;background:rgba(255,255,255,.05);
  padding:3px 11px;border-radius:20px}
.sec-badge{font-size:11px;padding:3px 10px;border-radius:20px;font-weight:700}
.src-zap-badge     {background:rgba(255,93,0,.15);color:#fb923c}
.src-vivareal-badge{background:rgba(124,58,237,.15);color:#a78bfa}
.src-ml-badge      {background:rgba(255,196,0,.15);color:#fbbf24}

/* ══ WELCOME / EMPTY ══ */
.welcome{text-align:center;padding:80px 20px 60px}
.welcome-icon{font-size:72px;margin-bottom:16px}
.welcome-t{font-size:24px;font-weight:700;color:#475569;margin-bottom:10px}
.welcome-s{font-size:15px;color:#334155;max-width:540px;margin:0 auto;line-height:1.8}

.no-res{text-align:center;padding:60px 20px}
.no-res-icon{font-size:56px;margin-bottom:14px}
.no-res-t{font-size:20px;font-weight:700;color:#475569;margin-bottom:8px}
.no-res-s{font-size:14px;color:#334155}

/* ══ API STATUS CHIPS ══ */
.api-status{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:8px}
.api-chip{font-size:12px;padding:4px 11px;border-radius:20px;font-weight:600}
.chip-ok  {background:rgba(16,185,129,.12);color:#34d399;border:1px solid rgba(52,211,153,.2)}
.chip-fail{background:rgba(100,116,139,.12);color:#475569;border:1px solid rgba(100,116,139,.2)}
</style>
""", unsafe_allow_html=True)

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
    "zap":          ("ZAP Imóveis",  "src-zap"),
    "vivareal":     ("Viva Real",    "src-vivareal"),
    "mercadolivre": ("MercadoLivre", "src-ml"),
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

@st.cache_data(ttl=86400)
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
    max_novas_queries: int = 8,
) -> list[dict]:
    """Atribui latitude/longitude a itens sem coordenadas.

    Geocodifica cada grupo único (bairro, cidade, estado) com jitter deterministico
    para que pins não se sobreponham no mapa.
    """
    geo_cache: dict[tuple, tuple[float, float]] = {}
    novas = 0
    result = []

    for item in items:
        if item.get("latitude") and item.get("longitude"):
            result.append(item)
            continue

        bairro = (item.get("bairro") or "").strip().title()
        cidade = (item.get("cidade") or "").strip().title()
        estado = (item.get("estado") or "").strip()
        key = (bairro, cidade, estado)

        if key not in geo_cache:
            if novas < max_novas_queries:
                coords = _geocode_bairro_cached(bairro, cidade, estado)
                novas += 1
            else:
                coords = None
            geo_cache[key] = coords or (fallback_lat, fallback_lon)

        lat_base, lon_base = geo_cache[key]

        # Jitter deterministico (~±200 m) por item para não empilhar pins
        h = int(hashlib.md5((item.get("id") or item.get("titulo") or "").encode()).hexdigest()[:8], 16)
        lat_jit = ((h & 0xFFFF) - 32768) * 0.002 / 32768
        lon_jit = (((h >> 16) & 0xFFFF) - 32768) * 0.002 / 32768

        result.append({**item, "latitude": lat_base + lat_jit, "longitude": lon_base + lon_jit})

    return result


# ══════════════════════════════════════════════════════════════════════════
# FILTRO PÓS-BUSCA POR BAIRRO
# ══════════════════════════════════════════════════════════════════════════

def _filtrar_bairro(items: list[dict], bairro_alvo: str) -> list[dict]:
    """Descarta imóveis cujo bairro não corresponde ao bairro buscado.

    Mantém item se:
    - bairro do item contém ou está contido no bairro buscado (match parcial)
    - item não tem bairro, mas o título menciona o bairro
    Descarta: item tem bairro diferente do buscado.
    Fallback: se o filtro zerar resultados, retorna lista original.
    """
    alvo = _slug(bairro_alvo)
    if not alvo:
        return items
    mantidos: list[dict] = []
    for it in items:
        b = _slug(it.get("bairro") or "")
        t = _slug(it.get("titulo") or "") + " " + _slug(it.get("descricao") or "")
        if not b:
            if alvo in t:
                mantidos.append(it)
        elif alvo in b or b in alvo:
            mantidos.append(it)
    return mantidos if mantidos else items


# ══════════════════════════════════════════════════════════════════════════
# BUSCA PARALELA NAS APIs REAIS
# ══════════════════════════════════════════════════════════════════════════

def _buscar_tudo(
    cidade: str, estado: str, bairro: str,
    tipos: list[str], negocio: str,
    quartos_min: int, preco_min: float, preco_max: float, area_min: float,
    query_raw: str = "",
    limit_por_fonte: int = 24,
) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    kw = dict(
        cidade=cidade, estado=estado, bairro=bairro,
        tipos=tipos if tipos else ["apartment", "house"],
        negocio=negocio,
        quartos_min=quartos_min,
        preco_min=preco_min,
        preco_max=preco_max if preco_max < 4_900_000 else 0,
        area_min=area_min if area_min > 30 else 0,
        limit=limit_por_fonte,
    )

    def _zap():   return zapimoveis.buscar_zap(**kw)
    def _viva():  return zapimoveis.buscar_vivareal(**kw)
    def _ml_pub():
        tipo_pt = TIPOS_PT.get(tipos[0], "") if tipos else ""
        q = " ".join(filter(None, [
            tipo_pt, bairro or cidade,
            f"{quartos_min} quartos" if quartos_min else "",
        ]))
        return ml_api.buscar_publico(q or cidade, limit=limit_por_fonte)
    def _ml_auth():
        if not ml_api.esta_autenticado():
            return []
        tipo_pt = TIPOS_PT.get(tipos[0], "") if tipos else ""
        q = " ".join(filter(None, [tipo_pt, bairro or cidade]))
        return ml_api.buscar(q or cidade, limit=limit_por_fonte)

    results: dict[str, list[dict]] = {}
    fns = {"zap": _zap, "viva": _viva, "ml_pub": _ml_pub, "ml_auth": _ml_auth}

    with ThreadPoolExecutor(max_workers=4) as ex:
        futures = {ex.submit(fn): key for key, fn in fns.items()}
        for future in as_completed(futures):
            key = futures[future]
            try:
                results[key] = future.result() or []
            except Exception:
                results[key] = []

    return (
        results.get("zap", []),
        results.get("viva", []),
        results.get("ml_pub", []),
        results.get("ml_auth", []),
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
        "zap_items":      [],
        "viva_items":     [],
        "ml_pub_items":   [],
        "ml_auth_items":  [],
        "api_status":     {},
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

    if ml_api.esta_autenticado():
        st.success("✅ Mercado Livre conectado", icon="🔗")
    elif ml_api.get_credentials():
        url = ml_api.auth_url()
        st.markdown(
            f'<a href="{url}" target="_blank" style="display:block;text-align:center;'
            f'padding:8px;background:rgba(255,196,0,.12);border:1px solid rgba(255,196,0,.25);'
            f'border-radius:10px;color:#fbbf24;text-decoration:none;font-size:13px;font-weight:600">'
            f'🔗 Conectar Mercado Livre</a>',
            unsafe_allow_html=True,
        )
        code = st.text_input("Código de autorização:", key="ml_code",
                             label_visibility="collapsed",
                             placeholder="Cole o código de autorização aqui...")
        if code and st.button("Ativar", key="ml_activate"):
            if ml_api.trocar_codigo(code.strip()):
                st.success("Conectado!")
                st.rerun()
            else:
                st.error("Código inválido")
    else:
        st.info("Adicione credenciais ML gratuitas no `.env` para acesso autenticado.", icon="ℹ️")

    st.divider()
    st.markdown("### Filtros")

    negocio_sel = st.radio("Negócio", ["Venda", "Aluguel"], horizontal=True)
    negocio_api = "SALE" if negocio_sel == "Venda" else "RENTAL"

    tipos_sel = st.multiselect(
        "Tipo de imóvel", list(TIPOS_PT.keys()),
        default=["apartment", "house"],
        format_func=lambda x: TIPOS_PT[x],
    )
    quartos_r   = st.slider("Quartos",     0, 6, (1, 4))
    banheiros_r = st.slider("Banheiros",   0, 6, (1, 4))
    area_r      = st.slider("Área (m²)",  20, 600, (30, 350))
    preco_r     = st.slider("Preço (R$)", 50_000, 5_000_000,
                            (100_000, 2_500_000), step=50_000, format="R$ %d")
    vagas_r     = st.slider("Vagas",       0, 6, (0, 4))
    raio_km     = st.slider("Raio no mapa (km)", 1, 50, 15)

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
  <div class="hero-sub">ZAP Imóveis · Viva Real · Mercado Livre · Previsão de preço por IA</div>
</div>""", unsafe_allow_html=True)

col_q, col_btn = st.columns([5, 1])
with col_q:
    query = st.text_input(
        "Buscar", label_visibility="collapsed",
        placeholder="Ex: apartamento 3 quartos Jardim Maia Guarulhos perto de escola",
        key="search_input",
    )
with col_btn:
    buscar_btn = st.button("🔍 Buscar", type="primary", use_container_width=True)

with st.expander("💡 Como buscar", expanded=False):
    st.markdown("""
| O que digitar | Exemplos |
|---|---|
| Tipo | `apartamento`, `casa`, `terreno`, `comercial` |
| Quartos | `3 quartos`, `2 dormitórios`, `1 suíte` |
| Localização | `Jardim Maia Guarulhos`, `Moema São Paulo`, `Savassi Belo Horizonte` |
| Proximidade | `perto de escola`, `próximo ao metrô`, `perto de mercado` |

Altere **Venda / Aluguel** e demais filtros na barra lateral antes de buscar.
    """)

# ══════════════════════════════════════════════════════════════════════════
# EXECUTAR BUSCA
# ══════════════════════════════════════════════════════════════════════════
if buscar_btn and query.strip():
    qp = parse_query(query)
    st.session_state.qp = qp
    st.session_state.buscou = True

    loc_text = qp.location_text or query
    with st.spinner(f"📍 Localizando «{loc_text}»..."):
        geo = geocode_detalhado(loc_text)

    lat     = geo.get("lat", DEFAULT_LAT)
    lon     = geo.get("lon", DEFAULT_LON)
    cidade  = geo.get("cidade") or ""
    estado  = geo.get("estado") or ""
    bairro  = geo.get("bairro") or ""

    # Fallback texto se geocoder não retornou cidade (raro)
    if not cidade:
        cidade, estado = _extrair_cidade(loc_text)
        bairro = _extrair_bairro(loc_text, cidade)

    st.session_state.geocoded       = (lat, lon)
    st.session_state.geocoded_label = (
        ", ".join(filter(None, [bairro, cidade])) or loc_text.title()
    )

    tipos_eff = qp.tipo_hint or tipos_sel or list(TIPOS_PT.keys())

    pois_eff = list(set(pois_sidebar + (qp.poi_hints or [])))
    if pois_eff:
        with st.spinner(f"📍 Buscando POIs: {', '.join(pois_eff)}..."):
            st.session_state.poi_locs = buscar_pois_localizacoes(
                lat, lon, raio_km * 1000 + 2000, pois_eff
            )
    else:
        st.session_state.poi_locs = {}

    with st.spinner("🔎 Buscando imóveis em ZAP Imóveis, Viva Real e Mercado Livre..."):
        zap_r, viva_r, ml_pub_r, ml_auth_r = _buscar_tudo(
            cidade=cidade, estado=estado,
            bairro=bairro,
            tipos=tipos_eff,
            negocio=negocio_api,
            quartos_min=qp.quartos_hint or quartos_r[0],
            preco_min=float(preco_r[0]),
            preco_max=float(preco_r[1]),
            area_min=float(area_r[0]),
            query_raw=query,
        )

    # Filtra por bairro quando um bairro específico foi identificado
    if bairro:
        zap_r  = _filtrar_bairro(zap_r,  bairro)
        viva_r = _filtrar_bairro(viva_r, bairro)

    st.session_state.zap_items     = zap_r
    st.session_state.viva_items    = viva_r
    st.session_state.ml_pub_items  = ml_pub_r
    st.session_state.ml_auth_items = ml_auth_r
    st.session_state.api_status    = {
        "ZAP Imóveis":    len(zap_r) > 0,
        "Viva Real":      len(viva_r) > 0,
        "ML (público)":   len(ml_pub_r) > 0,
        "ML (auth)":      len(ml_auth_r) > 0,
    }

    total = sum(map(len, [zap_r, viva_r, ml_pub_r, ml_auth_r]))
    if total:
        st.toast(f"✅ {total} imóveis reais encontrados!", icon="🏠")
        todos_items = zap_r + viva_r + ml_pub_r + ml_auth_r
        with st.spinner("🖼️ Carregando fotos dos imóveis..."):
            _prefetch_thumbnails(todos_items, max_imgs=60)
        # Atribui coordenadas para exibição no mapa
        with st.spinner("📍 Geocodificando localização dos imóveis..."):
            todos_geo = _atribuir_coords(todos_items, lat, lon)
        n = len(zap_r)
        st.session_state.zap_items     = todos_geo[:n]
        st.session_state.viva_items    = todos_geo[n:n + len(viva_r)]
        st.session_state.ml_pub_items  = todos_geo[n + len(viva_r):n + len(viva_r) + len(ml_pub_r)]
        st.session_state.ml_auth_items = todos_geo[n + len(viva_r) + len(ml_pub_r):]
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
        O Lastro vai buscar em tempo real no <b>ZAP Imóveis</b>, <b>Viva Real</b>
        e <b>Mercado Livre</b>, e usar IA para avaliar se o preço está
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
    out = []
    for i in items:
        if tipos_sel and i.get("tipo") not in tipos_sel:
            continue
        if not (q_min <= int(i.get("quartos") or 0) <= q_max):
            continue
        if not (b_min <= int(i.get("banheiros") or 0) <= b_max):
            continue
        a = float(i.get("area") or 0)
        if a > 0 and not (area_r[0] <= a <= area_r[1]):
            continue
        p = float(i.get("preco") or 0)
        if p > 0 and not (preco_r[0] <= p <= preco_r[1]):
            continue
        v = int(i.get("vagas_garagem") or 0)
        if vagas_r[0] > 0 and not (vagas_r[0] <= v <= vagas_r[1]):
            continue
        neg = i.get("negocio") or "SALE"
        if negocio_api == "SALE" and neg == "RENTAL":
            continue
        if negocio_api == "RENTAL" and neg == "SALE":
            continue
        out.append(i)
    return out


def _filtrar_poi(items: list[dict]) -> list[dict]:
    pois_eff = list(set(pois_sidebar + (qp.poi_hints if qp else [])))
    if not pois_eff or not st.session_state.poi_locs or not items:
        return items
    from src.search.engine import filtrar_por_poi
    df = pd.DataFrame(items)
    tem = df[df["latitude"].notna() & df["longitude"].notna()]
    sem = df[df["latitude"].isna() | df["longitude"].isna()]
    filtrado = filtrar_por_poi(tem, st.session_state.poi_locs, pois_eff, dist_poi)
    return filtrado.to_dict("records") + sem.to_dict("records")


zap_f      = _prever(_filtrar_poi(_filtrar(st.session_state.zap_items)))
viva_f     = _prever(_filtrar_poi(_filtrar(st.session_state.viva_items)))
ml_pub_f   = _prever(_filtrar_poi(_filtrar(st.session_state.ml_pub_items)))
ml_auth_f  = _prever(_filtrar_poi(_filtrar(st.session_state.ml_auth_items)))
all_items  = zap_f + viva_f + ml_auth_f + ml_pub_f
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
            _render_grid(zap_f[:30])

        if viva_f:
            if zap_f:
                st.divider()
            st.markdown(f"""
            <div class="sec-hdr">
              <span class="sec-hdr-t">Viva Real</span>
              <span class="sec-hdr-c">{len(viva_f)} anúncios</span>
              <span class="sec-badge src-vivareal-badge">vivareal.com.br</span>
            </div>""", unsafe_allow_html=True)
            _render_grid(viva_f[:30])

        ml_all = ml_auth_f + ml_pub_f
        if ml_all:
            if zap_f or viva_f:
                st.divider()
            st.markdown(f"""
            <div class="sec-hdr">
              <span class="sec-hdr-t">Mercado Livre</span>
              <span class="sec-hdr-c">{len(ml_all)} anúncios</span>
              <span class="sec-badge src-ml-badge">mercadolivre.com.br</span>
            </div>""", unsafe_allow_html=True)
            _render_grid(ml_all[:30])

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
