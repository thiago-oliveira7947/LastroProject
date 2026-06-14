"""
Lastro — Shell de navegação.
Subir: streamlit run app.py
"""
from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()

import streamlit as st

st.set_page_config(
    page_title="Lastro — Imóveis",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Navegação entre páginas ────────────────────────────────────────────────────
pg = st.navigation(
    [
        st.Page("pages/Home.py",    title="Buscar Imóveis",  icon="🏠", default=True),
        st.Page("pages/Analise.py", title="Análise Gráfica", icon="📈"),
    ],
    position="sidebar",
)

# ── CSS compartilhado (aplicado em todas as páginas antes de pg.run()) ─────────
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

/* ══ ANÁLISE CTA ══ */
.ana-cta{display:flex;align-items:center;gap:16px;
  background:linear-gradient(135deg,rgba(59,130,246,.1),rgba(99,102,241,.08));
  border:1px solid rgba(59,130,246,.25);border-radius:16px;
  padding:16px 20px;margin:16px 0 4px}
.ana-cta-icon{font-size:32px;flex-shrink:0}
.ana-cta-t{font-size:15px;font-weight:700;color:#e2e8f0;margin:0 0 3px}
.ana-cta-s{font-size:12px;color:#475569;margin:0}
.ana-cta-body{flex:1;min-width:0}

/* ══ ANÁLISE PAGE ══ */
.ana-hero{text-align:center;padding:28px 0 18px}
.ana-title{font-size:36px;font-weight:800;
  background:linear-gradient(135deg,#f1f5f9 0%,#60a5fa 45%,#a78bfa 100%);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;
  line-height:1.15;margin:0 0 8px}
.ana-sub{font-size:14px;color:#475569;margin:0}
.chart-card{background:#0d0d1f;border:1px solid rgba(255,255,255,.07);
  border-radius:16px;padding:20px 20px 12px;margin-bottom:0}
.chart-title{font-size:15px;font-weight:700;color:#e2e8f0;margin:0 0 3px}
.chart-sub{font-size:12px;color:#475569;margin:0 0 10px}
.no-data{text-align:center;padding:100px 20px}
.no-data-icon{font-size:72px;margin-bottom:16px}
.no-data-t{font-size:24px;font-weight:700;color:#475569;margin-bottom:10px}
.no-data-s{font-size:15px;color:#334155;max-width:480px;margin:0 auto;line-height:1.8}
.sec-label{font-size:11px;font-weight:700;letter-spacing:1.4px;
  text-transform:uppercase;color:#334155;margin:28px 0 14px;display:block}
</style>
""", unsafe_allow_html=True)

# ── Executa a página selecionada ───────────────────────────────────────────────
pg.run()
