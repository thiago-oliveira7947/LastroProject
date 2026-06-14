"""
Lastro — Análise Gráfica do Mercado Imobiliário
"""
from __future__ import annotations

import unicodedata

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from src import config
from src.models.predict import carregar_modelo


# ── session defaults (caso o usuário acesse esta página diretamente) ───────────
_DEFAULTS = {
    "zap_items":      [],
    "viva_items":     [],
    "geocoded_label": "",
    "buscou":         False,
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ── paletas e constantes ──────────────────────────────────────────────────────
TIPOS_PT = {
    "apartment": "Apartamento",
    "house":     "Casa",
    "commercial":"Comercial",
    "land":      "Terreno",
}
CORES_TIPO = {
    "Apartamento": "#60a5fa",
    "Casa":        "#34d399",
    "Comercial":   "#fbbf24",
    "Terreno":     "#f87171",
}
CORES_AVAL = {
    "Bom negócio": "#34d399",
    "Preço justo": "#fbbf24",
    "Caro":        "#f87171",
}
CORES_FONTE = {
    "ZAP Imóveis": "#fb923c",
    "Viva Real":   "#a78bfa",
}

_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="#0d0d1f",
    plot_bgcolor="#0d0d1f",
    font=dict(family="Inter, system-ui, sans-serif", color="#94a3b8", size=12),
    margin=dict(l=12, r=12, t=32, b=12),
)


# ── modelo ────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def _modelo():
    return carregar_modelo()


def _slug(text: str) -> str:
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c)).lower()


def _preparar_features(item: dict) -> dict:
    lat = float(item.get("latitude") or -23.4628)
    lon = float(item.get("longitude") or -46.5333)
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
        "latitude":      lat,
        "longitude":     lon,
    }


def _prever_batch(items: list[dict]) -> list[int]:
    if not items:
        return []
    modelo = _modelo()
    feats = [_preparar_features(i) for i in items]
    df_feat = pd.DataFrame(feats)[config.FEATURES]
    return modelo.predict(df_feat).round(0).astype(int).tolist()


def _avaliacao(preco: float, previsto: int) -> str:
    ratio = previsto / max(preco, 1)
    if ratio > 1.10:
        return "Bom negócio"
    if ratio < 0.90:
        return "Caro"
    return "Preço justo"


# ── montar dataframe ──────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False, ttl=300)
def _build_df(zap_len: int, viva_len: int) -> pd.DataFrame | None:
    raw = (
        [{"_fonte": "ZAP Imóveis", **i} for i in st.session_state.zap_items] +
        [{"_fonte": "Viva Real",   **i} for i in st.session_state.viva_items]
    )
    if not raw:
        return None
    preds = _prever_batch(raw)
    rows = []
    for item, pred in zip(raw, preds):
        preco = float(item.get("preco") or 0)
        if preco <= 0:
            continue
        area = max(float(item.get("area") or 0), 1.0)
        tipo_pt = TIPOS_PT.get(item.get("tipo", "apartment"), "Imóvel")
        rows.append({
            "Fonte":       item["_fonte"],
            "Tipo":        tipo_pt,
            "Bairro":      str(item.get("bairro") or "").title() or "—",
            "Cidade":      str(item.get("cidade") or "").title(),
            "Quartos":     int(item.get("quartos") or 0),
            "Área (m²)":  area,
            "Preço (R$)": preco,
            "Previsão IA": int(pred),
            "Preço/m²":   round(preco / area, 0),
            "Negócio":    "Venda" if item.get("negocio") == "SALE" else "Aluguel",
            "Avaliação":  _avaliacao(preco, int(pred)),
        })
    return pd.DataFrame(rows) if rows else None


# ─────────────────────────────────────────────────────────────────────────────
# HERO
# ─────────────────────────────────────────────────────────────────────────────
st.page_link("pages/Home.py", label="← Voltar à Busca")
label = st.session_state.geocoded_label or "—"
st.markdown(f"""
<div class="ana-hero">
  <div class="ana-title">📈 Análise do Mercado</div>
  <div class="ana-sub">Região buscada: <b style="color:#60a5fa">{label}</b></div>
</div>""", unsafe_allow_html=True)

# ── sem dados ─────────────────────────────────────────────────────────────────
_tem_dados = st.session_state.buscou and (
    st.session_state.zap_items or st.session_state.viva_items
)
if not _tem_dados:
    st.markdown("""
    <div class="no-data">
      <div class="no-data-icon">📊</div>
      <div class="no-data-t">Nenhuma busca realizada ainda</div>
      <div class="no-data-s">
        Volte à página principal, faça uma busca por imóveis e retorne aqui
        para ver a análise completa do mercado com gráficos interativos.
      </div>
    </div>""", unsafe_allow_html=True)
    st.stop()

# ── carregar dados ────────────────────────────────────────────────────────────
with st.spinner("Calculando previsões e montando gráficos..."):
    df = _build_df(len(st.session_state.zap_items), len(st.session_state.viva_items))

if df is None or df.empty:
    st.warning("Não há imóveis com preço válido para analisar.")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# KPIs
# ─────────────────────────────────────────────────────────────────────────────
total         = len(df)
preco_medio   = df["Preço (R$)"].mean()
previsao_med  = df["Previsão IA"].mean()
pct_bom       = (df["Avaliação"] == "Bom negócio").mean() * 100
pm2_medio     = df["Preço/m²"].mean()

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("🏘️ Imóveis",            f"{total:,}")
k2.metric("💰 Preço médio",         f"R$ {preco_medio:,.0f}")
k3.metric("🤖 Previsão IA média",  f"R$ {previsao_med:,.0f}")
k4.metric("📐 Preço/m² médio",     f"R$ {pm2_medio:,.0f}")
k5.metric("✅ Bons negócios",       f"{pct_bom:.0f}%")

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# SEÇÃO 1 — Distribuição de preços + Avaliação IA
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<span class="sec-label">Visão geral</span>', unsafe_allow_html=True)
c1, c2 = st.columns([3, 2], gap="medium")

with c1:
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.markdown('<p class="chart-title">Distribuição de Preços</p>', unsafe_allow_html=True)
    st.markdown('<p class="chart-sub">Concentração dos valores anunciados por tipo de imóvel</p>', unsafe_allow_html=True)

    fig_hist = px.histogram(
        df, x="Preço (R$)", color="Tipo",
        nbins=40, barmode="overlay", opacity=0.75,
        color_discrete_map=CORES_TIPO,
    )
    fig_hist.update_traces(marker_line_width=0)
    fig_hist.update_layout(
        **_LAYOUT,
        height=340,
        xaxis_title="Preço (R$)", yaxis_title="Quantidade",
        legend=dict(orientation="h", y=1.1, x=0, font_size=11),
        bargap=0.04,
        xaxis=dict(gridcolor="rgba(255,255,255,.04)"),
        yaxis=dict(gridcolor="rgba(255,255,255,.04)"),
    )
    st.plotly_chart(fig_hist, use_container_width=True, config={"displayModeBar": False})
    st.markdown('</div>', unsafe_allow_html=True)

with c2:
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.markdown('<p class="chart-title">Avaliação da IA</p>', unsafe_allow_html=True)
    st.markdown('<p class="chart-sub">Classificação dos anúncios pelo modelo de previsão de preço</p>', unsafe_allow_html=True)

    aval_counts = df["Avaliação"].value_counts()
    fig_donut = go.Figure(go.Pie(
        labels=aval_counts.index,
        values=aval_counts.values,
        hole=0.62,
        marker_colors=[CORES_AVAL.get(l, "#94a3b8") for l in aval_counts.index],
        textinfo="percent+label",
        textfont=dict(size=12, family="Inter, sans-serif"),
        hovertemplate="%{label}: %{value} imóveis (%{percent})<extra></extra>",
    ))
    fig_donut.update_layout(
        **_LAYOUT,
        height=340,
        showlegend=False,
        annotations=[dict(
            text=f"<b style='font-size:18px'>{total}</b><br>imóveis",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color="#e2e8f0", family="Inter, sans-serif"),
        )],
    )
    st.plotly_chart(fig_donut, use_container_width=True, config={"displayModeBar": False})
    st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SEÇÃO 2 — Área × Preço
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<span class="sec-label">Relação área e preço</span>', unsafe_allow_html=True)
st.markdown('<div class="chart-card">', unsafe_allow_html=True)
st.markdown('<p class="chart-title">Área × Preço por Tipo</p>', unsafe_allow_html=True)
st.markdown('<p class="chart-sub">Cada ponto representa um anúncio — passe o mouse para ver os detalhes</p>', unsafe_allow_html=True)

fig_scatter = px.scatter(
    df, x="Área (m²)", y="Preço (R$)", color="Tipo",
    symbol="Fonte", opacity=0.72,
    color_discrete_map=CORES_TIPO,
    hover_data={"Bairro": True, "Quartos": True, "Avaliação": True, "Fonte": True},
)
fig_scatter.update_traces(marker=dict(size=7, line=dict(width=0)))
fig_scatter.update_layout(
    **_LAYOUT,
    height=430,
    xaxis_title="Área (m²)", yaxis_title="Preço (R$)",
    legend=dict(orientation="h", y=1.07, x=0, font_size=11),
    xaxis=dict(gridcolor="rgba(255,255,255,.04)"),
    yaxis=dict(gridcolor="rgba(255,255,255,.04)"),
)
st.plotly_chart(fig_scatter, use_container_width=True, config={"displayModeBar": False})
st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SEÇÃO 3 — Preço/m² + Top bairros
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<span class="sec-label">Mercado por tipo e localização</span>', unsafe_allow_html=True)
c3, c4 = st.columns([2, 3], gap="medium")

with c3:
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.markdown('<p class="chart-title">Preço/m² por Tipo de Imóvel</p>', unsafe_allow_html=True)
    st.markdown('<p class="chart-sub">Distribuição do valor por metro quadrado</p>', unsafe_allow_html=True)

    fig_box = px.box(
        df, x="Tipo", y="Preço/m²", color="Tipo",
        color_discrete_map=CORES_TIPO,
        points="outliers",
    )
    fig_box.update_traces(marker=dict(size=4, opacity=0.45))
    fig_box.update_layout(
        **_LAYOUT,
        height=380, showlegend=False,
        xaxis_title="", yaxis_title="R$/m²",
        yaxis=dict(gridcolor="rgba(255,255,255,.04)"),
    )
    st.plotly_chart(fig_box, use_container_width=True, config={"displayModeBar": False})
    st.markdown('</div>', unsafe_allow_html=True)

with c4:
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.markdown('<p class="chart-title">Top 15 Bairros</p>', unsafe_allow_html=True)
    st.markdown('<p class="chart-sub">Bairros com maior oferta de imóveis disponíveis</p>', unsafe_allow_html=True)

    top_bairros = (
        df[df["Bairro"] != "—"]
        .groupby("Bairro")
        .agg(Quantidade=("Preço (R$)", "count"), Preco_medio=("Preço (R$)", "mean"))
        .sort_values("Quantidade", ascending=True)
        .tail(15)
    )
    fig_bar = go.Figure(go.Bar(
        y=top_bairros.index,
        x=top_bairros["Quantidade"],
        orientation="h",
        marker_color="#3b82f6",
        marker_line_width=0,
        text=top_bairros["Quantidade"],
        textposition="outside",
        textfont=dict(size=11, color="#94a3b8"),
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Imóveis: %{x}<br>"
            "Preço médio: R$ %{customdata:,.0f}<extra></extra>"
        ),
        customdata=top_bairros["Preco_medio"].values,
    ))
    fig_bar.update_layout(
        **_LAYOUT,
        height=380,
        xaxis_title="Imóveis",
        yaxis=dict(tickfont=dict(size=11)),
        xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,.04)"),
    )
    st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})
    st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SEÇÃO 4 — Preço real × Previsão IA
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<span class="sec-label">Previsão de preço pela IA</span>', unsafe_allow_html=True)
st.markdown('<div class="chart-card">', unsafe_allow_html=True)
st.markdown('<p class="chart-title">Preço Anunciado × Previsão da IA</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="chart-sub">'
    'Pontos <b style="color:#34d399">acima</b> da linha = previsão maior que o anúncio (possível oportunidade) · '
    'Pontos <b style="color:#f87171">abaixo</b> = caro em relação à previsão'
    '</p>',
    unsafe_allow_html=True,
)

max_val = max(float(df["Preço (R$)"].max()), float(df["Previsão IA"].max())) * 1.05
fig_ia = px.scatter(
    df, x="Preço (R$)", y="Previsão IA", color="Avaliação",
    color_discrete_map=CORES_AVAL, opacity=0.78,
    hover_data={"Tipo": True, "Bairro": True, "Área (m²)": True, "Fonte": True},
)
fig_ia.add_trace(go.Scatter(
    x=[0, max_val], y=[0, max_val],
    mode="lines",
    line=dict(color="rgba(148,163,184,.3)", dash="dash", width=1.5),
    name="Preço = Previsão",
    hoverinfo="skip",
))
fig_ia.update_traces(marker=dict(size=7, line=dict(width=0)), selector=dict(mode="markers"))
fig_ia.update_layout(
    **_LAYOUT,
    height=460,
    xaxis_title="Preço anunciado (R$)", yaxis_title="Previsão IA (R$)",
    legend=dict(orientation="h", y=1.07, x=0, font_size=11),
    xaxis=dict(gridcolor="rgba(255,255,255,.04)", range=[0, max_val]),
    yaxis=dict(gridcolor="rgba(255,255,255,.04)", range=[0, max_val]),
)
st.plotly_chart(fig_ia, use_container_width=True, config={"displayModeBar": False})
st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SEÇÃO 5 — Comparativos
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<span class="sec-label">Comparativos</span>', unsafe_allow_html=True)
c5, c6 = st.columns(2, gap="medium")

with c5:
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.markdown('<p class="chart-title">Imóveis por Fonte</p>', unsafe_allow_html=True)
    st.markdown('<p class="chart-sub">Comparativo entre ZAP Imóveis e Viva Real</p>', unsafe_allow_html=True)

    fonte_stats = (
        df.groupby("Fonte")
        .agg(Quantidade=("Preço (R$)", "count"), Preco_medio=("Preço (R$)", "mean"))
        .reset_index()
    )
    fig_fonte = px.bar(
        fonte_stats, x="Fonte", y="Quantidade", color="Fonte",
        color_discrete_map=CORES_FONTE,
        text="Quantidade",
        custom_data=["Preco_medio"],
    )
    fig_fonte.update_traces(
        marker_line_width=0, textposition="outside",
        hovertemplate="<b>%{x}</b><br>Imóveis: %{y}<br>Preço médio: R$ %{customdata[0]:,.0f}<extra></extra>",
    )
    fig_fonte.update_layout(
        **_LAYOUT,
        height=310, showlegend=False,
        xaxis_title="", yaxis_title="Imóveis",
        yaxis=dict(gridcolor="rgba(255,255,255,.04)"),
    )
    st.plotly_chart(fig_fonte, use_container_width=True, config={"displayModeBar": False})
    st.markdown('</div>', unsafe_allow_html=True)

with c6:
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.markdown('<p class="chart-title">Distribuição por Número de Quartos</p>', unsafe_allow_html=True)
    st.markdown('<p class="chart-sub">Quantidade de imóveis disponíveis por dormitório</p>', unsafe_allow_html=True)

    q_df = df[df["Quartos"] > 0]["Quartos"].value_counts().sort_index().reset_index()
    q_df.columns = ["Quartos", "Quantidade"]
    q_df["label"] = q_df["Quartos"].astype(str) + " quarto(s)"
    fig_q = px.bar(q_df, x="label", y="Quantidade", text="Quantidade",
                   color_discrete_sequence=["#3b82f6"])
    fig_q.update_traces(marker_line_width=0, textposition="outside", marker_color="#3b82f6")
    fig_q.update_layout(
        **_LAYOUT,
        height=310, showlegend=False,
        xaxis_title="", yaxis_title="Imóveis",
        yaxis=dict(gridcolor="rgba(255,255,255,.04)"),
    )
    st.plotly_chart(fig_q, use_container_width=True, config={"displayModeBar": False})
    st.markdown('</div>', unsafe_allow_html=True)
