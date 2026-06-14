"""Testes para src/external/quintoandar.py."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from src.external.quintoandar import (
    _slug,
    _url_qa,
    _fetch_html,
    _parece_listing,
    _buscar_listas_recursivo,
    _extrair_next_data,
    _extrair_listings_next,
    _extrair_json_scripts,
    _extrair_jsonld,
    _normalizar,
    buscar,
)


# ─── Fixtures / helpers ───────────────────────────────────────────────────────

VALID_LISTING = {
    "id": "qa_001",
    "bedrooms": 2,
    "area": 65.0,
    "rent": 2500.0,
    "neighborhood": "Pinheiros",
    "coverImageUrl": "https://cdn.quintoandar.com.br/img.jpg",
}

SALE_LISTING = {
    "id": "qa_002",
    "bedrooms": 3,
    "area": 90.0,
    "salePrice": 650000.0,
    "neighborhood": "Vila Madalena",
    "coverImageUrl": "https://cdn.quintoandar.com.br/img2.jpg",
    "businessType": "SALE",
}


def _make_next_data_html(pageprops_key: str, listings: list[dict]) -> str:
    """HTML com __NEXT_DATA__ contendo listings em pageProps[pageprops_key]."""
    data = {
        "props": {
            "pageProps": {
                pageprops_key: listings,
            }
        },
        "page": "/comprar",
        "query": {},
    }
    payload = json.dumps(data)
    return f'<html><script id="__NEXT_DATA__" type="application/json">{payload}</script></html>'


def _make_json_script_html(listings: list[dict]) -> str:
    """HTML com script tag contendo JSON com listings."""
    wrapper = {"results": listings, "total": len(listings)}
    payload = json.dumps(wrapper)
    return f'<html><script type="application/json">{payload}</script></html>'


def _make_jsonld_html(listings: list[dict]) -> str:
    """HTML com JSON-LD ItemList contendo listings."""
    elements = [{"@type": "ListItem", "item": listing} for listing in listings]
    ld = {"@type": "ItemList", "itemListElement": elements}
    payload = json.dumps(ld)
    return f'<html><script type="application/ld+json">{payload}</script></html>'


# ─── _slug ────────────────────────────────────────────────────────────────────

class TestSlug:
    def test_remove_accents(self):
        assert _slug("São Paulo") == "sao-paulo"

    def test_cedilla(self):
        assert _slug("Conceição") == "conceicao"

    def test_lowercase(self):
        assert _slug("GUARULHOS") == "guarulhos"

    def test_spaces_to_hyphens(self):
        assert _slug("Vila Nova Conceicao") == "vila-nova-conceicao"

    def test_multiple_spaces(self):
        assert _slug("bela  vista") == "bela-vista"

    def test_numbers_preserved(self):
        assert _slug("zona3") == "zona3"

    def test_empty_string(self):
        assert _slug("") == ""

    def test_already_ascii(self):
        assert _slug("guarulhos") == "guarulhos"

    def test_special_chars_removed(self):
        # Hífens e pontuação são removidos pelo regex [^a-zA-Z0-9 ]
        assert _slug("Itaim-Bibi!") == "itaimbibi"

    def test_strips_leading_trailing(self):
        assert _slug("  centro  ") == "centro"


# ─── _url_qa ─────────────────────────────────────────────────────────────────

class TestUrlQa:
    def test_sale_uses_comprar(self):
        url = _url_qa("sao-paulo", "", "SALE")
        assert "comprar" in url

    def test_rental_uses_alugar(self):
        url = _url_qa("sao-paulo", "", "RENTAL")
        assert "alugar" in url

    def test_cidade_slug_in_url(self):
        url = _url_qa("sao-paulo", "", "SALE")
        assert "sao-paulo" in url

    def test_bairro_slug_included_when_provided(self):
        url = _url_qa("sao-paulo", "pinheiros", "SALE")
        assert "pinheiros" in url
        assert "sao-paulo" in url

    def test_no_bairro_url_format(self):
        url = _url_qa("sao-paulo", "", "SALE")
        assert url == "https://www.quintoandar.com.br/comprar/imovel/sao-paulo/"

    def test_with_bairro_url_format(self):
        url = _url_qa("sao-paulo", "vila-madalena", "RENTAL")
        assert url == "https://www.quintoandar.com.br/alugar/imovel/sao-paulo/vila-madalena/"

    def test_url_starts_with_https(self):
        url = _url_qa("guarulhos", "", "SALE")
        assert url.startswith("https://")

    def test_quintoandar_domain(self):
        url = _url_qa("guarulhos", "", "SALE")
        assert "quintoandar.com.br" in url


# ─── _fetch_html ──────────────────────────────────────────────────────────────

class TestFetchHtml:
    def test_returns_html_on_200(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.content = b"<html>ola</html>"
        with patch("src.external.quintoandar.requests.get", return_value=mock_resp):
            result = _fetch_html("https://example.com")
        assert result == "<html>ola</html>"

    def test_returns_none_on_403(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 403
        with patch("src.external.quintoandar.requests.get", return_value=mock_resp):
            result = _fetch_html("https://example.com")
        assert result is None

    def test_returns_none_on_404(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        with patch("src.external.quintoandar.requests.get", return_value=mock_resp):
            result = _fetch_html("https://example.com")
        assert result is None

    def test_returns_none_on_connection_error(self):
        with patch("src.external.quintoandar.requests.get", side_effect=ConnectionError()):
            result = _fetch_html("https://example.com")
        assert result is None

    def test_returns_none_on_timeout(self):
        import requests as req
        with patch("src.external.quintoandar.requests.get", side_effect=req.Timeout()):
            result = _fetch_html("https://example.com")
        assert result is None

    def test_uses_user_agent_header(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.content = b"ok"
        with patch("src.external.quintoandar.requests.get", return_value=mock_resp) as mock_get:
            _fetch_html("https://example.com")
        headers = mock_get.call_args[1]["headers"]
        assert "User-Agent" in headers


# ─── _parece_listing ─────────────────────────────────────────────────────────

class TestParecelisting:
    def test_dois_listing_keys_retorna_true(self):
        assert _parece_listing({"id": "1", "bedrooms": 2}) is True

    def test_rent_e_area_retorna_true(self):
        assert _parece_listing({"rent": 1500.0, "area": 60.0}) is True

    def test_listing_completo_retorna_true(self):
        assert _parece_listing(VALID_LISTING) is True

    def test_uma_key_apenas_retorna_false(self):
        assert _parece_listing({"bedrooms": 2}) is False

    def test_sem_listing_keys_retorna_false(self):
        assert _parece_listing({"nome": "teste", "tipo": "a"}) is False

    def test_dict_vazio_retorna_false(self):
        assert _parece_listing({}) is False

    def test_nao_dict_retorna_false(self):
        assert _parece_listing([1, 2]) is False

    def test_string_retorna_false(self):
        assert _parece_listing("listing") is False

    def test_none_retorna_false(self):
        assert _parece_listing(None) is False

    def test_sale_price_e_neighborhood(self):
        assert _parece_listing({"salePrice": 500000, "neighborhood": "Centro"}) is True

    def test_listing_id_e_total_cost(self):
        assert _parece_listing({"listingId": "abc", "totalCost": 2200}) is True

    def test_cover_image_e_price(self):
        assert _parece_listing({"coverImageUrl": "http://x.com/img.jpg", "price": 3000}) is True


# ─── _buscar_listas_recursivo ─────────────────────────────────────────────────

class TestBuscarListasRecursivo:
    def test_lista_de_listings_retornada(self):
        listings = [VALID_LISTING, VALID_LISTING]
        result = _buscar_listas_recursivo(listings)
        assert len(result) == 1
        assert result[0] is listings

    def test_lista_vazia_retorna_lista_vazia(self):
        assert _buscar_listas_recursivo([]) == []

    def test_lista_de_nao_listings_recursivo(self):
        nested = {"homes": [VALID_LISTING, VALID_LISTING]}
        result = _buscar_listas_recursivo(nested)
        assert len(result) == 1
        assert result[0] == [VALID_LISTING, VALID_LISTING]

    def test_profundidade_maxima_retorna_vazio(self):
        # depth 9 → além do limite de 8
        assert _buscar_listas_recursivo([VALID_LISTING], depth=9) == []

    def test_depth_8_ainda_processa(self):
        # depth 8 → no limite, ainda deve processar
        result = _buscar_listas_recursivo([VALID_LISTING], depth=8)
        assert len(result) == 1

    def test_dict_com_lista_aninhada(self):
        obj = {"a": {"b": {"c": [VALID_LISTING]}}}
        result = _buscar_listas_recursivo(obj)
        assert len(result) == 1

    def test_nenhum_listing_retorna_vazio(self):
        obj = {"a": [{"x": 1}, {"y": 2}], "b": "texto"}
        result = _buscar_listas_recursivo(obj)
        assert result == []

    def test_objeto_primitivo_retorna_vazio(self):
        assert _buscar_listas_recursivo(42) == []
        assert _buscar_listas_recursivo("texto") == []
        assert _buscar_listas_recursivo(None) == []

    def test_lista_com_primeiro_invalido_recusa(self):
        # Primeiro item não é listing, demais sim — lista rejeitada
        mixed = [{"x": 1}, VALID_LISTING]
        result = _buscar_listas_recursivo(mixed)
        # Só aceita se o primeiro item é listing
        assert result == []

    def test_multiplos_niveis_retorna_todas_as_listas(self):
        obj = {
            "group_a": [VALID_LISTING, VALID_LISTING],
            "group_b": [SALE_LISTING],
        }
        result = _buscar_listas_recursivo(obj)
        assert len(result) == 2


# ─── _extrair_next_data ───────────────────────────────────────────────────────

class TestExtrairNextData:
    def test_retorna_dict_com_props(self):
        html = _make_next_data_html("homes", [VALID_LISTING])
        result = _extrair_next_data(html)
        assert "props" in result

    def test_retorna_vazio_sem_next_data(self):
        html = "<html><body>sem script</body></html>"
        assert _extrair_next_data(html) == {}

    def test_retorna_vazio_com_json_invalido(self):
        html = '<html><script id="__NEXT_DATA__" type="application/json">INVALID{}</script></html>'
        assert _extrair_next_data(html) == {}

    def test_conteudo_corretamente_parseado(self):
        data = {"props": {"pageProps": {"total": 5}}, "page": "/test"}
        payload = json.dumps(data)
        html = f'<script id="__NEXT_DATA__" type="application/json">{payload}</script>'
        result = _extrair_next_data(html)
        assert result["page"] == "/test"
        assert result["props"]["pageProps"]["total"] == 5

    def test_id_com_aspas_simples(self):
        data = {"props": {}}
        payload = json.dumps(data)
        html = f"<script id='__NEXT_DATA__' type='application/json'>{payload}</script>"
        result = _extrair_next_data(html)
        assert result == {"props": {}}

    def test_script_sem_id_nao_capturado(self):
        data = {"props": {}}
        payload = json.dumps(data)
        html = f'<script type="application/json">{payload}</script>'
        assert _extrair_next_data(html) == {}


# ─── _extrair_listings_next ───────────────────────────────────────────────────

class TestExtrairListingsNext:
    def test_extrai_de_homes(self):
        html = _make_next_data_html("homes", [VALID_LISTING, VALID_LISTING])
        data = _extrair_next_data(html)
        result = _extrair_listings_next(data)
        assert len(result) == 2

    def test_extrai_de_listings(self):
        html = _make_next_data_html("listings", [VALID_LISTING])
        data = _extrair_next_data(html)
        result = _extrair_listings_next(data)
        assert len(result) == 1

    def test_extrai_de_results(self):
        html = _make_next_data_html("results", [SALE_LISTING, VALID_LISTING])
        data = _extrair_next_data(html)
        result = _extrair_listings_next(data)
        assert len(result) == 2

    def test_extrai_de_items(self):
        html = _make_next_data_html("items", [VALID_LISTING])
        data = _extrair_next_data(html)
        result = _extrair_listings_next(data)
        assert len(result) == 1

    def test_extrai_de_properties(self):
        html = _make_next_data_html("properties", [SALE_LISTING])
        data = _extrair_next_data(html)
        result = _extrair_listings_next(data)
        assert len(result) == 1

    def test_retorna_vazio_com_data_vazio(self):
        assert _extrair_listings_next({}) == []

    def test_retorna_vazio_sem_listings(self):
        data = {"props": {"pageProps": {"title": "Sem imóveis"}}}
        assert _extrair_listings_next(data) == []

    def test_busca_recursiva_em_pageprops(self):
        # Listings aninhados em sub-objeto
        data = {
            "props": {
                "pageProps": {
                    "search": {
                        "section": {
                            "data": [VALID_LISTING, SALE_LISTING]
                        }
                    }
                }
            }
        }
        result = _extrair_listings_next(data)
        assert len(result) == 2

    def test_retorna_maior_lista_quando_multiplas(self):
        data = {
            "props": {
                "pageProps": {
                    "featured": [VALID_LISTING],
                    "all": [VALID_LISTING, SALE_LISTING, VALID_LISTING],
                }
            }
        }
        result = _extrair_listings_next(data)
        assert len(result) == 3  # Retorna a maior lista


# ─── _extrair_json_scripts ────────────────────────────────────────────────────

class TestExtrairJsonScripts:
    def test_extrai_listings_de_script_json(self):
        html = _make_json_script_html([VALID_LISTING, SALE_LISTING])
        result = _extrair_json_scripts(html)
        assert len(result) == 2

    def test_retorna_vazio_sem_scripts(self):
        html = "<html><body>sem script</body></html>"
        assert _extrair_json_scripts(html) == []

    def test_ignora_json_sem_listings(self):
        payload = json.dumps({"title": "Página", "count": 0})
        html = f'<html><script type="application/json">{payload}</script></html>'
        assert _extrair_json_scripts(html) == []

    def test_ignora_script_sem_chave_listagem(self):
        payload = json.dumps({"user": {"name": "test"}, "token": "abc"})
        html = f'<html><script>{payload}</script></html>'
        assert _extrair_json_scripts(html) == []

    def test_ignora_json_invalido(self):
        html = "<html><script>{isso nao e json}</script></html>"
        assert _extrair_json_scripts(html) == []

    def test_script_que_nao_comeca_com_brace_ignorado(self):
        html = '<html><script>var x = {"homes": []};</script></html>'
        # Começa com "var", não com "{" ou "[" → ignorado
        assert _extrair_json_scripts(html) == []

    def test_array_na_raiz_com_listings(self):
        payload = json.dumps([VALID_LISTING, SALE_LISTING])
        html = f'<html><script type="application/json">{payload}</script></html>'
        result = _extrair_json_scripts(html)
        assert len(result) == 2

    def test_listings_aninhados_em_json_complexo(self):
        wrapper = {
            "state": {
                "cache": {
                    "listings": [VALID_LISTING, SALE_LISTING]
                }
            }
        }
        payload = json.dumps(wrapper)
        html = f'<html><script type="application/json">{payload}</script></html>'
        result = _extrair_json_scripts(html)
        assert len(result) == 2


# ─── _extrair_jsonld ─────────────────────────────────────────────────────────

class TestExtrairJsonld:
    def test_extrai_de_item_list(self):
        html = _make_jsonld_html([VALID_LISTING, SALE_LISTING])
        result = _extrair_jsonld(html)
        assert len(result) == 2

    def test_retorna_vazio_para_tipo_errado(self):
        ld = {"@type": "Product", "itemListElement": [{"item": VALID_LISTING}]}
        html = f'<script type="application/ld+json">{json.dumps(ld)}</script>'
        assert _extrair_jsonld(html) == []

    def test_retorna_vazio_sem_script_ld_json(self):
        html = "<html><body>sem ld+json</body></html>"
        assert _extrair_jsonld(html) == []

    def test_ignora_json_invalido(self):
        html = '<script type="application/ld+json">INVALID</script>'
        assert _extrair_jsonld(html) == []

    def test_usa_item_dentro_de_elemento(self):
        ld = {
            "@type": "ItemList",
            "itemListElement": [{"@type": "ListItem", "item": VALID_LISTING}]
        }
        html = f'<script type="application/ld+json">{json.dumps(ld)}</script>'
        result = _extrair_jsonld(html)
        assert len(result) == 1
        assert result[0] == VALID_LISTING

    def test_usa_elemento_sem_item_key(self):
        ld = {
            "@type": "ItemList",
            "itemListElement": [VALID_LISTING]  # sem wrapper "item"
        }
        html = f'<script type="application/ld+json">{json.dumps(ld)}</script>'
        result = _extrair_jsonld(html)
        assert len(result) == 1

    def test_lista_vazia_no_elemento(self):
        ld = {"@type": "ItemList", "itemListElement": []}
        html = f'<script type="application/ld+json">{json.dumps(ld)}</script>'
        assert _extrair_jsonld(html) == []

    def test_multiplos_scripts_ld_json(self):
        # Dois scripts: um BreadcrumbList e um ItemList
        breadcrumb = json.dumps({"@type": "BreadcrumbList", "itemListElement": []})
        item_list = json.dumps({
            "@type": "ItemList",
            "itemListElement": [{"item": VALID_LISTING}]
        })
        html = (
            f'<script type="application/ld+json">{breadcrumb}</script>'
            f'<script type="application/ld+json">{item_list}</script>'
        )
        result = _extrair_jsonld(html)
        assert len(result) == 1


# ─── _normalizar ─────────────────────────────────────────────────────────────

class TestNormalizar:
    def test_normaliza_listing_completo(self):
        raw = {
            "id": "qa_001",
            "bedrooms": 3,
            "area": 85.0,
            "salePrice": 750000.0,
            "neighborhood": "Pinheiros",
            "coverImageUrl": "https://cdn.example.com/img.jpg",
            "businessType": "SALE",
        }
        result = _normalizar(raw, "São Paulo", "São Paulo")
        assert result is not None
        assert result["id"] == "qa_001"
        assert result["quartos"] == 3
        assert result["area"] == 85.0
        assert result["preco"] == 750000.0
        assert result["bairro"] == "Pinheiros"
        assert result["thumbnail"] == "https://cdn.example.com/img.jpg"
        assert result["negocio"] == "SALE"
        assert result["fonte"] == "quintoandar"

    def test_retorna_none_para_dict_vazio(self):
        assert _normalizar({}, "São Paulo", "SP") is None

    def test_retorna_none_para_none(self):
        assert _normalizar(None, "São Paulo", "SP") is None  # type: ignore

    def test_area_minima_18(self):
        raw = {**VALID_LISTING, "area": 5.0}
        result = _normalizar(raw, "São Paulo", "SP")
        assert result["area"] == 18.0

    def test_area_zero_defaults_18(self):
        raw = {**VALID_LISTING, "area": 0}
        result = _normalizar(raw, "São Paulo", "SP")
        assert result["area"] == 18.0

    def test_area_ausente_defaults_18(self):
        raw = {k: v for k, v in VALID_LISTING.items() if k != "area"}
        result = _normalizar(raw, "São Paulo", "SP")
        assert result["area"] == 18.0

    def test_area_valida_preservada(self):
        raw = {**VALID_LISTING, "area": 120.0}
        result = _normalizar(raw, "São Paulo", "SP")
        assert result["area"] == 120.0

    def test_preco_zero_quando_ausente(self):
        raw = {"id": "x", "bedrooms": 2, "area": 50.0, "neighborhood": "Centro"}
        result = _normalizar(raw, "Guarulhos", "SP")
        assert result["preco"] == 0.0

    def test_negocio_rental_por_businesstype(self):
        raw = {**VALID_LISTING, "businessType": "RENTAL", "salePrice": None, "rent": 2000}
        result = _normalizar(raw, "SP", "SP")
        assert result["negocio"] == "RENTAL"

    def test_negocio_sale_por_businesstype(self):
        raw = {**SALE_LISTING}
        result = _normalizar(raw, "SP", "SP")
        assert result["negocio"] == "SALE"

    def test_negocio_rental_inferido_por_preco_baixo(self):
        raw = {**VALID_LISTING, "rent": 3000}  # 3000 < 20000 → RENTAL
        raw.pop("businessType", None)
        result = _normalizar(raw, "SP", "SP")
        assert result["negocio"] == "RENTAL"

    def test_negocio_sale_inferido_por_preco_alto(self):
        raw = {"id": "x", "bedrooms": 2, "area": 60.0, "salePrice": 500000, "neighborhood": "X"}
        result = _normalizar(raw, "SP", "SP")
        assert result["negocio"] == "SALE"

    def test_cidade_default_para_cidade_ref(self):
        raw = {**VALID_LISTING}
        result = _normalizar(raw, "Guarulhos", "São Paulo")
        assert result["cidade"] == "guarulhos"

    def test_estado_default_para_estado_ref(self):
        raw = {**VALID_LISTING}
        result = _normalizar(raw, "Guarulhos", "São Paulo")
        assert result["estado"] == "São Paulo"

    def test_latitude_longitude_quando_presentes(self):
        raw = {**VALID_LISTING, "lat": -23.5505, "lng": -46.6333}
        result = _normalizar(raw, "SP", "SP")
        assert result["latitude"] == pytest.approx(-23.5505)
        assert result["longitude"] == pytest.approx(-46.6333)

    def test_latitude_longitude_none_quando_ausentes(self):
        result = _normalizar(VALID_LISTING, "SP", "SP")
        assert result["latitude"] is None
        assert result["longitude"] is None

    def test_thumbnail_de_image_list(self):
        raw = {**VALID_LISTING}
        raw.pop("coverImageUrl", None)
        raw["image"] = ["https://example.com/pic.jpg"]
        result = _normalizar(raw, "SP", "SP")
        assert result["thumbnail"] == "https://example.com/pic.jpg"

    def test_thumbnail_vazio_sem_imagem(self):
        raw = {k: v for k, v in VALID_LISTING.items() if k != "coverImageUrl"}
        result = _normalizar(raw, "SP", "SP")
        assert result["thumbnail"] == ""

    def test_listingid_como_id_alternativo(self):
        raw = {**VALID_LISTING}
        raw.pop("id", None)
        raw["listingId"] = "lista_999"
        result = _normalizar(raw, "SP", "SP")
        assert result["id"] == "lista_999"

    def test_rent_como_preco(self):
        raw = {**VALID_LISTING}  # VALID_LISTING tem rent=2500
        result = _normalizar(raw, "SP", "SP")
        assert result["preco"] == 2500.0

    def test_condominio_extraido(self):
        raw = {**VALID_LISTING, "condoFee": 450.0}
        result = _normalizar(raw, "SP", "SP")
        assert result["condominio"] == 450.0

    def test_condominio_zero_quando_ausente(self):
        result = _normalizar(VALID_LISTING, "SP", "SP")
        assert result["condominio"] == 0.0

    def test_vagas_de_parkingspots(self):
        raw = {**VALID_LISTING, "parkingSpots": 2}
        result = _normalizar(raw, "SP", "SP")
        assert result["vagas_garagem"] == 2

    def test_vagas_zero_quando_ausente(self):
        result = _normalizar(VALID_LISTING, "SP", "SP")
        assert result["vagas_garagem"] == 0

    def test_banheiros_extraidos(self):
        raw = {**VALID_LISTING, "bathrooms": 3}
        result = _normalizar(raw, "SP", "SP")
        assert result["banheiros"] == 3

    def test_descricao_truncada_em_500(self):
        raw = {**VALID_LISTING, "description": "x" * 700}
        result = _normalizar(raw, "SP", "SP")
        assert len(result["descricao"]) <= 500

    def test_fonte_sempre_quintoandar(self):
        result = _normalizar(VALID_LISTING, "SP", "SP")
        assert result["fonte"] == "quintoandar"

    def test_suites_sempre_zero(self):
        result = _normalizar(VALID_LISTING, "SP", "SP")
        assert result["suites"] == 0

    def test_iptu_sempre_zero(self):
        result = _normalizar(VALID_LISTING, "SP", "SP")
        assert result["iptu"] == 0.0

    def test_link_relativo_converte_para_absoluto(self):
        raw = {**VALID_LISTING, "href": "/imovel/qa_001/"}
        result = _normalizar(raw, "SP", "SP")
        assert result["link"].startswith("https://www.quintoandar.com.br")

    def test_link_absoluto_preservado(self):
        raw = {**VALID_LISTING, "href": "https://www.quintoandar.com.br/imovel/x/"}
        result = _normalizar(raw, "SP", "SP")
        assert result["link"] == "https://www.quintoandar.com.br/imovel/x/"

    def test_tipo_apartment_por_padrao(self):
        result = _normalizar(VALID_LISTING, "SP", "SP")
        assert result["tipo"] == "apartment"

    def test_tipo_house(self):
        raw = {**VALID_LISTING, "type": "HOUSE"}
        result = _normalizar(raw, "SP", "SP")
        assert result["tipo"] == "house"

    def test_tipo_land(self):
        raw = {**VALID_LISTING, "type": "LAND"}
        result = _normalizar(raw, "SP", "SP")
        assert result["tipo"] == "land"

    def test_tipo_commercial(self):
        raw = {**VALID_LISTING, "type": "COMMERCIAL"}
        result = _normalizar(raw, "SP", "SP")
        assert result["tipo"] == "commercial"

    def test_tipo_studio_mapeia_apartment(self):
        raw = {**VALID_LISTING, "type": "STUDIO"}
        result = _normalizar(raw, "SP", "SP")
        assert result["tipo"] == "apartment"

    def test_todas_as_chaves_presentes(self):
        result = _normalizar(VALID_LISTING, "São Paulo", "São Paulo")
        expected_keys = {
            "id", "titulo", "descricao", "preco", "thumbnail", "link",
            "cidade", "estado", "bairro", "endereco", "tipo", "quartos",
            "banheiros", "area", "vagas_garagem", "suites", "condominio",
            "iptu", "latitude", "longitude", "negocio", "fonte",
        }
        assert expected_keys.issubset(set(result.keys()))


# ─── buscar ───────────────────────────────────────────────────────────────────

class TestBuscar:
    def test_retorna_lista_vazia_quando_html_none(self):
        with patch("src.external.quintoandar._fetch_html", return_value=None), \
             patch("src.external.quintoandar.time.sleep"):
            result = buscar("Guarulhos", "São Paulo")
        assert result == []

    def test_extrai_via_next_data(self):
        html = _make_next_data_html("homes", [VALID_LISTING, SALE_LISTING])
        with patch("src.external.quintoandar._fetch_html", return_value=html), \
             patch("src.external.quintoandar.time.sleep"):
            result = buscar("São Paulo", "São Paulo")
        assert len(result) == 2

    def test_extrai_via_json_scripts_quando_next_data_vazio(self):
        # HTML sem __NEXT_DATA__, mas com script JSON
        html = _make_json_script_html([VALID_LISTING])
        with patch("src.external.quintoandar._fetch_html", return_value=html), \
             patch("src.external.quintoandar.time.sleep"):
            result = buscar("São Paulo", "São Paulo")
        assert len(result) == 1

    def test_extrai_via_jsonld_como_fallback(self):
        # HTML só com JSON-LD
        html = _make_jsonld_html([VALID_LISTING, VALID_LISTING])
        with patch("src.external.quintoandar._fetch_html", return_value=html), \
             patch("src.external.quintoandar.time.sleep"):
            result = buscar("São Paulo", "São Paulo")
        assert len(result) == 2

    def test_retorna_vazio_quando_nenhuma_estrategia_encontra(self):
        html = "<html><body>sem dados</body></html>"
        with patch("src.external.quintoandar._fetch_html", return_value=html), \
             patch("src.external.quintoandar.time.sleep"):
            result = buscar("São Paulo", "São Paulo")
        assert result == []

    def test_respeita_limit(self):
        listings = [VALID_LISTING] * 10
        html = _make_next_data_html("homes", listings)
        with patch("src.external.quintoandar._fetch_html", return_value=html), \
             patch("src.external.quintoandar.time.sleep"):
            result = buscar("São Paulo", "São Paulo", limit=3)
        assert len(result) <= 3

    def test_filtro_quartos_min(self):
        # VALID_LISTING tem bedrooms=2; filtrar por ≥3 deve excluir
        html = _make_next_data_html("homes", [VALID_LISTING, VALID_LISTING])
        with patch("src.external.quintoandar._fetch_html", return_value=html), \
             patch("src.external.quintoandar.time.sleep"):
            result = buscar("São Paulo", "São Paulo", quartos_min=3)
        assert result == []

    def test_filtro_quartos_min_passa_quando_atende(self):
        listing = {**VALID_LISTING, "bedrooms": 4}
        html = _make_next_data_html("homes", [listing])
        with patch("src.external.quintoandar._fetch_html", return_value=html), \
             patch("src.external.quintoandar.time.sleep"):
            result = buscar("São Paulo", "São Paulo", quartos_min=3)
        assert len(result) == 1

    def test_filtro_preco_max(self):
        # VALID_LISTING tem rent=2500; filtrar por preco_max=1000 deve excluir
        html = _make_next_data_html("homes", [VALID_LISTING])
        with patch("src.external.quintoandar._fetch_html", return_value=html), \
             patch("src.external.quintoandar.time.sleep"):
            result = buscar("São Paulo", "São Paulo", preco_max=1000)
        assert result == []

    def test_filtro_preco_min(self):
        # VALID_LISTING tem rent=2500; filtrar por preco_min=5000 deve excluir
        html = _make_next_data_html("homes", [VALID_LISTING])
        with patch("src.external.quintoandar._fetch_html", return_value=html), \
             patch("src.external.quintoandar.time.sleep"):
            result = buscar("São Paulo", "São Paulo", preco_min=5000)
        assert result == []

    def test_filtro_area_min(self):
        # VALID_LISTING tem area=65.0; filtrar por area_min=100 deve excluir
        html = _make_next_data_html("homes", [VALID_LISTING])
        with patch("src.external.quintoandar._fetch_html", return_value=html), \
             patch("src.external.quintoandar.time.sleep"):
            result = buscar("São Paulo", "São Paulo", area_min=100)
        assert result == []

    def test_filtro_preco_zero_nao_filtrado(self):
        # Item com preco=0 não é filtrado por preco_max (preco=0 é inválido)
        listing = {k: v for k, v in VALID_LISTING.items() if k not in ("rent",)}
        listing = {**listing, "salePrice": 0}
        html = _make_next_data_html("homes", [listing])
        with patch("src.external.quintoandar._fetch_html", return_value=html), \
             patch("src.external.quintoandar.time.sleep"):
            result = buscar("São Paulo", "São Paulo", preco_max=1000)
        assert len(result) == 1  # preco=0 não é filtrado

    def test_fonte_nos_resultados(self):
        html = _make_next_data_html("homes", [VALID_LISTING])
        with patch("src.external.quintoandar._fetch_html", return_value=html), \
             patch("src.external.quintoandar.time.sleep"):
            result = buscar("São Paulo", "São Paulo")
        assert all(r["fonte"] == "quintoandar" for r in result)

    def test_sleep_chamado(self):
        html = _make_next_data_html("homes", [VALID_LISTING])
        with patch("src.external.quintoandar._fetch_html", return_value=html), \
             patch("src.external.quintoandar.time.sleep") as mock_sleep:
            buscar("São Paulo", "São Paulo")
        assert mock_sleep.called

    def test_url_construida_com_cidade_slug(self):
        html = "<html></html>"
        with patch("src.external.quintoandar._fetch_html", return_value=html) as mock_fetch, \
             patch("src.external.quintoandar.time.sleep"):
            buscar("São Paulo", "São Paulo")
        url_called = mock_fetch.call_args[0][0]
        assert "sao-paulo" in url_called

    def test_url_rental_usa_alugar(self):
        html = "<html></html>"
        with patch("src.external.quintoandar._fetch_html", return_value=html) as mock_fetch, \
             patch("src.external.quintoandar.time.sleep"):
            buscar("São Paulo", "São Paulo", negocio="RENTAL")
        url_called = mock_fetch.call_args[0][0]
        assert "alugar" in url_called

    def test_url_sale_usa_comprar(self):
        html = "<html></html>"
        with patch("src.external.quintoandar._fetch_html", return_value=html) as mock_fetch, \
             patch("src.external.quintoandar.time.sleep"):
            buscar("São Paulo", "São Paulo", negocio="SALE")
        url_called = mock_fetch.call_args[0][0]
        assert "comprar" in url_called

    def test_extra_kwargs_ignorados(self):
        """buscar() aceita e ignora kwargs desconhecidos (**_kw)."""
        html = _make_next_data_html("homes", [VALID_LISTING])
        with patch("src.external.quintoandar._fetch_html", return_value=html), \
             patch("src.external.quintoandar.time.sleep"):
            result = buscar(
                "São Paulo", "São Paulo",
                limit=5,
                unknown_param="test",
                zona_slug="zona-oeste",
                lat_busca=-23.5, lon_busca=-46.6,
            )
        assert isinstance(result, list)

    def test_resultado_tem_todas_as_chaves_obrigatorias(self):
        html = _make_next_data_html("homes", [VALID_LISTING])
        with patch("src.external.quintoandar._fetch_html", return_value=html), \
             patch("src.external.quintoandar.time.sleep"):
            result = buscar("São Paulo", "São Paulo")
        required = {"id", "titulo", "preco", "area", "quartos", "bairro",
                    "cidade", "estado", "tipo", "negocio", "fonte", "thumbnail"}
        assert len(result) == 1
        assert required.issubset(set(result[0].keys()))
