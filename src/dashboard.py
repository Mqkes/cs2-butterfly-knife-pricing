import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# ── Konfiguracja strony ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="CSFloat – Butterfly Knife EDA",
    page_icon="🦋",
    layout="wide"
)

# ── Wczytanie danych ─────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("data/csfloat_butterfly_knife_clean.csv")

    # Usunięcie rekordów z float_value == 0 (brak danych, nie rzeczywisty float)
    df = df[df["float_value"] > 0].copy()

    # Ekstrakcja nazwy skina z market_hash_name
    df["skin_name"] = df["market_hash_name"].str.replace(
        r"★ (StatTrak™ )?Butterfly Knife\s?\|?\s?", "", regex=True
    ).str.strip()
    df["skin_name"] = df["skin_name"].replace("", "Vanilla").fillna("Vanilla")

    # Kategoria wear na podstawie float_value
    bins   = [0, 0.07, 0.15, 0.38, 0.45, 1.0]
    labels = ["Factory New", "Minimal Wear", "Field-Tested", "Well-Worn", "Battle-Scarred"]
    df["wear"] = pd.cut(df["float_value"], bins=bins, labels=labels)

    return df

df = load_data()

# ── Nagłówek ─────────────────────────────────────────────────────────────────
st.title("🦋 CSFloat – Butterfly Knife | Eksploracyjna Analiza Danych")
st.caption(f"Liczba obserwacji po czyszczeniu: {len(df):,} | Źródło: CSFloat Marketplace")

# ── Metryki główne ───────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
col1.metric("Średnia cena", f"${df['price_usd'].mean():,.2f}")
col2.metric("Mediana ceny", f"${df['price_usd'].median():,.2f}")
col3.metric("Maks. cena",   f"${df['price_usd'].max():,.2f}")
col4.metric("Liczba skinów", df["skin_name"].nunique())

st.divider()

# ── Filtry w sidebarze ───────────────────────────────────────────────────────
st.sidebar.header("Filtry")

selected_skins = st.sidebar.multiselect(
    "Skin",
    options=sorted(df["skin_name"].unique()),
    default=sorted(df["skin_name"].unique())
)

selected_wear = st.sidebar.multiselect(
    "Wear",
    options=["Factory New", "Minimal Wear", "Field-Tested", "Well-Worn", "Battle-Scarred"],
    default=["Factory New", "Minimal Wear", "Field-Tested", "Well-Worn", "Battle-Scarred"]
)

stattrak_filter = st.sidebar.radio(
    "StatTrak",
    options=["Wszystkie", "Tylko StatTrak", "Bez StatTrak"]
)

price_min, price_max = st.sidebar.slider(
    "Zakres ceny (USD)",
    min_value=int(df["price_usd"].min()),
    max_value=int(df["price_usd"].max()),
    value=(int(df["price_usd"].min()), int(df["price_usd"].max()))
)

# ── Filtrowanie danych ───────────────────────────────────────────────────────
mask = (
    df["skin_name"].isin(selected_skins) &
    df["wear"].isin(selected_wear) &
    df["price_usd"].between(price_min, price_max)
)

if stattrak_filter == "Tylko StatTrak":
    mask &= df["is_stattrak"] == True
elif stattrak_filter == "Bez StatTrak":
    mask &= df["is_stattrak"] == False

dff = df[mask]

st.caption(f"Obserwacji po filtrach: {len(dff):,}")

# ── Wykres 1: Rozkład cen ────────────────────────────────────────────────────
st.subheader("Rozkład cen (USD)")
fig1 = px.histogram(
    dff, x="price_usd", nbins=80,
    color="is_stattrak",
    color_discrete_map={True: "#e05c5c", False: "#5c8de0"},
    labels={"price_usd": "Cena (USD)", "is_stattrak": "StatTrak"},
    template="plotly_dark"
)
st.plotly_chart(fig1, use_container_width=True)

# ── Wykres 2: Float vs Cena ──────────────────────────────────────────────────
st.subheader("Float Value vs Cena")
fig2 = px.scatter(
    dff, x="float_value", y="price_usd",
    color="wear",
    hover_data=["skin_name", "is_stattrak", "paint_seed"],
    opacity=0.6,
    labels={"float_value": "Float Value", "price_usd": "Cena (USD)", "wear": "Wear"},
    template="plotly_dark"
)
st.plotly_chart(fig2, use_container_width=True)

# ── Wykres 3: Średnia cena wg skina ─────────────────────────────────────────
st.subheader("Średnia cena według skina")
avg_price = (
    dff.groupby("skin_name")["price_usd"]
    .mean()
    .sort_values(ascending=False)
    .reset_index()
)
fig3 = px.bar(
    avg_price, x="skin_name", y="price_usd",
    labels={"skin_name": "Skin", "price_usd": "Średnia cena (USD)"},
    template="plotly_dark",
    color="price_usd",
    color_continuous_scale="Blues"
)
st.plotly_chart(fig3, use_container_width=True)

# ── Wykres 4: Średnia cena wg wear ───────────────────────────────────────────
st.subheader("Średnia cena według kategorii wear")
wear_order = ["Factory New", "Minimal Wear", "Field-Tested", "Well-Worn", "Battle-Scarred"]
avg_wear = (
    dff.groupby("wear", observed=True)["price_usd"]
    .mean()
    .reindex(wear_order)
    .reset_index()
)
fig4 = px.bar(
    avg_wear, x="wear", y="price_usd",
    labels={"wear": "Wear", "price_usd": "Średnia cena (USD)"},
    template="plotly_dark",
    color="price_usd",
    color_continuous_scale="Reds"
)
st.plotly_chart(fig4, use_container_width=True)

# ── Wykres 5: StatTrak premium ───────────────────────────────────────────────
st.subheader("StatTrak Premium")
fig5 = px.box(
    dff, x="is_stattrak", y="price_usd",
    color="is_stattrak",
    color_discrete_map={True: "#e05c5c", False: "#5c8de0"},
    labels={"is_stattrak": "StatTrak", "price_usd": "Cena (USD)"},
    template="plotly_dark"
)
st.plotly_chart(fig5, use_container_width=True)

# ── Tabela surowych danych ───────────────────────────────────────────────────
with st.expander("Podgląd surowych danych"):
    st.dataframe(dff.reset_index(drop=True), use_container_width=True)