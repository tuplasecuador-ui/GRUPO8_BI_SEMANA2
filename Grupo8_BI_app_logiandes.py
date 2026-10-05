"""
LogiAndes S.A. · Tablero de gestión (prototipo)
Prueba práctica 2 · Visualización, Storytelling y BI · UIDE

Ejecutar:  streamlit run app_logiandes.py
Necesita logiandes.csv en la misma carpeta. kpis_referencia.json (lo genera el
notebook) es opcional y sirve para verificar que la app y el notebook coinciden.
"""
from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

CARPETA = Path(__file__).resolve().parent
RUTA_CSV = CARPETA / "logiandes.csv"
RUTA_REFERENCIA = CARPETA / "kpis_referencia.json"

AZUL, NARANJA, GRIS, VERDE_OSC = "#1F4E79", "#D9822B", "#9AA5B1", "#2E7D5B"
CONFIG_PLOTLY = {"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]}
MIN_PEDIDOS = 100  # debajo de esto avisamos que la muestra es chica

SLA_POR_DEFECTO = {"Express": 120, "Estándar urbano": 180, "Programado": 240}

# capitales provinciales (la burbuja va ahí, no en la dirección del pedido)
CAPITALES = {
    "Guayas": ("Guayaquil", -2.19, -79.89),
    "Pichincha": ("Quito", -0.18, -78.47),
    "Azuay": ("Cuenca", -2.90, -79.00),
    "Manabí": ("Portoviejo", -1.05, -80.45),
    "Tungurahua": ("Ambato", -1.24, -78.62),
    "Loja": ("Loja", -4.00, -79.20),
}

# misma tabla de metas que el notebook (sección 3)
METAS = {
    "cumplimiento_sla": {"meta": 0.90, "alerta": 0.85, "sentido": "subir"},
    "tasa_reclamos": {"meta": 0.025, "alerta": 0.035, "sentido": "bajar"},
    "ventas_por_dia": {"meta": 1650, "alerta": 1500, "sentido": "subir"},
    "ticket_promedio": {"meta": 25.0, "alerta": 23.0, "sentido": "subir"},
    "p90_entrega": {"meta": 180, "alerta": 200, "sentido": "bajar"},
    "tiempo_atencion": {"meta": 8.0, "alerta": 9.0, "sentido": "bajar"},
    "ventas_expuestas": {"meta": 0.10, "alerta": 0.15, "sentido": "bajar"},
}

FICHAS = pd.DataFrame([
    ["K1", "Cumplimiento de SLA", "cumplimiento_sla", "Medir si se cumple la promesa de tiempo de cada canal",
     "pedidos con entrega ≤ SLA del canal / pedidos", "tiempo_individual_entrega_min, canal_entrega",
     "Diaria", "≥ 90%", "🟡 85–90% · 🔴 < 85%", "Operaciones"],
    ["K2", "Tasa de reclamos", "tasa_reclamos", "Medir la calidad percibida",
     "Σ reclamos_registrados / pedidos × 100", "reclamos_registrados", "Semanal", "≤ 2,5%",
     "🟡 2,5–3,5% · 🔴 > 3,5%", "Atención al cliente"],
    ["K3", "Ventas por día", "ventas_por_dia", "Seguir la venta sin el efecto del largo del mes",
     "Σ ventas_asociadas_usd / días con operación", "ventas_asociadas_usd, fecha_pedido", "Semanal",
     "≥ 1.650 USD", "🟡 1.500–1.650 · 🔴 < 1.500", "Comercial"],
    ["K4", "Ticket promedio", "ticket_promedio", "Vigilar el valor por pedido y la mezcla de canales",
     "Σ ventas_asociadas_usd / pedidos", "ventas_asociadas_usd", "Mensual", "≥ 25 USD",
     "🟡 23–25 · 🔴 < 23", "Comercial"],
    ["K5", "P90 del tiempo de entrega", "p90_entrega", "Medir la cola lenta de entregas",
     "percentil 90 de tiempo_individual_entrega_min", "tiempo_individual_entrega_min", "Diaria",
     "≤ 180 min", "🟡 180–200 · 🔴 > 200", "Operaciones"],
    ["K6", "Tiempo medio de atención", "tiempo_atencion", "Medir la agilidad del primer contacto",
     "promedio de tiempo_atencion_min", "tiempo_atencion_min", "Diaria", "≤ 8 min",
     "🟡 8–9 · 🔴 > 9", "Atención al cliente"],
    ["K7", "Ventas expuestas", "ventas_expuestas", "Cuantificar la venta ligada a pedidos tardíos",
     "Σ ventas fuera de SLA / Σ ventas × 100", "ventas_asociadas_usd, tiempo_individual_entrega_min",
     "Mensual", "≤ 10%", "🟡 10–15% · 🔴 > 15%", "Finanzas"],
], columns=["#", "Indicador", "clave", "Objetivo", "Fórmula", "Fuente", "Frecuencia", "Meta",
            "Umbral", "Responsable"])

st.set_page_config(page_title="LogiAndes · Tablero de gestión", page_icon="🚚",
                   layout="wide", initial_sidebar_state="expanded")


# ----------------------------------------------------------------------------- datos
@st.cache_data
def cargar_datos() -> pd.DataFrame:
    df = pd.read_csv(RUTA_CSV, encoding="utf-8-sig", parse_dates=["fecha_pedido"])
    df["motivo_reclamo"] = df["motivo_reclamo"].fillna("Sin reclamo")
    df["fecha"] = df["fecha_pedido"].dt.normalize()
    df["mes"] = df["fecha_pedido"].dt.to_period("M").astype(str)
    df["hora"] = df["fecha_pedido"].dt.hour
    df["semana"] = df["fecha"] - pd.to_timedelta(df["fecha"].dt.dayofweek, unit="D")
    return df


def aplicar_sla(df: pd.DataFrame, sla: dict) -> pd.DataFrame:
    d = df.copy()
    d["sla_min"] = d["canal_entrega"].map(sla)
    d["dentro_sla"] = d["tiempo_individual_entrega_min"] <= d["sla_min"]
    d["minutos_retraso"] = (d["tiempo_individual_entrega_min"] - d["sla_min"]).clip(lower=0)
    return d


# ----------------------------------------------------------------------------- KPIs
def calcular_kpis(d: pd.DataFrame) -> dict:
    """Idéntica a la del notebook. Si se cambia aquí, cambiarla allá."""
    if d.empty:
        return {}
    ventas = d["ventas_asociadas_usd"].sum()
    return {
        "pedidos": len(d),
        "cumplimiento_sla": d["dentro_sla"].mean(),
        "tasa_reclamos": d["reclamos_registrados"].sum() / len(d),
        "ventas_totales": ventas,
        "ventas_por_dia": ventas / d["fecha"].nunique(),
        "ticket_promedio": ventas / len(d),
        "p90_entrega": d["tiempo_individual_entrega_min"].quantile(0.90),
        "tiempo_atencion": d["tiempo_atencion_min"].mean(),
        "ventas_expuestas": d.loc[~d["dentro_sla"], "ventas_asociadas_usd"].sum() / ventas,
    }


def semaforo(clave: str, valor: float) -> tuple[str, str]:
    m = METAS[clave]
    if m["sentido"] == "subir":
        bien, regular = valor >= m["meta"], valor >= m["alerta"]
    else:
        bien, regular = valor <= m["meta"], valor <= m["alerta"]
    if bien:
        return "🟢", "en meta"
    return ("🟡", "vigilar") if regular else ("🔴", "crítico")


def intervalo_wilson(exitos: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (np.nan, np.nan)
    p = exitos / n
    centro = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    margen = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return centro - margen, centro + margen


FORMATOS = {
    "cumplimiento_sla": lambda v: f"{v:.1%}",
    "tasa_reclamos": lambda v: f"{v:.2%}",
    "ventas_por_dia": lambda v: f"${v:,.0f}",
    "ticket_promedio": lambda v: f"${v:,.2f}",
    "p90_entrega": lambda v: f"{v:.0f} min",
    "tiempo_atencion": lambda v: f"{v:.2f} min",
    "ventas_expuestas": lambda v: f"{v:.1%}",
}


def resumen_por(d: pd.DataFrame, columnas) -> pd.DataFrame:
    g = d.groupby(columnas, observed=True)
    r = g.agg(
        pedidos=("pedido_id", "count"),
        reclamos=("reclamos_registrados", "sum"),
        ventas=("ventas_asociadas_usd", "sum"),
        cumplimiento_sla=("dentro_sla", "mean"),
        ticket_promedio=("ventas_asociadas_usd", "mean"),
        p90_entrega=("tiempo_individual_entrega_min", lambda s: s.quantile(0.9)),
        tiempo_atencion=("tiempo_atencion_min", "mean"),
    )
    expuestas = d[~d["dentro_sla"]].groupby(columnas, observed=True)["ventas_asociadas_usd"].sum()
    r["ventas_fuera_sla"] = expuestas.reindex(r.index).fillna(0)
    r["ventas_expuestas"] = r["ventas_fuera_sla"] / r["ventas"]
    r["tasa_reclamos"] = r["reclamos"] / r["pedidos"]
    return r.reset_index()


def aviso_muestra(n: int, contexto: str = "la selección") -> None:
    if n < MIN_PEDIDOS:
        st.warning(f"⚠️ {contexto.capitalize()} tiene solo {n} pedidos. Las tasas pueden moverse mucho "
                   "por unos pocos casos; úsalas como pista, no como conclusión.")


def estilo(fig, alto: int = 380):
    fig.update_layout(template="plotly_white", height=alto, margin=dict(l=10, r=10, t=50, b=10),
                      font=dict(size=12), legend=dict(orientation="h", y=-0.18, title=None),
                      title=dict(x=0, xanchor="left", font=dict(size=14)))
    return fig


# ----------------------------------------------------------------------------- filtros
if not RUTA_CSV.exists():
    st.error(f"No encuentro logiandes.csv. Tiene que estar en la misma carpeta que la app ({CARPETA}).")
    st.stop()

BASE = cargar_datos()
FECHA_MIN, FECHA_MAX = BASE["fecha"].min().date(), BASE["fecha"].max().date()
CANALES = sorted(BASE["canal_entrega"].unique())
TIPOS = sorted(BASE["tipo_cliente"].unique())


def limpiar_filtros() -> None:
    for k in list(st.session_state.keys()):
        if k.startswith("f_") or k.startswith("sla_"):
            del st.session_state[k]


with st.sidebar:
    st.markdown("### 🚚 LogiAndes S.A.")
    st.caption("Tablero de gestión · I trimestre 2026")
    st.markdown("**Filtros**")
    rango = st.date_input("Periodo", value=(FECHA_MIN, FECHA_MAX), min_value=FECHA_MIN,
                          max_value=FECHA_MAX, format="DD/MM/YYYY", key="f_periodo")
    region = st.radio("Región", ["Todas", "Costa", "Sierra"], horizontal=True, key="f_region")
    provs_disp = sorted(BASE.loc[(BASE["region"] == region) | (region == "Todas"), "provincia"].unique())
    # la key depende de la región para que la lista se reinicie al cambiarla
    provincias = st.multiselect("Provincias", provs_disp, default=provs_disp, key=f"f_prov_{region}")
    canales = st.multiselect("Canal de entrega", CANALES, default=CANALES, key="f_canal")
    tipos = st.multiselect("Tipo de cliente", TIPOS, default=TIPOS, key="f_tipo")

    with st.expander("Supuestos de SLA (min)"):
        st.caption("Propuesta del equipo, pendiente de validar con Operaciones. "
                   "Cambiarlos modifica K1, K7 y todo lo que dependa de 'a tiempo'.")
        sla = {c: st.number_input(c, 30, 600, SLA_POR_DEFECTO[c], 10, key=f"sla_{c}")
               for c in SLA_POR_DEFECTO}
    st.button("Restablecer filtros", on_click=limpiar_filtros, width="stretch")

if isinstance(rango, (list, tuple)) and len(rango) == 2:
    desde, hasta = rango
else:  # mientras el usuario elige la segunda fecha
    desde, hasta = FECHA_MIN, FECHA_MAX

mascara = (
    BASE["fecha"].dt.date.between(desde, hasta)
    & BASE["provincia"].isin(provincias)
    & BASE["canal_entrega"].isin(canales)
    & BASE["tipo_cliente"].isin(tipos)
)
DATOS = aplicar_sla(BASE[mascara], sla)
TODO = aplicar_sla(BASE, sla)  # referencia nacional para comparar
SLA_MODIFICADO = sla != SLA_POR_DEFECTO

with st.sidebar:
    st.divider()
    st.caption(f"Mostrando **{len(DATOS):,}** de {len(BASE):,} pedidos")
    if SLA_MODIFICADO:
        st.caption("⚙️ Estás usando SLA distintos a la propuesta original.")


def sin_datos() -> bool:
    if DATOS.empty:
        st.info("No hay pedidos con estos filtros. Prueba ampliar el periodo o volver a marcar provincias.")
        return True
    return False


# ============================================================================= páginas
def pagina_ejecutiva() -> None:
    st.title("Vista ejecutiva")
    st.caption(f"Del {desde:%d/%m/%Y} al {hasta:%d/%m/%Y} · {len(DATOS):,} pedidos · "
               "Para Gerencia general, Comercial y Finanzas")
    if sin_datos():
        return
    aviso_muestra(len(DATOS))

    kpis = calcular_kpis(DATOS)

    # comparación: último mes de la selección contra el anterior
    meses = sorted(DATOS["mes"].unique())
    previo = None
    if len(meses) >= 2:
        actual = calcular_kpis(DATOS[DATOS["mes"] == meses[-1]])
        previo = calcular_kpis(DATOS[DATOS["mes"] == meses[-2]])
        etiqueta_delta = f"{meses[-1]} vs {meses[-2]}"

    orden = [("K1 Cumplimiento SLA", "cumplimiento_sla"), ("K2 Tasa de reclamos", "tasa_reclamos"),
             ("K3 Ventas por día", "ventas_por_dia"), ("K4 Ticket promedio", "ticket_promedio"),
             ("K5 P90 de entrega", "p90_entrega"), ("K6 Tiempo de atención", "tiempo_atencion"),
             ("K7 Ventas expuestas", "ventas_expuestas")]
    filas = [st.columns(4), st.columns(4)]
    celdas = filas[0] + filas[1][:3]
    for celda, (nombre, clave) in zip(celdas, orden):
        icono, estado = semaforo(clave, kpis[clave])
        delta, color_delta = None, "normal"
        if previo:
            dif = actual[clave] - previo[clave]
            es_pct = clave in ("cumplimiento_sla", "tasa_reclamos", "ventas_expuestas")
            delta = f"{dif * 100:+.1f} pp" if es_pct else FORMATOS[clave](dif).replace("$-", "-$")
            if not es_pct and dif > 0:
                delta = "+" + delta
            color_delta = "normal" if METAS[clave]["sentido"] == "subir" else "inverse"
        ficha = FICHAS.set_index("clave").loc[clave]
        with celda:
            st.metric(nombre, FORMATOS[clave](kpis[clave]), delta, delta_color=color_delta, border=True,
                      help=f"{ficha['Fórmula']}. Meta {ficha['Meta']}. Responsable: {ficha['Responsable']}.")
            st.caption(f"{icono} {estado} · meta {ficha['Meta']}")
    with filas[1][3]:
        with st.container(border=True):
            st.markdown(f"**Ventas del periodo**  \n### ${kpis['ventas_totales']:,.0f}")
            st.caption(f"{kpis['pedidos']:,} pedidos · {DATOS['fecha'].nunique()} días con operación")
    if previo:
        st.caption(f"Las flechas comparan {etiqueta_delta} (último mes completo de la selección contra el anterior).")

    # ------------------------------------------------------------- lectura rápida
    st.subheader("Lectura rápida")
    st.markdown(redactar_alertas(DATOS, kpis))

    # ------------------------------------------------------------- tendencia
    st.subheader("Evolución semanal")
    sem = DATOS.groupby("semana").agg(
        dias=("fecha", "nunique"), pedidos=("pedido_id", "count"),
        ventas=("ventas_asociadas_usd", "sum"), reclamos=("reclamos_registrados", "sum"),
        cumplimiento=("dentro_sla", "mean"))
    sem["ventas_dia"] = sem["ventas"] / sem["dias"]
    sem["tasa"] = sem["reclamos"] / sem["pedidos"]
    sem["completa"] = sem["dias"] == 7
    sem = sem.reset_index()

    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure(go.Bar(
            x=sem["semana"], y=sem["ventas_dia"],
            marker_color=[AZUL if c else GRIS for c in sem["completa"]],
            customdata=np.stack([sem["dias"], sem["pedidos"]], axis=1),
            hovertemplate="Semana del %{x|%d/%m}<br>%{y:$,.0f} por día<br>%{customdata[0]} días · "
                          "%{customdata[1]} pedidos<extra></extra>"))
        fig.add_hline(y=METAS["ventas_por_dia"]["meta"], line_dash="dash", line_color="black",
                      annotation_text="meta", annotation_position="top left")
        fig.update_yaxes(title="USD por día", rangemode="tozero")
        st.plotly_chart(estilo(fig.update_layout(title="Ventas promedio por día (barras grises = semana incompleta)")),
                        width="stretch", config=CONFIG_PLOTLY)
    with c2:
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=sem["semana"], y=sem["cumplimiento"], name="Cumplimiento SLA",
                                 mode="lines+markers", line=dict(color=AZUL, width=2.5),
                                 hovertemplate="%{y:.1%}<extra>SLA</extra>"), secondary_y=False)
        fig.add_trace(go.Scatter(x=sem["semana"], y=sem["tasa"], name="Tasa de reclamos",
                                 mode="lines+markers", line=dict(color=NARANJA, width=2.5, dash="dot"),
                                 marker_symbol="diamond",
                                 hovertemplate="%{y:.2%}<extra>Reclamos</extra>"), secondary_y=True)
        fig.update_yaxes(title="cumplimiento", tickformat=".0%", range=[0.5, 1], secondary_y=False)
        fig.update_yaxes(title="reclamos", tickformat=".0%", range=[0, 0.08], secondary_y=True,
                         showgrid=False)
        st.plotly_chart(estilo(fig.update_layout(title="Servicio: cumplimiento y reclamos por semana")),
                        width="stretch", config=CONFIG_PLOTLY)
        st.caption("Los dos ejes tienen escalas distintas a propósito (los reclamos son ~10 veces menores). "
                   "Leer cada línea contra su propio eje.")

    # ------------------------------------------------------------- comercial
    st.subheader("De dónde viene la venta")
    c1, c2 = st.columns([1.2, 1])
    with c1:
        mezcla = resumen_por(DATOS, ["tipo_cliente", "canal_entrega"])
        fig = px.bar(mezcla, y="tipo_cliente", x="ventas", color="canal_entrega", orientation="h",
                     color_discrete_map={"Estándar urbano": AZUL, "Express": NARANJA, "Programado": GRIS},
                     labels={"ventas": "Ventas (USD)", "tipo_cliente": "", "canal_entrega": "Canal"},
                     title="Ventas por tipo de cliente y canal",
                     hover_data={"pedidos": ":,", "ticket_promedio": ":$.2f", "ventas": ":$,.0f"})
        st.plotly_chart(estilo(fig, 330), width="stretch", config=CONFIG_PLOTLY)
    with c2:
        canal = resumen_por(DATOS, "canal_entrega").set_index("canal_entrega")
        tabla = pd.DataFrame({
            "% de la venta": canal["ventas"] / canal["ventas"].sum(),
            "Ticket": canal["ticket_promedio"],
            "Cumplimiento SLA": canal["cumplimiento_sla"],
            "Tasa reclamos": canal["tasa_reclamos"],
        })
        st.markdown("**Cada canal en una línea**")
        st.dataframe(tabla.style.format({"% de la venta": "{:.0%}", "Ticket": "${:.2f}",
                                         "Cumplimiento SLA": "{:.0%}", "Tasa reclamos": "{:.1%}"}),
                     width="stretch")
        st.caption("En el trimestre completo, Express deja el mejor ticket pero también la mayor tasa de reclamos: "
                   "crecer en Express sin capacidad para cumplirlo sube K4, pero también K2 y K7.")


def redactar_alertas(d: pd.DataFrame, kpis: dict) -> str:
    """Arma 3-4 frases a partir de los datos filtrados. Solo usa segmentos con muestra suficiente."""
    frases = []
    prov = resumen_por(d, "provincia")
    prov = prov[prov["pedidos"] >= MIN_PEDIDOS]
    if len(prov) >= 2:
        peor = prov.loc[prov["tasa_reclamos"].idxmax()]
        mejor = prov.loc[prov["tasa_reclamos"].idxmin()]
        frases.append(
            f"**{peor['provincia']}** tiene la tasa de reclamos más alta ({peor['tasa_reclamos']:.1%}), "
            f"{peor['tasa_reclamos'] / max(mejor['tasa_reclamos'], 1e-9):.1f} veces la de "
            f"{mejor['provincia']} ({mejor['tasa_reclamos']:.1%}).")
    celdas = resumen_por(d, ["provincia", "canal_entrega"])
    celdas = celdas[celdas["pedidos"] >= 50]
    if not celdas.empty:
        c = celdas.loc[celdas["cumplimiento_sla"].idxmin()]
        if c["cumplimiento_sla"] < METAS["cumplimiento_sla"]["meta"]:
            frases.append(f"El punto más débil de la operación es **{c['canal_entrega']} en {c['provincia']}**: "
                          f"solo {c['cumplimiento_sla']:.0%} de {int(c['pedidos'])} pedidos llegó dentro del SLA.")
    fuera = d[~d["dentro_sla"]]
    if len(fuera) and len(d) - len(fuera):
        r_fuera = fuera["reclamos_registrados"].mean()
        r_dentro = d[d["dentro_sla"]]["reclamos_registrados"].mean()
        monto = (f"Hay **${fuera['ventas_asociadas_usd'].sum():,.0f}** en ventas ligadas a pedidos tardíos "
                 f"({kpis['ventas_expuestas']:.0%} del total).")
        if len(fuera) >= 50 and r_dentro > 0 and r_fuera > r_dentro:
            frases.append(f"Un pedido fuera de plazo reclama **{r_fuera / r_dentro:.1f} veces** más que uno a tiempo "
                          f"({r_fuera:.1%} contra {r_dentro:.1%}). " + monto)
        else:
            frases.append(f"Con solo {len(fuera)} pedidos tardíos en la selección no se puede comparar bien "
                          f"su tasa de reclamo con la de los pedidos a tiempo. " + monto)
    motivos = d.loc[d["reclamos_registrados"] > 0, "motivo_reclamo"].value_counts(normalize=True)
    if len(motivos):
        frases.append(f"El motivo de reclamo más frecuente es **{motivos.index[0].lower()}** ({motivos.iloc[0]:.0%}).")
        if "Falta de actualización" in motivos.index and motivos.index[0] != "Falta de actualización":
            frases[-1] += (f" Le sigue la falta de actualización ({motivos['Falta de actualización']:.0%}), "
                           "que se podría atacar con avisos automáticos sin tocar la flota.")
    if not frases:
        return "_No hay suficientes datos en la selección para redactar alertas._"
    return "\n".join(f"- {f}" for f in frases)


def pagina_territorial() -> None:
    st.title("Análisis territorial")
    st.caption("Dónde está el problema y qué tan seguros podemos estar · Para Gerencia, Comercial y Operaciones")
    if sin_datos():
        return

    opciones = {
        "Tasa de reclamos": ("tasa_reclamos", ".1%", "bajar"),
        "Cumplimiento de SLA": ("cumplimiento_sla", ".0%", "subir"),
        "Ventas expuestas": ("ventas_expuestas", ".0%", "bajar"),
        "Ticket promedio": ("ticket_promedio", "$.2f", "subir"),
        "P90 de entrega (min)": ("p90_entrega", ".0f", "bajar"),
    }
    elegido = st.radio("Indicador a comparar", list(opciones), horizontal=True, key="t_indicador")
    clave, fmt, sentido = opciones[elegido]
    escala = "Blues" if sentido == "subir" else "Oranges"

    prov = resumen_por(DATOS, "provincia")
    prov["ciudad"] = prov["provincia"].map(lambda p: CAPITALES[p][0])
    prov["lat"] = prov["provincia"].map(lambda p: CAPITALES[p][1])
    prov["lon"] = prov["provincia"].map(lambda p: CAPITALES[p][2])

    c1, c2 = st.columns([1.1, 1])
    with c1:
        fig = px.scatter_geo(prov, lat="lat", lon="lon", size="pedidos", color=clave,
                             color_continuous_scale=escala, hover_name="provincia", size_max=45,
                             text="provincia",
                             hover_data={"lat": False, "lon": False, "ciudad": True, "pedidos": ":,",
                                         clave: f":{fmt}"},
                             title=f"{elegido} por provincia (tamaño = pedidos)")
        fig.update_traces(textposition="top center", marker=dict(line=dict(width=1, color="white")))
        fig.update_geos(showcountries=True, countrycolor="#BBB",
                        showland=True, landcolor="#F4F4F4", showocean=True, oceancolor="#EAF2F8",
                        lataxis_range=[-5.2, 1.6], lonaxis_range=[-81.5, -75],
                        resolution=50, projection_type="mercator")
        fig.update_layout(coloraxis_colorbar=dict(title="", tickformat=fmt.replace("$", "")))
        st.plotly_chart(estilo(fig, 430), width="stretch", config=CONFIG_PLOTLY)
        st.caption("Cada burbuja está en la capital provincial. No representa la ubicación real de los pedidos.")
    with c2:
        celdas = resumen_por(DATOS, ["provincia", "canal_entrega"])
        matriz = celdas.pivot(index="provincia", columns="canal_entrega", values=clave)
        n = celdas.pivot(index="provincia", columns="canal_entrega", values="pedidos").fillna(0).astype(int)
        formateo = (lambda v: f"{v:.0%}" if "%" in fmt else (f"${v:.2f}" if "$" in fmt else f"{v:.0f}"))
        texto = matriz.map(lambda v: "" if pd.isna(v) else formateo(v)) + n.map(
            lambda k: f"<br><sub>n={k}{' *' if k < 50 else ''}</sub>")
        fig = go.Figure(go.Heatmap(z=matriz.values, x=matriz.columns, y=matriz.index, text=texto.values,
                                   texttemplate="%{text}", colorscale=escala, showscale=False,
                                   hovertemplate="%{y} · %{x}<br>%{text}<extra></extra>"))
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(estilo(fig.update_layout(title=f"{elegido}: provincia × canal"), 430),
                        width="stretch", config=CONFIG_PLOTLY)
        st.caption("\\* celdas con menos de 50 pedidos: el valor es poco estable.")

    st.subheader("Ranking de reclamos con margen de error")
    c1, c2 = st.columns(2)
    with c1:
        ic = [intervalo_wilson(r, k) for r, k in zip(prov["reclamos"], prov["pedidos"])]
        prov["ic_inf"], prov["ic_sup"] = zip(*ic)
        rk = prov.sort_values("tasa_reclamos")
        colores = [NARANJA if t > METAS["tasa_reclamos"]["alerta"] else AZUL for t in rk["tasa_reclamos"]]
        fig = go.Figure(go.Scatter(
            x=rk["tasa_reclamos"], y=rk["provincia"], mode="markers",
            marker=dict(size=13, color=colores),
            error_x=dict(type="data", symmetric=False, array=rk["ic_sup"] - rk["tasa_reclamos"],
                         arrayminus=rk["tasa_reclamos"] - rk["ic_inf"], color=GRIS, thickness=1.5),
            customdata=np.stack([rk["reclamos"], rk["pedidos"], rk["ic_inf"], rk["ic_sup"]], axis=1),
            hovertemplate="%{y}: %{x:.1%}<br>%{customdata[0]} reclamos / %{customdata[1]} pedidos"
                          "<br>IC 95%: %{customdata[2]:.1%} – %{customdata[3]:.1%}<extra></extra>"))
        fig.add_vline(x=METAS["tasa_reclamos"]["meta"], line_dash="dash",
                      annotation_text="meta 2,5%", annotation_position="bottom right")
        fig.update_xaxes(tickformat=".0%", rangemode="tozero", title="tasa de reclamos")
        st.plotly_chart(estilo(fig.update_layout(title="Si dos barras grises se cruzan, la diferencia puede ser azar"), 360),
                        width="stretch", config=CONFIG_PLOTLY)
    with c2:
        nac = calcular_kpis(DATOS)
        fig = px.scatter(prov, x="pedidos", y="tasa_reclamos", size="ventas_fuera_sla", text="provincia",
                         size_max=50, color_discrete_sequence=[AZUL],
                         labels={"pedidos": "Pedidos (volumen)", "tasa_reclamos": "Tasa de reclamos",
                                 "ventas_fuera_sla": "Ventas fuera de SLA"},
                         title="Prioridad: volumen vs tasa (burbuja = USD fuera de SLA)")
        fig.update_traces(textposition="top center")
        fig.add_hline(y=nac["tasa_reclamos"], line_dash="dot", line_color=GRIS,
                      annotation_text="promedio selección", annotation_position="top left")
        fig.add_vline(x=prov["pedidos"].median(), line_dash="dot", line_color=GRIS)
        fig.update_yaxes(tickformat=".0%", rangemode="tozero")
        fig.update_xaxes(rangemode="tozero")
        st.plotly_chart(estilo(fig, 360), width="stretch", config=CONFIG_PLOTLY)
        st.caption("Arriba a la derecha: mucho volumen y mucha tasa. Es donde una mejora mueve más el total.")

    # ------------------------------------------------------------- ficha provincia
    st.subheader("Ficha de provincia")
    elegida = st.selectbox("Provincia", sorted(prov["provincia"]), key="t_prov")
    dp = DATOS[DATOS["provincia"] == elegida]
    kp = calcular_kpis(dp)
    aviso_muestra(len(dp), f"{elegida} con los filtros actuales")
    cols = st.columns(5)
    for col, (nombre, k) in zip(cols, [("Pedidos", "pedidos"), ("Cumplimiento SLA", "cumplimiento_sla"),
                                       ("Tasa de reclamos", "tasa_reclamos"), ("P90 entrega", "p90_entrega"),
                                       ("Ventas expuestas", "ventas_expuestas")]):
        if k == "pedidos":
            col.metric(nombre, f"{kp[k]:,}", f"{kp[k] / len(DATOS):.0%} del total", delta_color="off", border=True)
            continue
        dif = kp[k] - nac[k]
        es_pct = k != "p90_entrega"
        col.metric(nombre, FORMATOS[k](kp[k]),
                   f"{dif * 100:+.1f} pp vs selección" if es_pct else f"{dif:+.0f} min vs selección",
                   delta_color="normal" if METAS[k]["sentido"] == "subir" else "inverse", border=True)
    c1, c2 = st.columns(2)
    with c1:
        mot = (dp.loc[dp["reclamos_registrados"] > 0, "motivo_reclamo"].value_counts()
               .rename_axis("motivo").reset_index(name="reclamos"))
        if mot.empty:
            st.info("Sin reclamos registrados en esta selección.")
        else:
            fig = px.bar(mot, x="reclamos", y="motivo", orientation="h", text="reclamos",
                         color_discrete_sequence=[NARANJA], title=f"Motivos de reclamo en {elegida}")
            fig.update_yaxes(title="", autorange="reversed")
            st.plotly_chart(estilo(fig, 300), width="stretch", config=CONFIG_PLOTLY)
    with c2:
        porcanal = resumen_por(dp, "canal_entrega")
        fig = px.bar(porcanal, x="canal_entrega", y="cumplimiento_sla", text_auto=".0%",
                     color_discrete_sequence=[AZUL], title=f"Cumplimiento por canal en {elegida}",
                     hover_data={"pedidos": True}, labels={"canal_entrega": "", "cumplimiento_sla": ""})
        fig.add_hline(y=METAS["cumplimiento_sla"]["meta"], line_dash="dash", annotation_text="meta 90%")
        fig.update_yaxes(tickformat=".0%", range=[0, 1.05])
        st.plotly_chart(estilo(fig, 300), width="stretch", config=CONFIG_PLOTLY)


def pagina_operativa() -> None:
    st.title("Detalle operativo")
    st.caption("Qué pasa pedido a pedido · Para Operaciones y Atención al cliente")
    if sin_datos():
        return
    aviso_muestra(len(DATOS))

    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        for canal_i in sorted(DATOS["canal_entrega"].unique()):
            t = DATOS.loc[DATOS["canal_entrega"] == canal_i, "tiempo_individual_entrega_min"]
            fig.add_trace(go.Box(x=t, name=canal_i, marker_color=AZUL, boxpoints="outliers",
                                 hovertemplate="%{x:.0f} min<extra></extra>", showlegend=False))
            fig.add_trace(go.Scatter(x=[sla[canal_i]], y=[canal_i], mode="markers",
                                     marker=dict(symbol="line-ns-open", size=34, color=NARANJA,
                                                 line=dict(width=3)),
                                     name="SLA", showlegend=canal_i == sorted(DATOS["canal_entrega"].unique())[0],
                                     hovertemplate=f"SLA {sla[canal_i]} min<extra></extra>"))
        fig.update_xaxes(title="minutos", rangemode="tozero")
        st.plotly_chart(estilo(fig.update_layout(title="Tiempo de entrega por canal (línea naranja = SLA)"), 360),
                        width="stretch", config=CONFIG_PLOTLY)
    with c2:
        ef = DATOS.groupby(["canal_entrega", "dentro_sla"]).agg(
            pedidos=("pedido_id", "count"), reclamos=("reclamos_registrados", "sum")).reset_index()
        ef["tasa"] = ef["reclamos"] / ef["pedidos"]
        ef["estado"] = ef["dentro_sla"].map({True: "A tiempo", False: "Fuera de SLA"})
        fig = px.bar(ef, x="canal_entrega", y="tasa", color="estado", barmode="group", text_auto=".1%",
                     category_orders={"estado": ["A tiempo", "Fuera de SLA"]},
                     color_discrete_map={"A tiempo": AZUL, "Fuera de SLA": NARANJA},
                     hover_data={"pedidos": True, "reclamos": True},
                     labels={"tasa": "tasa de reclamo", "canal_entrega": ""},
                     title="Tasa de reclamo dentro y fuera de plazo")
        fig.update_yaxes(tickformat=".0%", rangemode="tozero")
        st.plotly_chart(estilo(fig, 360), width="stretch", config=CONFIG_PLOTLY)
        chicos = ef[ef["pedidos"] < MIN_PEDIDOS]
        if not chicos.empty:
            st.caption("Ojo: " + ", ".join(f"{r.canal_entrega} {r.estado.lower()} ({r.pedidos} pedidos)"
                                           for r in chicos.itertuples()) + " tienen poca muestra.")

    c1, c2 = st.columns(2)
    with c1:
        mot = (DATOS.loc[DATOS["reclamos_registrados"] > 0, "motivo_reclamo"].value_counts()
               .rename_axis("motivo").reset_index(name="reclamos"))
        if mot.empty:
            st.info("No hay reclamos en la selección.")
        else:
            mot["acumulado"] = mot["reclamos"].cumsum() / mot["reclamos"].sum()
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Bar(x=mot["motivo"], y=mot["reclamos"], name="Reclamos",
                                 marker_color=[NARANJA if a <= 0.7 else AZUL for a in mot["acumulado"]]),
                          secondary_y=False)
            fig.add_trace(go.Scatter(x=mot["motivo"], y=mot["acumulado"], name="% acumulado",
                                     mode="lines+markers", line=dict(color="black", width=1.5)),
                          secondary_y=True)
            fig.update_yaxes(tickformat=".0%", range=[0, 1.05], secondary_y=True, showgrid=False)
            st.plotly_chart(estilo(fig.update_layout(title="Pareto de motivos de reclamo"), 360),
                            width="stretch", config=CONFIG_PLOTLY)
    with c2:
        hora = DATOS.groupby("hora").agg(pedidos=("pedido_id", "count"),
                                         cumplimiento=("dentro_sla", "mean")).reset_index()
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(x=hora["hora"], y=hora["pedidos"], name="Pedidos", marker_color=GRIS),
                      secondary_y=False)
        fig.add_trace(go.Scatter(x=hora["hora"], y=hora["cumplimiento"], name="Cumplimiento SLA",
                                 mode="lines+markers", line=dict(color=AZUL, width=2.5)), secondary_y=True)
        fig.update_xaxes(title="hora del pedido", dtick=1)
        fig.update_yaxes(title="pedidos", secondary_y=False)
        fig.update_yaxes(tickformat=".0%", range=[0.5, 1], secondary_y=True, showgrid=False)
        st.plotly_chart(estilo(fig.update_layout(title="Carga y cumplimiento por hora del día"), 360),
                        width="stretch", config=CONFIG_PLOTLY)
        tarde = DATOS[DATOS["hora"] >= 16]
        if len(tarde) >= MIN_PEDIDOS:
            st.caption(f"Pedidos desde las 16h: {tarde['dentro_sla'].mean():.0%} a tiempo, contra "
                       f"{DATOS.loc[DATOS['hora'] < 16, 'dentro_sla'].mean():.0%} del resto del día, "
                       "aunque son menos pedidos. Hipótesis a revisar: cambio de turno o tráfico de salida.")

    with st.expander("Atención al cliente: tiempo de atención vs reclamos"):
        tramos = DATOS.assign(tramo=pd.cut(DATOS["tiempo_atencion_min"], [0, 6, 8, 10, 99],
                                           labels=["≤ 6 min", "6–8 min", "8–10 min", "> 10 min"]))
        t = tramos.groupby("tramo", observed=True).agg(pedidos=("pedido_id", "count"),
                                                       tasa=("reclamos_registrados", "mean")).reset_index()
        fig = px.bar(t, x="tramo", y="tasa", text_auto=".1%", color_discrete_sequence=[NARANJA],
                     hover_data={"pedidos": True}, labels={"tramo": "tiempo de atención", "tasa": "tasa de reclamo"})
        fig.update_yaxes(tickformat=".0%", rangemode="tozero")
        st.plotly_chart(estilo(fig, 300), width="stretch", config=CONFIG_PLOTLY)

    # ------------------------------------------------------------- tabla
    st.subheader("Pedidos fuera de SLA")
    fuera = DATOS[~DATOS["dentro_sla"]].sort_values("minutos_retraso", ascending=False)
    c1, c2, c3 = st.columns([1, 1, 2])
    c1.metric("Pedidos fuera de SLA", f"{len(fuera):,}", f"{len(fuera) / len(DATOS):.1%} del total",
              delta_color="off", border=True)
    c2.metric("Retraso mediano", f"{fuera['minutos_retraso'].median():.0f} min" if len(fuera) else "—", border=True)
    buscar = c3.text_input("Buscar por código de pedido", placeholder="LA-00…", key="o_buscar")
    if buscar:
        fuera = fuera[fuera["pedido_id"].str.contains(buscar.strip(), case=False, na=False)]
    columnas = ["pedido_id", "fecha_pedido", "provincia", "canal_entrega", "tipo_cliente", "distancia_km",
                "tiempo_individual_entrega_min", "sla_min", "minutos_retraso", "motivo_reclamo"]
    st.dataframe(
        fuera[columnas], hide_index=True, width="stretch", height=320,
        column_config={
            "pedido_id": "Pedido", "fecha_pedido": st.column_config.DatetimeColumn("Fecha", format="DD/MM/YY HH:mm"),
            "provincia": "Provincia", "canal_entrega": "Canal", "tipo_cliente": "Cliente",
            "distancia_km": st.column_config.NumberColumn("Km", format="%.1f"),
            "tiempo_individual_entrega_min": st.column_config.NumberColumn("Entrega (min)", format="%.0f"),
            "sla_min": st.column_config.NumberColumn("SLA", format="%d"),
            "minutos_retraso": st.column_config.ProgressColumn(
                "Retraso (min)", format="%.0f", min_value=0,
                max_value=float(max(fuera["minutos_retraso"].max(), 1)) if len(fuera) else 1.0),
            "motivo_reclamo": "Reclamo",
        })
    st.download_button("Descargar pedidos fuera de SLA (CSV)",
                       data=fuera[columnas].to_csv(index=False).encode("utf-8-sig"),
                       file_name="logiandes_fuera_sla.csv", mime="text/csv")
    st.caption("La descarga incluye solo lo necesario para gestionar los retrasos. "
               "No contiene datos personales, pero es de uso interno.")


def pagina_gobernanza() -> None:
    st.title("Fichas de KPI y gobernanza")
    st.caption("Cómo se calcula cada número, quién responde por él y qué tan confiable es")

    st.subheader("Fichas técnicas")
    st.dataframe(FICHAS.drop(columns="clave"), hide_index=True, width="stretch")

    st.subheader("Supuestos activos")
    sup = pd.DataFrame({"Canal": list(sla), "SLA en uso (min)": list(sla.values()),
                        "Propuesta original (min)": [SLA_POR_DEFECTO[c] for c in sla]})
    st.dataframe(sup, hide_index=True)
    if SLA_MODIFICADO:
        st.warning("Los SLA en uso no son los de la propuesta. Los KPIs K1 y K7 no son comparables con el notebook.")

    st.subheader("Verificación contra el notebook")
    if not RUTA_REFERENCIA.exists():
        st.info("No está kpis_referencia.json. Ejecuta el notebook completo y copia el archivo junto a la app.")
    else:
        ref = json.loads(RUTA_REFERENCIA.read_text(encoding="utf-8"))
        app = calcular_kpis(aplicar_sla(BASE, ref["sla_por_canal"]))
        filas = []
        for k, v_ref in ref["kpis"].items():
            v_app = float(app[k])
            ok = abs(v_app - v_ref) <= 1e-4 * max(1, abs(v_ref))
            filas.append({"KPI": k, "Notebook": v_ref, "App": round(v_app, 6), "Coincide": "✔️" if ok else "❌"})
        tabla = pd.DataFrame(filas)
        st.dataframe(tabla, hide_index=True, width="stretch")
        if (tabla["Coincide"] == "✔️").all():
            st.success(f"Los {len(tabla)} valores coinciden con el notebook (sin filtros, mismos SLA, "
                       f"{ref['filas']:,} filas).")
        else:
            st.error("Hay diferencias entre el notebook y la app. Revisar que ambos usen el mismo CSV y la misma función.")

    st.subheader("Límites que hay que tener presentes")
    st.markdown("""
- **Tres meses de datos.** No alcanza para ver estacionalidad.
- **Metas y SLA propuestos por el equipo**, no acordados con la empresa. Un 🔴 significa "no cumple nuestra propuesta".
- **Sin costos.** *Ventas expuestas* es venta en riesgo, no pérdida contable.
- **Valores recortados en el origen**: 36 entregas en exactamente 28 min, 4 en exactamente 520 min, 17 pedidos en 45 km. Los extremos pueden estar subestimados.
- **Correlación ≠ causalidad.** Que los pedidos tardíos reclamen más no prueba que el retraso sea la única causa.
- **Reclamos registrados** solo cuenta los que llegaron por canales formales.
""")

    with st.expander("Diccionario de datos"):
        st.dataframe(pd.DataFrame([
            ["pedido_id", "Código interno del pedido", "texto"],
            ["fecha_pedido", "Fecha y hora en que se registró el pedido", "fecha/hora"],
            ["provincia / region", "Provincia y región (Costa, Sierra) del destino", "categoría"],
            ["canal_entrega", "Estándar urbano, Express o Programado", "categoría"],
            ["tipo_cliente", "Comercio minorista, PyME o Corporativo", "categoría"],
            ["tiempo_atencion_min", "Minutos del primer contacto / gestión del pedido", "número"],
            ["tiempo_individual_entrega_min", "Minutos desde el despacho hasta la entrega", "número"],
            ["reclamos_registrados", "1 si el pedido tuvo reclamo formal, 0 si no", "0/1"],
            ["motivo_reclamo", "Motivo declarado (vacío = sin reclamo)", "categoría"],
            ["ventas_asociadas_usd", "Venta asociada al pedido (definición por confirmar)", "USD"],
            ["distancia_km", "Distancia del recorrido", "km"],
        ], columns=["Campo", "Descripción", "Tipo"]), hide_index=True, width="stretch")


# ----------------------------------------------------------------------------- navegación
paginas = st.navigation({
    "Tablero": [
        st.Page(pagina_ejecutiva, title="Vista ejecutiva", icon="📊", default=True),
        st.Page(pagina_territorial, title="Análisis territorial", icon="🗺️"),
        st.Page(pagina_operativa, title="Detalle operativo", icon="🛠️"),
    ],
    "Soporte": [st.Page(pagina_gobernanza, title="Fichas y gobernanza", icon="📘")],
})
paginas.run()
