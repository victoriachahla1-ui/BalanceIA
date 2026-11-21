"""BalanceIA - Diagnóstico financiero automático de estados contables.
Ejecutar con:  streamlit run app.py   (requiere balanceia.py en la misma carpeta)
"""
import os
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from balanceia import (PARTIDAS, SECTORES, RATIOS, simular_empresas, tabla_ratios, calcular_ratios,
                       completar_balance, z_altman, zona_altman, dupont, evaluar,
                       diagnostico_texto, diagnostico_llm)

st.set_page_config(page_title="BalanceIA", page_icon="📊", layout="wide")

ETIQUETAS = {"caja": "Caja y bancos", "creditos": "Créditos por ventas", "bienes_cambio": "Bienes de cambio",
             "activo_no_corriente": "Activo no corriente", "pasivo_corriente": "Pasivo corriente",
             "pasivo_no_corriente": "Pasivo no corriente", "resultados_acumulados": "Resultados acumulados",
             "ventas": "Ventas", "costo_ventas": "Costo de ventas", "gastos_operativos": "Gastos operativos",
             "intereses": "Intereses pagados", "impuestos": "Impuesto a las ganancias"}
EJEMPLOS = {
    "Distribuidora del Sur S.A. (en problemas)": ("Comercio", dict(caja=55, creditos=470, bienes_cambio=610,
        activo_no_corriente=700, pasivo_corriente=930, pasivo_no_corriente=340, resultados_acumulados=90,
        ventas=2520, costo_ventas=1900, gastos_operativos=500, intereses=190, impuestos=0)),
    "Metalúrgica Andina S.R.L. (sana)": ("Industria", dict(caja=180, creditos=260, bienes_cambio=300,
        activo_no_corriente=900, pasivo_corriente=330, pasivo_no_corriente=250, resultados_acumulados=520,
        ventas=1900, costo_ventas=1230, gastos_operativos=390, intereses=45, impuestos=70)),
    "Cargar mi propio balance": ("Servicios", {k: 0.0 for k in PARTIDAS}),
}
FEATS = list(RATIOS)


def preparar(Rdf):
    X = Rdf[FEATS].copy()
    X["endeudamiento"] = X["endeudamiento"].fillna(20).clip(upper=20)
    X["roe"] = X["roe"].fillna(-1).clip(-2, 2)
    X["cobertura_intereses"] = X["cobertura_intereses"].clip(-10, 30)
    return X.fillna(X.median())


@st.cache_resource
def entrenar():
    df = simular_empresas(600, 42)
    R = tabla_ratios(df)
    modelo = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(preparar(R), df["dificultad"])
    bench = pd.concat([df[["sector"]], R], axis=1).groupby("sector").median()
    return modelo, bench


modelo, bench = entrenar()

# ---------------- barra lateral: carga del balance ----------------
st.sidebar.header("📥 Balance a analizar")
ej = st.sidebar.selectbox("Empresa", list(EJEMPLOS))
sector_ej, valores = EJEMPLOS[ej]
nombre = st.sidebar.text_input("Nombre de la empresa", "" if ej.startswith("Cargar") else ej.split(" (")[0])
sector = st.sidebar.selectbox("Sector", SECTORES, index=SECTORES.index(sector_ej))
archivo = st.sidebar.file_uploader("…o subí un CSV con las partidas", type=["csv"])
origen = ""
if archivo is not None:
    valores = pd.read_csv(archivo).iloc[0][PARTIDAS].astype(float).to_dict()
    origen = archivo.name
st.sidebar.caption("Valores en millones de $")
b = {}
with st.sidebar.expander("Estado de situación patrimonial", expanded=False):
    for k in PARTIDAS[:7]:
        b[k] = st.number_input(ETIQUETAS[k], value=float(valores[k]), step=10.0, key=f"{ej}{origen}{k}")
with st.sidebar.expander("Estado de resultados", expanded=False):
    for k in PARTIDAS[7:]:
        b[k] = st.number_input(ETIQUETAS[k], value=float(valores[k]), step=10.0, key=f"{ej}{origen}{k}")

st.title("📊 BalanceIA")
st.write("Diagnóstico financiero automático de estados contables: ratios, comparación con el sector, "
         "score de riesgo con Machine Learning e informe redactado por IA.")

if b["ventas"] <= 0 or (b["caja"] + b["creditos"] + b["bienes_cambio"] + b["activo_no_corriente"]) <= 0:
    st.info("👈 Completá las partidas del balance en la barra lateral (o elegí un ejemplo) para ver el diagnóstico.")
    st.stop()

bb = completar_balance(b)
r = calcular_ratios(b)
z = z_altman(b)
prob = modelo.predict_proba(preparar(pd.DataFrame([r])))[0, 1]
tabla = evaluar(b, bench.loc[sector])
riesgo = "ALTO" if prob >= 0.4 else ("MODERADO" if prob >= 0.2 else "BAJO")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Probabilidad de dificultad", f"{100*prob:.0f}%", riesgo, delta_color="off")
c2.metric("Z'' de Altman", f"{z:.2f}", f"Zona {zona_altman(z).lower()}", delta_color="off")
c3.metric("Liquidez corriente", f"{r['liquidez_corriente']:.2f}")
c4.metric("Endeudamiento (P/PN)", "PN negativo" if pd.isna(r["endeudamiento"]) else f"{r['endeudamiento']:.2f}")
c5.metric("ROE", "n/d" if pd.isna(r["roe"]) else f"{100*r['roe']:.1f}%")
{"ALTO": st.error, "MODERADO": st.warning, "BAJO": st.success}[riesgo](
    f"Riesgo de dificultad financiera **{riesgo}** para {nombre or 'la empresa'} "
    f"(modelo logit entrenado con 600 PyMEs; {int((tabla['Estado']=='🔴').sum())} de {len(tabla)} ratios en rojo frente al sector {sector}).")

tab1, tab2, tab3 = st.tabs(["🚦 Ratios vs. sector", "📈 Gráficos", "📝 Diagnóstico IA"])
with tab1:
    st.dataframe(tabla.drop(columns=["clave", "valor"]), hide_index=True, height=530)
with tab2:
    g1, g2 = st.columns(2)
    with g1:
        fig, ax = plt.subplots(figsize=(6, 4))
        claves = ["liquidez_corriente", "prueba_acida", "solvencia", "rotacion_activos"]
        x = np.arange(len(claves))
        ax.bar(x - .2, [r[k] for k in claves], .4, label=nombre or "Empresa", color="#1f5aa6")
        ax.bar(x + .2, [bench.loc[sector, k] for k in claves], .4, label=f"Mediana {sector}", color="#e07b39")
        ax.set_xticks(x, [RATIOS[k][0].split(" (")[0] for k in claves], fontsize=8)
        ax.set_title("Empresa vs. sector"); ax.legend(); ax.grid(alpha=.3)
        st.pyplot(fig)
    with g2:
        fig, ax = plt.subplots(figsize=(6, 4))
        partes = {"Caja": b["caja"], "Créditos": b["creditos"], "Bienes de cambio": b["bienes_cambio"],
                  "Activo no corriente": b["activo_no_corriente"]}
        fin = {"Pasivo corriente": b["pasivo_corriente"], "Pasivo no corriente": b["pasivo_no_corriente"],
               "Patrimonio neto": max(bb["patrimonio_neto"], 0)}
        izq = 0
        for (k, v), c in zip(partes.items(), ["#9ecae1", "#6baed6", "#3182bd", "#08519c"]):
            ax.bar("Activo", v, bottom=izq, color=c, label=k); izq += v
        izq = 0
        for (k, v), c in zip(fin.items(), ["#fdae6b", "#e6550d", "#31a354"]):
            ax.bar("Pasivo + PN", v, bottom=izq, color=c, label=k); izq += v
        ax.set_title("Estructura del balance (millones de $)"); ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=4, frameon=False)
        st.pyplot(fig)
    d = dupont(b)
    st.caption(f"DuPont: ROE {100*d['roe']:.1f}% = margen neto {100*d['margen_neto']:.1f}% × rotación "
               f"{d['rotacion']:.2f} × apalancamiento {d['apalancamiento']:.2f}")
with tab3:
    contexto = {"empresa": nombre, "sector": sector, "probabilidad_dificultad": round(prob, 2), "z_altman": round(z, 2),
                "ratios": dict(zip(tabla["Ratio"], tabla["Empresa"])), "semaforos": dict(zip(tabla["Ratio"], tabla["Estado"]))}
    texto = None
    if os.getenv("ANTHROPIC_API_KEY") and st.button("✨ Redactar con IA generativa (LLM)"):
        try:
            texto = diagnostico_llm(contexto)
        except Exception as e:
            st.warning(f"No se pudo usar el LLM ({type(e).__name__}); se muestra el diagnóstico por reglas.")
    if texto is None:
        texto = diagnostico_texto(nombre or "Empresa", sector, b, tabla, prob, z)
    st.write(texto)
    st.download_button("⬇️ Descargar diagnóstico (.txt)", texto.encode("utf-8"), "diagnostico_balanceia.txt")
