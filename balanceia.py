"""BalanceIA - funciones de diagnóstico financiero de estados contables.
Módulo compartido por el notebook y por la app Streamlit (app.py).
"""
import numpy as np
import pandas as pd

# Partidas de un balance simplificado + estado de resultados (en millones de $)
PARTIDAS = ["caja", "creditos", "bienes_cambio", "activo_no_corriente",
            "pasivo_corriente", "pasivo_no_corriente", "resultados_acumulados",
            "ventas", "costo_ventas", "gastos_operativos", "intereses", "impuestos"]

SECTORES = ["Comercio", "Industria", "Servicios", "Agro", "Construcción"]

# Ratios: (nombre legible, ¿más alto es mejor?, categoría)
RATIOS = {
    "liquidez_corriente": ("Liquidez corriente", True, "Liquidez"),
    "prueba_acida": ("Prueba ácida", True, "Liquidez"),
    "liquidez_inmediata": ("Liquidez inmediata", True, "Liquidez"),
    "endeudamiento": ("Endeudamiento (Pasivo/PN)", False, "Solvencia"),
    "solvencia": ("Solvencia (Activo/Pasivo)", True, "Solvencia"),
    "cobertura_intereses": ("Cobertura de intereses (EBIT/Int.)", True, "Solvencia"),
    "margen_bruto": ("Margen bruto", True, "Rentabilidad"),
    "margen_operativo": ("Margen operativo", True, "Rentabilidad"),
    "margen_neto": ("Margen neto", True, "Rentabilidad"),
    "roa": ("ROA", True, "Rentabilidad"),
    "roe": ("ROE", True, "Rentabilidad"),
    "rotacion_activos": ("Rotación de activos", True, "Eficiencia"),
    "dias_cobro": ("Días de cobro", False, "Eficiencia"),
    "dias_inventario": ("Días de inventario", False, "Eficiencia"),
}
PORCENTUALES = {"margen_bruto", "margen_operativo", "margen_neto", "roa", "roe"}


# ------------------------------------------------------------------ cálculos
def completar_balance(b: dict) -> dict:
    """Agrega totales y resultados intermedios a un balance simplificado."""
    b = dict(b)
    b["activo_corriente"] = b["caja"] + b["creditos"] + b["bienes_cambio"]
    b["activo_total"] = b["activo_corriente"] + b["activo_no_corriente"]
    b["pasivo_total"] = b["pasivo_corriente"] + b["pasivo_no_corriente"]
    b["patrimonio_neto"] = b["activo_total"] - b["pasivo_total"]
    b["resultado_bruto"] = b["ventas"] - b["costo_ventas"]
    b["ebit"] = b["resultado_bruto"] - b["gastos_operativos"]
    b["resultado_neto"] = b["ebit"] - b["intereses"] - b["impuestos"]
    b["capital_trabajo"] = b["activo_corriente"] - b["pasivo_corriente"]
    return b


def calcular_ratios(b: dict) -> dict:
    """Calcula los ratios financieros clásicos de un balance."""
    b = completar_balance(b)
    div = lambda a, c: a / c if c not in (0, 0.0) else np.nan
    r = {
        "liquidez_corriente": div(b["activo_corriente"], b["pasivo_corriente"]),
        "prueba_acida": div(b["activo_corriente"] - b["bienes_cambio"], b["pasivo_corriente"]),
        "liquidez_inmediata": div(b["caja"], b["pasivo_corriente"]),
        "endeudamiento": div(b["pasivo_total"], b["patrimonio_neto"]) if b["patrimonio_neto"] > 0 else np.nan,
        "solvencia": div(b["activo_total"], b["pasivo_total"]),
        "cobertura_intereses": div(b["ebit"], b["intereses"]),
        "margen_bruto": div(b["resultado_bruto"], b["ventas"]),
        "margen_operativo": div(b["ebit"], b["ventas"]),
        "margen_neto": div(b["resultado_neto"], b["ventas"]),
        "roa": div(b["resultado_neto"], b["activo_total"]),
        "roe": div(b["resultado_neto"], b["patrimonio_neto"]) if b["patrimonio_neto"] > 0 else np.nan,
        "rotacion_activos": div(b["ventas"], b["activo_total"]),
        "dias_cobro": div(b["creditos"], b["ventas"]) * 365,
        "dias_inventario": div(b["bienes_cambio"], b["costo_ventas"]) * 365,
    }
    return r


def z_altman(b: dict) -> float:
    """Z''-score de Altman para mercados emergentes (Altman, 2005)."""
    b = completar_balance(b)
    at = b["activo_total"]
    x1 = b["capital_trabajo"] / at
    x2 = b["resultados_acumulados"] / at
    x3 = b["ebit"] / at
    x4 = b["patrimonio_neto"] / b["pasivo_total"]
    return 3.25 + 6.56 * x1 + 3.26 * x2 + 6.72 * x3 + 1.05 * x4


def zona_altman(z: float) -> str:
    if z > 5.85:
        return "Segura"
    if z >= 4.15:
        return "Gris"
    return "Riesgo"


def dupont(b: dict) -> dict:
    """Descomposición DuPont: ROE = margen neto x rotación x apalancamiento."""
    b = completar_balance(b)
    return {"margen_neto": b["resultado_neto"] / b["ventas"],
            "rotacion": b["ventas"] / b["activo_total"],
            "apalancamiento": b["activo_total"] / b["patrimonio_neto"],
            "roe": b["resultado_neto"] / b["patrimonio_neto"]}


# --------------------------------------------------------- datos simulados
def simular_empresas(n=600, seed=42) -> pd.DataFrame:
    """Simula balances de n PyMEs argentinas con parámetros por sector.
    La variable 'dificultad' indica si la empresa entró en dificultad financiera
    (cheques rechazados / atraso de deuda) al año siguiente."""
    rng = np.random.default_rng(seed)
    # parámetros por sector: margen bruto medio, rotación media, % activo corriente, días cobro, días inventario
    par = {"Comercio": (0.30, 2.2, 0.70, 35, 60), "Industria": (0.32, 1.1, 0.50, 60, 90),
           "Servicios": (0.45, 1.3, 0.45, 50, 10), "Agro": (0.28, 0.6, 0.35, 70, 120),
           "Construcción": (0.25, 0.9, 0.55, 80, 45)}
    filas = []
    for i in range(n):
        sec = rng.choice(SECTORES, p=[.28, .24, .22, .14, .12])
        mb, rot, pac, dc, di = par[sec]
        activo = float(np.exp(rng.normal(np.log(800), 0.9)))           # millones de $
        rot_i = max(0.2, rng.normal(rot, rot * 0.25))
        ventas = activo * rot_i
        mb_i = np.clip(rng.normal(mb, 0.07), 0.05, 0.75)
        costo = ventas * (1 - mb_i)
        gop = ventas * np.clip(rng.normal(mb - 0.10, 0.07), 0.02, 0.7)  # algunos tienen pérdida operativa
        creditos = ventas * max(5, rng.normal(dc, dc * 0.35)) / 365
        bc = costo * max(0, rng.normal(di, di * 0.35)) / 365
        ac_obj = activo * np.clip(rng.normal(pac, 0.1), 0.15, 0.9)
        caja = max(activo * 0.01, ac_obj - creditos - bc)
        anc = max(activo * 0.05, activo - caja - creditos - bc)
        at = caja + creditos + bc + anc
        apal = np.clip(rng.beta(2.6, 2.4), 0.08, 0.93)                    # pasivo / activo
        pt = at * apal
        pc = pt * np.clip(rng.normal(0.6, 0.15), 0.2, 0.95)
        pnc = pt - pc
        intereses = pt * np.clip(rng.normal(0.12, 0.04), 0.02, 0.3)
        ebit = ventas - costo - gop
        imp = max(0, (ebit - intereses) * 0.30)
        ra = (at - pt) * rng.normal(0.25, 0.35)
        b = dict(caja=caja, creditos=creditos, bienes_cambio=bc, activo_no_corriente=anc,
                 pasivo_corriente=pc, pasivo_no_corriente=pnc, resultados_acumulados=ra,
                 ventas=ventas, costo_ventas=costo, gastos_operativos=gop, intereses=intereses, impuestos=imp)
        filas.append({"empresa": f"PyME_{i+1:03d}", "sector": sec, **{k: round(v, 2) for k, v in b.items()}})
    df = pd.DataFrame(filas)
    # Proceso "verdadero" (desconocido para los modelos) que genera la dificultad financiera
    R = df.apply(lambda f: pd.Series(calcular_ratios(f.to_dict())), axis=1)
    lin = (-1.2 - 1.4 * np.log(R["liquidez_corriente"]) + 0.9 * np.log1p(R["endeudamiento"].fillna(20))
           - 9.0 * R["roa"] - 0.08 * np.clip(R["cobertura_intereses"], -10, 10)
           + 0.006 * (R["dias_cobro"] - 60) + rng.normal(0, 0.6, len(df)))
    df["dificultad"] = (rng.uniform(size=len(df)) < 1 / (1 + np.exp(-lin))).astype(int)
    return df


def tabla_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica calcular_ratios + Z de Altman a cada fila de un DataFrame de balances."""
    R = df.apply(lambda f: pd.Series(calcular_ratios(f.to_dict())), axis=1)
    R["z_altman"] = df.apply(lambda f: z_altman(f.to_dict()), axis=1)
    return R


# ------------------------------------------------------------- diagnóstico
def semaforo(valor, ref, mayor_mejor=True, tol=0.15):
    """Compara un ratio con la mediana del sector. Devuelve 🟢, 🟡 o 🔴."""
    if pd.isna(valor):
        return "🔴"
    if ref == 0 or pd.isna(ref):
        return "🟡"
    rel = (valor - ref) / abs(ref)
    if not mayor_mejor:
        rel = -rel
    return "🟢" if rel >= -tol else ("🟡" if rel >= -0.40 else "🔴")


def fmt(k, v):
    if pd.isna(v):
        return "n/d"
    if k in PORCENTUALES:
        return f"{100*v:.1f}%"
    if k.startswith("dias"):
        return f"{v:.0f} días"
    return f"{v:.2f}"


def evaluar(b: dict, ref_sector: pd.Series) -> pd.DataFrame:
    """Tabla de ratios de la empresa vs. mediana del sector, con semáforo."""
    r = calcular_ratios(b)
    filas = []
    for k, (nombre, mm, cat) in RATIOS.items():
        filas.append({"Categoría": cat, "Ratio": nombre, "Empresa": fmt(k, r[k]),
                      "Mediana sector": fmt(k, ref_sector.get(k, np.nan)),
                      "Estado": semaforo(r[k], ref_sector.get(k, np.nan), mm), "clave": k, "valor": r[k]})
    return pd.DataFrame(filas)


def diagnostico_texto(nombre, sector, b, tabla, prob, z):
    """Informe ejecutivo por reglas (se usa si no hay un LLM disponible)."""
    bb = completar_balance(b)
    r = dict(zip(tabla["clave"], tabla["valor"]))
    rojos = tabla.loc[tabla["Estado"] == "🔴", "Ratio"].tolist()
    verdes = tabla.loc[tabla["Estado"] == "🟢", "Ratio"].tolist()
    riesgo = "ALTO" if prob >= 0.4 else ("MODERADO" if prob >= 0.2 else "BAJO")
    t = [f"DIAGNÓSTICO FINANCIERO: {nombre} ({sector}).",
         f"Riesgo de dificultad financiera: {riesgo} (probabilidad estimada {100*prob:.0f}%; Z'' de Altman {z:.2f}, zona {zona_altman(z).lower()})."]
    if r["liquidez_corriente"] < 1:
        t.append(f"Liquidez: el activo corriente no alcanza a cubrir las deudas de corto plazo (liquidez corriente {r['liquidez_corriente']:.2f}); "
                 f"hay un capital de trabajo negativo de $ {-bb['capital_trabajo']:,.0f} M.")
    elif r["liquidez_corriente"] < 1.5:
        t.append(f"Liquidez: ajustada (liquidez corriente {r['liquidez_corriente']:.2f}); sin vender inventario cubre {100*r['prueba_acida']:.0f}% de las deudas de corto plazo (prueba ácida {r['prueba_acida']:.2f}).")
    elif r["prueba_acida"] < 0.8:
        t.append(f"Liquidez: holgada en total (liquidez corriente {r['liquidez_corriente']:.2f}), pero depende del inventario: sin venderlo cubre solo {100*r['prueba_acida']:.0f}% de las deudas de corto plazo (prueba ácida {r['prueba_acida']:.2f}).")
    else:
        t.append(f"Liquidez: holgada (liquidez corriente {r['liquidez_corriente']:.2f}, prueba ácida {r['prueba_acida']:.2f}).")
    if pd.isna(r["endeudamiento"]):
        t.append("Solvencia: el patrimonio neto es negativo; la empresa está técnicamente quebrada.")
    elif r["endeudamiento"] > 2:
        t.append(f"Solvencia: endeudamiento elevado ({r['endeudamiento']:.2f} pesos de deuda por peso propio).")
    else:
        t.append(f"Solvencia: endeudamiento razonable ({r['endeudamiento']:.2f} veces el patrimonio).")
    if r["cobertura_intereses"] < 1.5:
        t.append(f"El resultado operativo cubre solo {r['cobertura_intereses']:.1f} veces los intereses: el costo financiero compromete la rentabilidad.")
    t.append(f"Rentabilidad: margen neto {100*r['margen_neto']:.1f}%, ROA {100*r['roa']:.1f}%" +
             (f", ROE {100*r['roe']:.1f}%." if not pd.isna(r['roe']) else "."))
    if verdes:
        t.append("Fortalezas frente al sector: " + ", ".join(verdes[:4]) + ".")
    if rojos:
        t.append("Alertas frente al sector: " + ", ".join(rojos[:4]) + ".")
    rec = []
    if r["liquidez_corriente"] < 1.5: rec.append("refinanciar deuda de corto a largo plazo")
    if r["dias_cobro"] > 60: rec.append("acortar plazos de cobro a clientes")
    if r["dias_inventario"] > 90: rec.append("reducir inventario inmovilizado")
    if r["margen_operativo"] < 0.05: rec.append("revisar estructura de costos y precios")
    if not pd.isna(r["endeudamiento"]) and r["endeudamiento"] > 2: rec.append("capitalizar utilidades o reducir pasivos")
    if rec:
        t.append("Recomendaciones: " + "; ".join(rec) + ".")
    return " ".join(t)


def diagnostico_llm(contexto: dict, modelo="claude-sonnet-4-5"):
    """Redacción del diagnóstico con un LLM (requiere ANTHROPIC_API_KEY)."""
    import json, anthropic
    cli = anthropic.Anthropic()
    msg = cli.messages.create(model=modelo, max_tokens=600, messages=[{"role": "user", "content":
          "Sos un analista financiero argentino. Con estos datos, redactá un diagnóstico de 150 palabras para el "
          "dueño de la PyME, en lenguaje claro, con 3 recomendaciones concretas. No inventes cifras: "
          + json.dumps(contexto, ensure_ascii=False, default=str)}])
    return msg.content[0].text
