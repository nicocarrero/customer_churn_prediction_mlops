# -*- coding: utf-8 -*-
"""
app.py  —  Churn Intelligence Dashboard
Correr con: streamlit run app.py
API en:     http://localhost:8000
"""

import datetime
import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://api:8000")

st.set_page_config(
    page_title="Churn Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

#  ESTILOS CSS
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0a0a0a !important;
    color: #e2e8f0 !important;
}

.stApp { background: #0a0a0a !important; }
.block-container { padding: 0 2rem 4rem; max-width: 1280px; }
#MainMenu, footer, header { visibility: hidden; }

/* ── Header bar ── */
.top-bar {
    background: #111111; border-bottom: 1px solid #222222;
    padding: 1rem 2rem; margin: 0 -2rem 2.5rem;
    display: flex; align-items: center; justify-content: space-between;
}
.top-bar-left { display: flex; align-items: center; gap: 12px; }
.top-icon {
    width: 38px; height: 38px; background: #6366f1;
    border-radius: 9px; display: flex; align-items: center;
    justify-content: center; font-size: 1.1rem; flex-shrink: 0;
}
.top-brand { font-size: 1rem; font-weight: 600; color: #f1f5f9; letter-spacing: -0.02em; line-height: 1; }
.top-sub   { font-family: 'JetBrains Mono', monospace; font-size: 0.58rem; color: #475569; letter-spacing: 0.06em; margin-top: 2px; }
.api-pill  { display: flex; align-items: center; gap: 6px; font-family: 'JetBrains Mono', monospace; font-size: 0.65rem; color: #64748b; }
.api-dot   { width: 7px; height: 7px; border-radius: 50%; flex-shrink: 0; }
.dot-ok    { background: #22c55e; box-shadow: 0 0 8px #22c55e55; }
.dot-err   { background: #ef4444; box-shadow: 0 0 8px #ef444455; }
.dot-warn  { background: #f59e0b; box-shadow: 0 0 8px #f59e0b55; }

/* ── Section label ── */
.sec-lbl {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.6rem; letter-spacing: 0.2em; text-transform: uppercase;
    color: #334155; display: flex; align-items: center; gap: 10px; margin-bottom: 1.25rem;
}
.sec-lbl::after { content: ''; flex: 1; height: 1px; background: #1e1e1e; }

/* ── Panel tag ── */
.panel-tag {
    font-family: 'JetBrains Mono', monospace; font-size: 0.55rem; letter-spacing: 0.14em;
    text-transform: uppercase; color: #a5b4fc; border-left: 2px solid #6366f1;
    padding-left: 8px; margin-bottom: 1rem; display: block;
}

/* ── Streamlit widget overrides ── */
div[data-testid="stNumberInput"] label, div[data-testid="stSelectbox"] label, div[data-testid="stToggle"] label {
    font-family: 'JetBrains Mono', monospace !important; font-size: 0.65rem !important;
    color: #475569 !important; letter-spacing: 0.06em !important;
    text-transform: uppercase !important; font-weight: 400 !important;
}

div[data-testid="stNumberInput"] input {
    background: #0f0f0f !important; border: 1px solid #1e1e1e !important;
    border-radius: 8px !important; color: #e2e8f0 !important;
    font-family: 'JetBrains Mono', monospace !important; font-size: 0.9rem !important; padding: 0.5rem 0.75rem !important;
}
div[data-testid="stNumberInput"] input:focus { border-color: #6366f1 !important; box-shadow: 0 0 0 2px #6366f120 !important; }
div[data-testid="stNumberInput"] button { background: #1a1a1a !important; border-color: #1e1e1e !important; color: #64748b !important; }

div[data-baseweb="select"] > div { background: #0f0f0f !important; border: 1px solid #1e1e1e !important; border-radius: 8px !important; }
div[data-baseweb="select"] span, div[data-baseweb="select"] div { color: #e2e8f0 !important; font-family: 'JetBrains Mono', monospace !important; font-size: 0.82rem !important; background: transparent !important; }
div[data-baseweb="popover"], div[data-baseweb="menu"] { background: #111111 !important; border: 1px solid #1e1e1e !important; }
li[role="option"] { color: #e2e8f0 !important; }
li[role="option"]:hover { background: #1a1a2e !important; }

div[data-testid="stToggle"] span[data-checked="true"] { background: #6366f1 !important; }

.div-line { border: none; border-top: 1px solid #1a1a1a; margin: 1.5rem 0; }

/* ── Predict button (Form + Standard) ── */
div[data-testid="stButton"] > button, div[data-testid="stFormSubmitButton"] > button {
    background: #6366f1 !important; color: #ffffff !important; border: none !important;
    border-radius: 10px !important; padding: 0.75rem 2rem !important;
    font-family: 'Inter', sans-serif !important; font-size: 0.9rem !important;
    font-weight: 600 !important; width: 100% !important; transition: all 0.15s !important; letter-spacing: -0.01em !important;
}
div[data-testid="stButton"] > button:hover, div[data-testid="stFormSubmitButton"] > button:hover {
    background: #4f46e5 !important; transform: translateY(-1px) !important;
}

/* ── Result card ── */
.res-frame { border-radius: 16px; border: 1px solid #1e1e1e; background: #111111; overflow: hidden; margin-top: 1.5rem; }
.res-topbar { background: #0d0d0d; border-bottom: 1px solid #1e1e1e; padding: 0.75rem 1.5rem; display: flex; justify-content: space-between; align-items: center; font-family: 'JetBrains Mono', monospace; font-size: 0.6rem; color: #334155; letter-spacing: 0.1em; text-transform: uppercase; }
.res-body { padding: 1.75rem 1.5rem; }
.res-verdict-row { display: flex; align-items: flex-start; justify-content: space-between; gap: 1rem; flex-wrap: wrap; margin-bottom: 1.5rem; }
.res-verdict { font-size: 2.2rem; font-weight: 700; letter-spacing: -0.05em; line-height: 1; }
.verdict-churn  { color: #f87171; }
.verdict-retain { color: #4ade80; }
.res-caption { font-size: 0.78rem; color: #475569; margin-top: 0.35rem; }
.risk-pill { font-family: 'JetBrains Mono', monospace; font-size: 0.6rem; letter-spacing: 0.1em; text-transform: uppercase; padding: 0.3rem 0.85rem; border-radius: 999px; flex-shrink: 0; margin-top: 4px; }
.pill-alto  { background: #2d0e0e; color: #f87171; border: 1px solid #4d1a1a; }
.pill-medio { background: #2d1f0a; color: #fbbf24; border: 1px solid #4d380f; }
.pill-bajo  { background: #0a2d14; color: #4ade80; border: 1px solid #0f4d20; }

.metric-row { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin-bottom: 1.5rem; }
.m-tile { background: #0a0a0a; border: 1px solid #1e1e1e; border-radius: 12px; padding: 1rem 1.1rem; }
.m-lbl { font-family: 'JetBrains Mono', monospace; font-size: 0.55rem; letter-spacing: 0.14em; text-transform: uppercase; color: #334155; margin-bottom: 0.5rem; }
.m-val { font-size: 1.9rem; font-weight: 700; letter-spacing: -0.05em; line-height: 1; }
.mv-churn  { color: #f87171; }
.mv-retain { color: #4ade80; }
.mv-white  { color: #e2e8f0; }
.mv-sm     { font-size: 1.1rem; padding-top: 0.3rem; }

.bar-lbl { font-family: 'JetBrains Mono', monospace; font-size: 0.55rem; letter-spacing: 0.14em; text-transform: uppercase; color: #334155; margin-bottom: 0.5rem; }
.bar-track { background: #1a1a1a; border-radius: 999px; height: 8px; overflow: hidden; margin-bottom: 0.35rem; }
.bar-fill-alto  { background: #ef4444; height: 8px; border-radius: 999px; }
.bar-fill-medio { background: #f59e0b; height: 8px; border-radius: 999px; }
.bar-fill-bajo  { background: #22c55e; height: 8px; border-radius: 999px; }
.bar-ticks { display: flex; justify-content: space-between; font-family: 'JetBrains Mono', monospace; font-size: 0.55rem; color: #1e293b; }

.rec-box { margin-top: 1.5rem; background: #0d0d0d; border: 1px solid #1e1e1e; border-radius: 12px; padding: 1.1rem 1.25rem; display: flex; align-items: flex-start; gap: 12px; }
.rec-icon { font-size: 1.2rem; flex-shrink: 0; margin-top: 2px; }
.rec-title { font-size: 0.82rem; font-weight: 600; color: #e2e8f0; margin-bottom: 0.25rem; }
.rec-text  { font-size: 0.75rem; color: #475569; line-height: 1.6; }

.err-box { background: #1a0a0a; border: 1px solid #4d1a1a; border-radius: 10px; padding: 1rem 1.25rem; font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; color: #f87171; margin-top: 1rem; }
</style>
""",
    unsafe_allow_html=True,
)


# ═══════════════════════════════════════════════════════════
#  API CHECK CACHEADO
# ═══════════════════════════════════════════════════════════
@st.cache_data(ttl=60, show_spinner=False)
def check_api_health():
    try:
        health = requests.get(f"{API_URL}/health", timeout=3).json()
        return True, health.get("model_loaded", False)
    except Exception:
        return False, False


api_ok, model_ok = check_api_health()

dot_cls = "dot-ok" if (api_ok and model_ok) else ("dot-warn" if api_ok else "dot-err")
api_txt = (
    "API conectada · modelo cargado"
    if (api_ok and model_ok)
    else (
        "La API responde pero sin modelo" if api_ok else "Sin conexión · localhost:8000"
    )
)

st.markdown(
    f"""
<div class="top-bar">
    <div class="top-bar-left">
        <div class="top-icon">📊</div>
        <div>
            <div class="top-brand">Churn Intelligence</div>
            <div class="top-sub">retention analytics platform</div>
        </div>
    </div>
    <div class="api-pill">
        <div class="api-dot {dot_cls}"></div>
        <span>{api_txt}</span>
    </div>
</div>
""",
    unsafe_allow_html=True,
)

if not api_ok:
    st.error(
        "⚠️ No se puede conectar con la API. Verificá que el contenedor de FastAPI esté corriendo y que la URL sea correcta."
    )
    st.stop()
if not model_ok:
    st.warning(
        "⚠️ La API responde pero el modelo no está cargado. Verificá la ruta del archivo del modelo."
    )
    st.stop()


# ═══════════════════════════════════════════════════════════
#  FORMULARIO ENCAPSULADO
# ═══════════════════════════════════════════════════════════
st.markdown('<div class="sec-lbl">Perfil del cliente</div>', unsafe_allow_html=True)

# st.form envuelve todos los inputs, evitando recargas hasta apretar el botón
with st.form("churn_input_form", border=False):
    col1, col2, col3 = st.columns(3, gap="large")

    with col1:
        st.markdown(
            '<span class="panel-tag">Facturación</span>', unsafe_allow_html=True
        )
        tenure_months = st.number_input(
            "Meses con el servicio (tenure_months)",
            min_value=0,
            max_value=72,
            value=12,
            step=1,
            help="Antigüedad del cliente en meses.",
        )
        monthly_charge = st.number_input(
            "Cargo mensual en USD (monthly_charge)",
            min_value=15.0,
            max_value=130.0,
            value=65.5,
            step=0.5,
        )
        total_charges = st.number_input(
            "Cargos totales acumulados (total_charges)",
            min_value=0.0,
            value=786.0,
            step=10.0,
            help="Suma de lo facturado. Debe ser coherente con los meses de antigüedad.",
        )
        avg_monthly_usage_gb = st.number_input(
            "Uso mensual promedio en GB (avg_monthly_usage_gb)",
            min_value=0.0,
            value=95.3,
            step=1.0,
        )

    with col2:
        st.markdown(
            '<span class="panel-tag">Comportamiento</span>', unsafe_allow_html=True
        )
        support_tickets = st.number_input(
            "Tickets de soporte (support_tickets)", min_value=0, max_value=20, value=1
        )
        late_payments = st.number_input(
            "Pagos tardíos (late_payments)",
            min_value=0,
            max_value=12,
            value=0,
            help="Atrasos en los últimos 12 meses.",
        )
        num_products = st.number_input(
            "Productos contratados (num_products)", min_value=1, max_value=4, value=2
        )
        customer_age = st.number_input(
            "Edad del cliente (customer_age)", min_value=18, max_value=99, value=35
        )

    with col3:
        st.markdown('<span class="panel-tag">Contrato</span>', unsafe_allow_html=True)
        contract_type = st.selectbox(
            "Tipo de contrato (contract_type)", ["mensual", "anual", "bianual"]
        )
        payment_method = st.selectbox(
            "Método de pago (payment_method)",
            ["transferencia", "debito", "efectivo", "credito"],
        )
        internet_service = st.selectbox(
            "Servicio de internet (internet_service)",
            ["cable", "fibra", "movil", "ninguno"],
        )
        region = st.selectbox("Región (region)", ["centro", "norte", "oeste", "sur"])

        st.markdown("<br>", unsafe_allow_html=True)
        t1, t2, t3 = st.columns(3)
        with t1:
            has_streaming = st.toggle("Streaming (has_streaming)", value=False)
        with t2:
            has_security_pack = st.toggle("Seguridad (has_security_pack)", value=False)
        with t3:
            is_promo = st.toggle("Promo (is_promo)", value=False)

    st.markdown('<hr class="div-line">', unsafe_allow_html=True)
    _, btn_col, _ = st.columns([3, 2, 3])
    with btn_col:
        # st.form_submit_button hace de gatillo único
        predict_btn = st.form_submit_button(
            "Analizar cliente →", use_container_width=True
        )

# ═══════════════════════════════════════════════════════════
#  MANEJO DE ERRORES Y RESULTADO
# ═══════════════════════════════════════════════════════════
if predict_btn:
    # 1. Validación de usuario simple antes de consultar API
    validation_errors = []
    if total_charges < monthly_charge and tenure_months > 0:
        validation_errors.append(
            "Los cargos totales no pueden ser menores al cargo mensual actual si el cliente tiene más de 0 meses."
        )
    if tenure_months == 0 and total_charges > 0:
        validation_errors.append(
            "Un cliente nuevo (0 meses) no debería tener cargos acumulados."
        )

    if validation_errors:
        for err in validation_errors:
            st.warning(f"⚠️ **Error de validación:** {err}")

    # 2. Si no hay errores, se dispara la solicitud
    else:
        payload = {
            "tenure_months": tenure_months,
            "monthly_charge": monthly_charge,
            "total_charges": total_charges,
            "support_tickets": support_tickets,
            "late_payments": late_payments,
            "avg_monthly_usage_gb": avg_monthly_usage_gb,
            "contract_type": contract_type,
            "payment_method": payment_method,
            "internet_service": internet_service,
            "has_streaming": int(has_streaming),
            "has_security_pack": int(has_security_pack),
            "num_products": num_products,
            "region": region,
            "customer_age": customer_age,
            "is_promo": int(is_promo),
        }

        try:
            with st.spinner("Consultando modelo predictivo…"):
                response = requests.post(f"{API_URL}/predict", json=payload, timeout=10)

            response.raise_for_status()
            result = response.json()

            pred = result.get("churn_prediction", 0)
            prob = result.get("churn_probability", 0.0)
            risk = result.get("risk_level", "bajo")
            pct = round(prob * 100, 1)

            is_churn = pred == 1
            verdict_lbl = "CHURN DETECTADO" if is_churn else "CLIENTE ESTABLE"
            verdict_cls = "verdict-churn" if is_churn else "verdict-retain"
            mv_cls = "mv-churn" if is_churn else "mv-retain"
            pred_txt = "Abandono" if is_churn else "Retención"
            caption = (
                "Alta probabilidad de abandono. Se recomienda acción inmediata."
                if is_churn
                else "Baja probabilidad de abandono. Perfil de retención saludable."
            )

            risk_label = {
                "alto": "Riesgo alto",
                "medio": "Riesgo medio",
                "bajo": "Riesgo bajo",
            }.get(risk, risk)

            recs = {
                "alto": (
                    "🔴",
                    "Acción urgente requerida",
                    "Señales críticas de abandono. Contacto proactivo inmediato, "
                    "oferta de retención personalizada y revisión de tickets pendientes.",
                ),
                "medio": (
                    "🟡",
                    "Seguimiento activo recomendado",
                    "Riesgo moderado. Evaluar descuento en próxima factura, upgrade de plan "
                    "o revisión de soporte en los últimos 30 días.",
                ),
                "bajo": (
                    "🟢",
                    "Cliente en zona segura",
                    "Perfil estable. Mantener comunicación periódica y aprovechar "
                    "para ofrecer productos complementarios.",
                ),
            }
            icon_r, title_r, text_r = recs.get(risk, recs["bajo"])
            now_str = datetime.datetime.now().strftime("%d/%m/%Y  %H:%M")

            # Muestra del panel de resultados
            st.markdown(
                f"""
            <div class="res-frame">
              <div class="res-topbar">
                <span>◈ resultado del análisis</span>
                <span>{now_str}</span>
              </div>
              <div class="res-body">
                <div class="res-verdict-row">
                  <div>
                    <div class="res-verdict {verdict_cls}">{verdict_lbl}</div>
                    <div class="res-caption">{caption}</div>
                  </div>
                  <span class="risk-pill pill-{risk}">{risk_label}</span>
                </div>
                <div class="metric-row">
                  <div class="m-tile">
                    <div class="m-lbl">Probabilidad de churn</div>
                    <div class="m-val {mv_cls}">{pct}%</div>
                  </div>
                  <div class="m-tile">
                    <div class="m-lbl">Predicción del modelo</div>
                    <div class="m-val mv-white">{pred_txt}</div>
                  </div>
                  <div class="m-tile">
                    <div class="m-lbl">Nivel de riesgo</div>
                    <div class="m-val mv-white mv-sm">{risk_label}</div>
                  </div>
                </div>
                <div class="bar-lbl">Índice de probabilidad</div>
                <div class="bar-track">
                  <div class="bar-fill-{risk}" style="width:{pct}%"></div>
                </div>
                <div class="bar-ticks">
                  <span>0%</span><span style="color:#475569">{pct}%</span><span>100%</span>
                </div>
                <div class="rec-box">
                  <div class="rec-icon">{icon_r}</div>
                  <div>
                    <div class="rec-title">{title_r}</div>
                    <div class="rec-text">{text_r}</div>
                  </div>
                </div>
              </div>
            </div>
            """,
                unsafe_allow_html=True,
            )

        except requests.exceptions.HTTPError as he:
            st.error(
                f"🛑 **Error en la API (HTTP {response.status_code}):** {response.text}"
            )
        except requests.exceptions.ConnectionError:
            st.error(
                "🛑 **Error de conexión:** No se pudo establecer conexión con la API en el puerto 8000."
            )
        except Exception as e:
            st.error(f"🛑 **Error inesperado:** {str(e)}")
