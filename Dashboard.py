import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# ---------------- LOGO (Base64 Embedded) ---------------- #
base64_logo = """
iVBORw0KGgoAAAANSUhEUgAAA...your_base64_string_here...==
"""  # Replace with full base64 from your image

def render_logo(base64_str, width=120):
    return f"<img src='data:image/png;base64,{base64_str}' width='{width}'>"

# ---------------- PAGE CONFIG ---------------- #
st.set_page_config(
    page_title="Sistema Integral NOM-035 & LEAN 2.0",
    layout="wide",
    page_icon="📊"
)

# ---------------- STYLES ---------------- #
st.markdown("""
<style>
    .main { background-color: #f9f9f9; }
    .kpi-card { border-radius: 10px; padding: 1.5rem; box-shadow: 0 4px 6px rgba(0,0,0,0.1); transition: transform 0.3s; }
    .kpi-card:hover { transform: translateY(-5px); }
    .kpi-good { background: linear-gradient(135deg, #4CAF50 0%, #81C784 100%); color: white; }
    .kpi-warning { background: linear-gradient(135deg, #FFC107 0%, #FFD54F 100%); color: #333; }
    .kpi-danger { background: linear-gradient(135deg, #F44336 0%, #E57373 100%); color: white; }
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ---------------- #
col1, col2 = st.columns([1, 5])
with col1:
    st.markdown(render_logo(base64_logo), unsafe_allow_html=True)
with col2:
    st.markdown("<h1 style='margin-bottom:0; color: #2d3e50;'>📈 Sistema Integral NOM-035 & LEAN 2.0</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #4a6fa5; margin-top:0;'>Monitoreo Estratégico de Bienestar Psicosocial y Eficiencia Operacional</p>", unsafe_allow_html=True)

st.markdown(f"<div style='text-align: right; color: #666; font-size:0.8rem;'>Última actualización: {datetime.now().strftime('%d/%m/%Y %H:%M')}</div>", unsafe_allow_html=True)
st.markdown("---")

# ---------------- SIDEBAR ---------------- #
with st.sidebar:
    st.markdown("### 🔍 Filtros Avanzados")
    
    with st.expander("📅 Rango Temporal", expanded=True):
        fecha_inicio = st.date_input("Fecha inicio", value=datetime(2025, 1, 1))
        fecha_fin = st.date_input("Fecha fin", value=datetime(2025, 4, 1))
    
    with st.expander("🏢 Unidades Organizacionales", expanded=True):
        departamentos = st.multiselect(
            "Seleccionar departamentos",
            ['Producción', 'Calidad', 'Logística', 'Administración', 'Ventas', 'RH', 'TI'],
            default=['Producción', 'Calidad', 'Logística']
        )
    
    with st.expander("📊 Métricas Clave"):
        metricas = st.multiselect(
            "Indicadores a visualizar",
            ['NOM-035', 'Calidad', 'Productividad', 'Bienestar', 'LEAN'],
            default=['NOM-035', 'Calidad']
        )

# ---------------- DATA ---------------- #
@st.cache_data
def cargar_datos():
    nom_data = pd.DataFrame({
        'Departamento': ['Producción', 'Calidad', 'Logística', 'Administración', 'Ventas', 'RH', 'TI'],
        'Evaluaciones': [92, 95, 70, 88, 85, 97, 90],
        'Capacitaciones': [85, 90, 65, 82, 78, 95, 88],
        'Incidentes': [3, 1, 8, 2, 4, 0, 1],
        'Tendencia': [1.2, 0.8, -2.5, 0.5, -1.2, 0.3, 0.7]
    })

    lean_data = pd.DataFrame({
        'Departamento': ['Producción', 'Calidad', 'Logística', 'Administración', 'Ventas', 'RH', 'TI'],
        'Eficiencia': [82, 88, 65, 75, 78, 85, 90],
        'Reducción Desperdicio': [15, 20, 5, 12, 8, 18, 25],
        'Proyectos Activos': [3, 2, 1, 4, 2, 3, 5]
    })

    bienestar_data = pd.DataFrame({
        'Mes': pd.date_range(start='2024-01-01', periods=12, freq='M'),
        'Índice Bienestar': np.round(np.random.normal(75, 5, 12), 1),
        'Ausentismo': np.round(np.random.normal(8, 2, 12), 1),
        'Rotación': np.round(np.random.normal(12, 3, 12), 1)
    })

    return nom_data, lean_data, bienestar_data

nom_data, lean_data, bienestar_data = cargar_datos()

# ---------------- KPI ---------------- #
st.markdown("## 📊 Tablero de Control Integral")

def crear_kpi(valor, titulo, meta=90, icono="📊"):
    delta = valor - meta
    status = "good" if valor >= meta else "warning" if valor >= meta - 15 else "danger"
    card = f"""
    <div class="kpi-card kpi-{status}">
        <h3>{icono} {titulo}</h3>
        <span style="font-size:1.5rem; font-weight:bold;">{valor}% ({delta:+})</span>
        <div style="margin-top:10px; height:6px; background:rgba(255,255,255,0.3); border-radius:3px;">
            <div style="width:{valor}%; height:6px; background:white; border-radius:3px;"></div>
        </div>
        <p style="font-size:0.8rem;">Meta: {meta}%</p>
    </div>
    """
    return card

kpis = [
    (92, "Cumplimiento NOM-035", 90, "🧠"),
    (85, "Adopción LEAN 2.0", 80, "🔄"),
    (78, "Índice Bienestar", 85, "❤️"),
    (65, "Eficiencia Operativa", 75, "⚙️")
]

cols = st.columns(len(kpis))
for i, kpi in enumerate(kpis):
    with cols[i]:
        st.markdown(crear_kpi(*kpi), unsafe_allow_html=True)

# ---------------- TABS ---------------- #
tab1, tab2, tab3 = st.tabs(["NOM-035", "LEAN 2.0", "Bienestar"])

with tab1:
    st.markdown("### 🧠 Cumplimiento NOM-035")
    df_nom = nom_data[nom_data['Departamento'].isin(departamentos)]
    fig_nom = px.bar(df_nom, x='Departamento', y=['Evaluaciones', 'Capacitaciones'], barmode='group')
    st.plotly_chart(fig_nom, use_container_width=True)

with tab2:
    st.markdown("### 🔄 Progreso LEAN 2.0")
    df_lean = lean_data[lean_data['Departamento'].isin(departamentos)]
    fig_lean = px.bar(df_lean, x='Departamento', y='Eficiencia', color='Eficiencia', color_continuous_scale='Greens')
    st.plotly_chart(fig_lean, use_container_width=True)

with tab3:
    st.markdown("### ❤️ Tendencias de Bienestar")
    fig_bienestar = px.line(bienestar_data, x='Mes', y=['Índice Bienestar', 'Ausentismo', 'Rotación'], markers=True)
    st.plotly_chart(fig_bienestar, use_container_width=True)

# --- PIE ---
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#666; font-size:0.9rem;">
    <p>© 2025 Sistema NOM-035 & LEAN 2.0</p>
    <p>📞 Soporte: (663) 558 2452 | 📧 contacto@lean2institute.org</p>
</div>
""", unsafe_allow_html=True)
