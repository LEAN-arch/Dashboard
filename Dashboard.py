import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import numpy as np

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(
    page_title="Sistema Integral NOM-035 & LEAN 2.0",
    layout="wide",
    page_icon="📊"
)

# --- ESTILOS PERSONALIZADOS ---
st.markdown("""
<style>
    .main { background-color: #f9f9f9; }
    .kpi-card { border-radius: 10px; padding: 1.5rem; box-shadow: 0 4px 6px rgba(0,0,0,0.1); transition: transform 0.3s; }
    .kpi-card:hover { transform: translateY(-5px); }
    .kpi-good { background: linear-gradient(135deg, #4CAF50, #81C784); color: white; }
    .kpi-warning { background: linear-gradient(135deg, #FFC107, #FFD54F); color: #333; }
    .kpi-danger { background: linear-gradient(135deg, #F44336, #E57373); color: white; }
</style>
""", unsafe_allow_html=True)

# --- FUNCIONES AUXILIARES ---
@st.cache_data

def cargar_datos():
    departamentos = ['Producción', 'Calidad', 'Logística', 'Administración', 'Ventas', 'RH', 'TI']
    nom = pd.DataFrame({
        'Departamento': departamentos,
        'Evaluaciones': np.random.randint(65, 100, len(departamentos)),
        'Capacitaciones': np.random.randint(60, 100, len(departamentos)),
        'Incidentes': np.random.randint(0, 10, len(departamentos)),
        'Tendencia': np.random.uniform(-3, 2, len(departamentos)).round(1)
    })
    lean = pd.DataFrame({
        'Departamento': departamentos,
        'Eficiencia': np.random.randint(60, 95, len(departamentos)),
        'Reducción Desperdicio': np.random.randint(5, 30, len(departamentos)),
        'Proyectos Activos': np.random.randint(1, 6, len(departamentos))
    })
    bienestar = pd.DataFrame({
        'Mes': pd.date_range(start='2024-01-01', periods=12, freq='M'),
        'Índice Bienestar': np.round(np.random.normal(75, 5, 12), 1),
        'Ausentismo': np.round(np.random.normal(8, 2, 12), 1),
        'Rotación': np.round(np.random.normal(12, 3, 12), 1)
    })
    return nom, lean, bienestar

def crear_kpi(valor, titulo, meta=90, icono="📊"):
    delta = valor - meta
    status = "good" if valor >= meta else "warning" if valor >= meta - 15 else "danger"
    return f"""
    <div class="kpi-card kpi-{status}">
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <h3 style="margin:0;">{icono} {titulo}</h3>
            <span style="font-size:1.2rem; font-weight:bold;">{valor}% ({delta:+})</span>
        </div>
        <div style="margin-top:10px; height:6px; background:rgba(255,255,255,0.3); border-radius:3px;">
            <div style="width:{valor}%; height:6px; background:white; border-radius:3px;"></div>
        </div>
        <p style="margin:5px 0 0 0; font-size:0.8rem;">Meta: {meta}%</p>
    </div>
    """

# --- INTERFAZ ---
col1, col2 = st.columns([1, 5])
with col1:
    st.image("logo_empresa.png", width=100)
with col2:
    st.markdown("<h1 style='margin-bottom:0;'>📈 Sistema Integral NOM-035 & LEAN 2.0</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #4a6fa5;'>Monitoreo de Bienestar Psicosocial y Eficiencia Operacional</p>", unsafe_allow_html=True)
st.markdown(f"<div style='text-align: right; color: #666; font-size:0.8rem;'>Última actualización: {datetime.now().strftime('%d/%m/%Y %H:%M')}</div>", unsafe_allow_html=True)

# --- FILTROS ---
with st.sidebar:
    st.header("🔍 Filtros")
    fecha_inicio = st.date_input("Fecha inicio", value=datetime(2025, 1, 1))
    fecha_fin = st.date_input("Fecha fin", value=datetime(2025, 4, 1))
    if fecha_inicio > fecha_fin:
        st.error("La fecha de inicio debe ser anterior a la fecha de fin")

    departamentos = st.multiselect("Departamentos", ['Producción', 'Calidad', 'Logística', 'Administración', 'Ventas', 'RH', 'TI'], default=['Producción', 'Calidad'])
    st.button("🔄 Actualizar", on_click=lambda: st.experimental_rerun())

# --- CARGA DE DATOS ---
nom_data, lean_data, bienestar_data = cargar_datos()

# --- KPI PRINCIPALES ---
st.markdown("## 📊 KPIs Principales")
cols = st.columns(4)
kpis = [
    (92, "Cumplimiento NOM-035", 90, "🧠"),
    (85, "Adopción LEAN 2.0", 80, "🔄"),
    (78, "Índice Bienestar", 85, "❤️"),
    (65, "Eficiencia Operativa", 75, "⚙️")
]
for i, kpi in enumerate(kpis):
    with cols[i]:
        st.markdown(crear_kpi(*kpi), unsafe_allow_html=True)

# --- TABS ---
tab1, tab2, tab3 = st.tabs(["NOM-035", "LEAN 2.0", "Bienestar"])

with tab1:
    st.subheader("🧠 Cumplimiento NOM-035")
    df_nom = nom_data[nom_data['Departamento'].isin(departamentos)]
    fig = px.bar(df_nom, x='Departamento', y=['Evaluaciones', 'Capacitaciones'], barmode='group')
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("🗺 Mapa de Riesgo Psicosocial")
    fig2 = go.Figure(data=go.Heatmap(
        z=df_nom[['Evaluaciones', 'Capacitaciones', 'Incidentes']].T.values,
        x=df_nom['Departamento'],
        y=['Evaluaciones', 'Capacitaciones', 'Incidentes'],
        colorscale='RdYlGn', reversescale=True))
    st.plotly_chart(fig2, use_container_width=True)

with tab2:
    st.subheader("🔄 Progreso LEAN 2.0")
    df_lean = lean_data[lean_data['Departamento'].isin(departamentos)]
    fig = px.bar(df_lean, x='Departamento', y='Eficiencia', color='Eficiencia', color_continuous_scale='Greens')
    st.plotly_chart(fig, use_container_width=True)

    fig_radar = go.Figure()
    for dept in departamentos:
        row = df_lean[df_lean['Departamento'] == dept].iloc[0]
        fig_radar.add_trace(go.Scatterpolar(r=[row['Eficiencia'], row['Reducción Desperdicio'], row['Proyectos Activos']],
                                            theta=['Eficiencia', 'Red. Desperdicio', 'Proy. Activos'], fill='toself', name=dept))
    fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])))
    st.plotly_chart(fig_radar, use_container_width=True)

with tab3:
    st.subheader("❤️ Tendencias de Bienestar")
    fig = px.line(bienestar_data, x='Mes', y=['Índice Bienestar', 'Ausentismo', 'Rotación'], markers=True)
    st.plotly_chart(fig, use_container_width=True)

# --- ALERTAS ---
st.markdown("## 🚨 Alertas")
col1, col2 = st.columns(2)
with col1:
    st.subheader("🔴 Áreas Críticas")
    criticos = nom_data[nom_data['Evaluaciones'] < 80][['Departamento', 'Evaluaciones', 'Incidentes']]
    st.dataframe(criticos.style.background_gradient(cmap='Reds'), use_container_width=True)

with col2:
    st.subheader("🟡 Oportunidades de Mejora")
    mejora = lean_data[lean_data['Eficiencia'] < 75][['Departamento', 'Eficiencia', 'Reducción Desperdicio']]
    st.dataframe(mejora.style.background_gradient(cmap='Oranges'), use_container_width=True)

# --- PLAN DE ACCIÓN ---
with st.expander("📝 Plan de Acción"):
    form = st.form("plan_form")
    dept = form.selectbox("Departamento", nom_data['Departamento'].unique())
    problema = form.text_input("Problema")
    accion = form.text_area("Acción Propuesta")
    responsable = form.text_input("Responsable")
    plazo = form.date_input("Plazo")
    submit = form.form_submit_button("💾 Guardar")
    if submit:
        st.success(f"Plan de acción para {dept} guardado.")

# --- PIE ---
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#666; font-size:0.9rem;">
    <p>© 2025 Sistema NOM-035 & LEAN 2.0</p>
    <p>📞 Soporte: (663) 558 2452 | 📧 contacto@lean2institute.org</p>
</div>
""", unsafe_allow_html=True)
