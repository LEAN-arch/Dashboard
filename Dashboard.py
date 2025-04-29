import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, date
from typing import Tuple
# -----------------------------
# CONFIGURACIÓN DE PÁGINA
# -----------------------------
st.set_page_config(
    page_title="Dashboard Integral NOM-035 + LEAN 2.0",
    layout="wide",
    page_icon="📊"
)

# -----------------------------
# CARGA Y SIMULACIÓN DE DATOS
# -----------------------------
@st.cache_data
def cargar_datos(seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Simula datos para NOM-035, LEAN 2.0 y Bienestar Organizacional."""
    np.random.seed(seed)
    departamentos = ['Producción', 'Calidad', 'Logística', 'Administración', 'Ventas', 'RH', 'TI']

    nom_df = pd.DataFrame({
        'Departamento': departamentos,
        'Evaluaciones': np.random.randint(70, 100, size=7),
        'Capacitaciones': np.random.randint(60, 100, size=7),
        'Incidentes': np.random.randint(0, 10, size=7),
        'Tendencia': np.round(np.random.normal(0.5, 1.5, size=7), 2)
    })

    lean_df = pd.DataFrame({
        'Departamento': departamentos,
        'Eficiencia': np.random.randint(60, 95, size=7),
        'Reducción Desperdicio': np.random.randint(5, 25, size=7),
        'Proyectos Activos': np.random.randint(1, 6, size=7)
    })

    bienestar_df = pd.DataFrame({
        'Mes': pd.date_range(start='2024-01-01', periods=12, freq='M'),
        'Índice Bienestar': np.round(np.random.normal(75, 5, 12), 1),
        'Ausentismo': np.round(np.random.normal(8, 2, 12), 1),
        'Rotación': np.round(np.random.normal(12, 3, 12), 1)
    })

    return nom_df, lean_df, bienestar_df


nom_data, lean_data, bienestar_data = cargar_datos()

# -----------------------------
# FILTROS LATERALES
# -----------------------------
with st.sidebar:
    st.header("🎛️ Filtros Generales")

    fecha_inicio = st.date_input("📅 Fecha de inicio", date(2025, 1, 1))
    fecha_fin = st.date_input("📅 Fecha de fin", date(2025, 4, 1))

    departamentos_filtro = st.multiselect(
        "🏢 Departamentos",
        options=nom_data['Departamento'].unique().tolist(),
        default=['Producción', 'Calidad', 'Logística']
    )

    metricas = st.multiselect(
        "📊 Métricas clave",
        ['NOM-035', 'Calidad', 'Productividad', 'Bienestar', 'LEAN'],
        default=['NOM-035', 'Calidad']
    )

    if st.button("🔄 Aplicar filtros"):
        st.experimental_rerun()

# -----------------------------
# ENCABEZADO Y FECHA
# -----------------------------
st.markdown(f"""
<div style="display: flex; justify-content: space-between; align-items: center;">
    <div>
        <h1>📈 Dashboard Integral NOM-035 & LEAN 2.0</h1>
        <p style="color:gray; margin-top:-1rem;">Monitoreo de riesgos psicosociales, eficiencia y bienestar organizacional.</p>
    </div>
    <div style="text-align:right; font-size:0.9rem; color:#888;">
        Última actualización: <strong>{datetime.now().strftime('%d/%m/%Y %H:%M')}</strong>
    </div>
</div>
<hr>
""", unsafe_allow_html=True)

# -----------------------------
# KPI Cards
# -----------------------------
def kpi_card(valor: float, titulo: str, meta: float = 90):
    delta = valor - meta
    color = "green" if valor >= meta else "orange" if valor >= meta - 10 else "red"
    emoji = "✅" if valor >= meta else "⚠️" if valor >= meta - 10 else "❌"

    st.markdown(f"""
    <div style="background-color:{color}; padding:1rem; border-radius:10px; color:white;">
        <h4 style="margin:0;">{emoji} {titulo}</h4>
        <p style="font-size:1.8rem; margin:0;">{valor:.1f}%</p>
        <p style="font-size:0.85rem;">Meta: {meta:.0f}%</p>
    </div>
    """, unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)
with col1: kpi_card(92, "Cumplimiento NOM-035")
with col2: kpi_card(85, "Adopción LEAN 2.0", 80)
with col3: kpi_card(78, "Índice Bienestar", 85)
with col4: kpi_card(65, "Eficiencia Operativa", 75)

# -----------------------------
# VISUALIZACIONES
# -----------------------------
tab1, tab2, tab3 = st.tabs(["📘 NOM-035", "🧩 LEAN 2.0", "💚 Bienestar"])

with tab1:
    st.subheader("📘 Cumplimiento por Departamento")
    filtered_nom = nom_data[nom_data['Departamento'].isin(departamentos_filtro)]

    st.plotly_chart(
        px.bar(filtered_nom, x="Departamento", y=["Evaluaciones", "Capacitaciones"],
               barmode="group", color_discrete_sequence=px.colors.qualitative.Pastel),
        use_container_width=True
    )

    st.subheader("🧠 Mapa de Riesgo Psicosocial")
    st.plotly_chart(go.Figure(
        data=go.Heatmap(
            z=filtered_nom[['Evaluaciones', 'Capacitaciones', 'Incidentes']].values.T,
            x=filtered_nom['Departamento'],
            y=['Evaluaciones', 'Capacitaciones', 'Incidentes'],
            colorscale='RdYlGn',
            reversescale=True
        )
    ), use_container_width=True)

with tab2:
    st.subheader("🧩 Avance LEAN por Departamento")
    filtered_lean = lean_data[lean_data['Departamento'].isin(departamentos_filtro)]

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(
            px.bar(filtered_lean, x="Departamento", y="Eficiencia",
                   color="Eficiencia", color_continuous_scale="greens"),
            use_container_width=True
        )
    with col2:
        fig_radar = go.Figure()
        for _, row in filtered_lean.iterrows():
            fig_radar.add_trace(go.Scatterpolar(
                r=[row['Eficiencia'], row['Reducción Desperdicio'], row['Proyectos Activos'] * 20],
                theta=['Eficiencia', 'Reducción', 'Proyectos'],
                fill='toself',
                name=row['Departamento']
            ))
        fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True)), showlegend=True)
        st.plotly_chart(fig_radar, use_container_width=True)

with tab3:
    st.subheader("💚 Tendencias de Bienestar")
    st.plotly_chart(
        px.line(bienestar_data, x="Mes", y=["Índice Bienestar", "Ausentismo", "Rotación"], markers=True),
        use_container_width=True
    )

# -----------------------------
# ALERTAS
# -----------------------------
st.markdown("## 🚨 Alertas y Planes de Acción")

col1, col2 = st.columns(2)
with col1:
    st.markdown("### 🔴 Áreas Críticas NOM-035")
    st.dataframe(
        nom_data[nom_data['Evaluaciones'] < 80]
        .sort_values('Evaluaciones')
        .style.background_gradient(cmap='Reds')
    )

with col2:
    st.markdown("### 🟠 Oportunidades de Mejora LEAN")
    st.dataframe(
        lean_data[lean_data['Eficiencia'] < 75]
        .sort_values('Eficiencia')
        .style.background_gradient(cmap='Oranges')
    )

# -----------------------------
# PLAN DE ACCIÓN
# -----------------------------
with st.expander("📝 Registrar nuevo plan de acción"):
    col1, col2 = st.columns(2)
    with col1:
        dept = st.selectbox("Departamento", nom_data['Departamento'].unique())
        problema = st.text_input("Problema identificado")
        responsable = st.text_input("Responsable asignado")
    with col2:
        accion = st.text_area("Acción propuesta")
        plazo = st.date_input("Plazo estimado")

    if st.button("💾 Guardar Plan"):
        st.success(f"✅ Plan registrado para {dept}.")

# -----------------------------
# EXPORTACIÓN (Simulada)
# -----------------------------
st.markdown("---")
st.subheader("📤 Exportación")
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("📄 Generar PDF"):
        st.success("✅ Reporte PDF simulado.")
with col2:
    if st.button("📊 Exportar Excel"):
        st.success("✅ Datos exportados (simulado).")
with col3:
    if st.button("📧 Enviar Reporte"):
        st.success("✅ Reporte enviado (simulado).")

# -----------------------------
# PIE DE PÁGINA
# -----------------------------
st.markdown("""
<hr>
<div style='text-align:center; color:gray; font-size:0.85rem;'>
    © 2025 Sistema NOM-035 + LEAN 2.0 • 📧 contacto@lean2institute.org • 📞 Soporte: (663) 558-2452
</div>
""", unsafe_allow_html=True)
