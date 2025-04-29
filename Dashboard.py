import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, date
from sklearn.linear_model import LinearRegression

# ---- CONFIGURACIÓN GENERAL ----
st.set_page_config(
    page_title="Sistema Integral NOM-035 & LEAN 2.0",
    layout="wide",
    page_icon="📊"
)

# ---- CARGA DE DATOS ----
@st.cache_data(ttl=600)
def cargar_datos():
    departamentos = ['Producción', 'Calidad', 'Logística', 'Administración', 'Ventas', 'RH', 'TI']

    nom = pd.DataFrame({
        'Departamento': departamentos,
        'Evaluaciones': np.random.randint(70, 100, 7),
        'Capacitaciones': np.random.randint(60, 100, 7),
        'Incidentes': np.random.randint(0, 10, 7),
        'Tendencia': np.round(np.random.normal(0.5, 1.5, 7), 2)
    })

    lean = pd.DataFrame({
        'Departamento': departamentos,
        'Eficiencia': np.random.randint(60, 95, 7),
        'Reducción Desperdicio': np.random.randint(5, 25, 7),
        'Proyectos Activos': np.random.randint(1, 6, 7)
    })

    bienestar = pd.DataFrame({
        'Mes': pd.date_range(start='2024-01-01', periods=12, freq='M'),
        'Índice Bienestar': np.round(np.random.normal(75, 5, 12), 1),
        'Ausentismo': np.round(np.random.normal(8, 2, 12), 1),
        'Rotación': np.round(np.random.normal(12, 3, 12), 1)
    })

    return nom, lean, bienestar

nom_df, lean_df, bienestar_df = cargar_datos()

# ---- SIDEBAR ----
with st.sidebar:
    st.title("Filtros Avanzados")
    fecha_inicio = st.date_input("Fecha de inicio", value=date(2025, 1, 1))
    fecha_fin = st.date_input("Fecha de fin", value=date(2025, 4, 1))
    departamentos_filtro = st.multiselect(
        "Seleccionar Departamentos",
        options=nom_df['Departamento'].unique().tolist(),
        default=['Producción', 'Calidad', 'Logística']
    )
    metricas = st.multiselect(
        "Seleccionar Métricas",
        ['NOM-035', 'Calidad', 'Productividad', 'Bienestar', 'LEAN'],
        default=['NOM-035', 'Calidad']
    )
actualizar = st.button("🔄 Actualizar")

# Main code continues and optionally checks this:
if actualizar:
    st.toast("Actualizando datos...", icon="🔄")
    st.experimental_rerun()

# ---- ENCABEZADO ----
st.markdown(f"""
    <div style='display: flex; align-items: center;'>
        <h1 style='margin-bottom: 0;'>📈 Sistema Integral NOM-035 & LEAN 2.0</h1>
        <p style='margin-top: 0; color: gray;'>Monitoreo Estratégico de Bienestar Psicosocial y Eficiencia Operacional</p>
    </div>
    <div style='text-align: right; color: #888; font-size:0.85rem;'>Última actualización: {datetime.now().strftime('%d/%m/%Y %H:%M')}</div>
    <hr style='margin-top: 0.5rem;'>
""", unsafe_allow_html=True)

# ---- KPI CARDS ----
def mostrar_kpi(valor, titulo, meta=90):
    delta = valor - meta
    estado = "✅" if valor >= meta else "⚠️" if valor >= meta - 10 else "❌"
    color = "#4CAF50" if valor >= meta else "#FF9800" if valor >= meta - 10 else "#F44336"

    st.markdown(f"""
        <div style='background-color:{color};padding:1rem;border-radius:12px;color:white;text-align:center;'>
            <div style='font-size:1.25rem;'>{estado} {titulo}</div>
            <div style='font-size:2rem;font-weight:bold;'>{valor}%</div>
            <div style='font-size:0.8rem;'>Meta: {meta}%</div>
        </div>
    """, unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)
with col1: mostrar_kpi(92, "Cumplimiento NOM-035")
with col2: mostrar_kpi(85, "Adopción LEAN 2.0", 80)
with col3: mostrar_kpi(78, "Índice Bienestar", 85)
with col4: mostrar_kpi(65, "Eficiencia Operativa", 75)

# ---- TABS ----
tab1, tab2, tab3 = st.tabs(["NOM-035", "LEAN 2.0", "Bienestar"])

with tab1:
    st.subheader("Cumplimiento NOM-035 por Departamento")
    filtered_nom = nom_df[nom_df['Departamento'].isin(departamentos_filtro)]
    fig = px.bar(filtered_nom, x="Departamento", y=["Evaluaciones", "Capacitaciones"], barmode="group")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Mapa de Riesgo Psicosocial")
    fig_heat = go.Figure(data=go.Heatmap(
        z=filtered_nom[['Evaluaciones', 'Capacitaciones', 'Incidentes']].values.T,
        x=filtered_nom['Departamento'],
        y=['Evaluaciones', 'Capacitaciones', 'Incidentes'],
        colorscale='RdYlGn',
        reversescale=True
    ))
    st.plotly_chart(fig_heat, use_container_width=True)

with tab2:
    st.subheader("Progreso LEAN 2.0")
    filtered_lean = lean_df[lean_df['Departamento'].isin(departamentos_filtro)]

    col1, col2 = st.columns(2)
    with col1:
        fig_lean = px.bar(filtered_lean, x='Departamento', y='Eficiencia', color='Eficiencia', color_continuous_scale='Greens')
        st.plotly_chart(fig_lean, use_container_width=True)

    with col2:
        fig_radar = go.Figure()
        for dept in filtered_lean['Departamento']:
            row = filtered_lean[filtered_lean['Departamento'] == dept].iloc[0]
            fig_radar.add_trace(go.Scatterpolar(
                r=[row['Eficiencia'], row['Reducción Desperdicio'], row['Proyectos Activos']*20],
                theta=['Eficiencia', 'Reducción', 'Proyectos'],
                fill='toself',
                name=dept
            ))
        fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True)), showlegend=True)
        st.plotly_chart(fig_radar, use_container_width=True)

with tab3:
    st.subheader("Tendencias de Bienestar Organizacional")
    fig_bienestar = px.line(bienestar_df, x='Mes', y=['Índice Bienestar', 'Ausentismo', 'Rotación'], markers=True)
    st.plotly_chart(fig_bienestar, use_container_width=True)

    X = np.arange(len(bienestar_df)).reshape(-1, 1)
    y = bienestar_df['Índice Bienestar'].values
    modelo = LinearRegression().fit(X, y)
    tendencia = modelo.coef_[0]
    st.info(f"Tendencia de Bienestar (pendiente): {'↑' if tendencia > 0 else '↓'} {tendencia:.2f} puntos/mes")

# ---- ALERTAS Y PLANES DE ACCIÓN ----
st.markdown("## Alertas y Planes de Acción")
col1, col2 = st.columns(2)

with col1:
    st.markdown("### Áreas Críticas NOM-035")
    st.dataframe(nom_df[nom_df['Evaluaciones'] < 80].style.background_gradient(cmap='Reds'))

with col2:
    st.markdown("### Oportunidades de Mejora LEAN")
    st.dataframe(lean_df[lean_df['Eficiencia'] < 75].style.background_gradient(cmap='Oranges'))

with st.expander("📝 Registrar nuevo plan de acción"):
    col1, col2 = st.columns(2)
    with col1:
        dept = st.selectbox("Departamento", nom_df['Departamento'].unique())
        problema = st.text_input("Problema identificado")
        responsable = st.text_input("Responsable")
    with col2:
        accion = st.text_area("Acción propuesta")
        plazo = st.date_input("Plazo estimado")
    if st.button("💾 Guardar Plan de Acción"):
        st.success(f"✅ Plan registrado para {dept} - Responsable: {responsable}")

# ---- EXPORTACIÓN ----
st.markdown("---")
st.subheader("Exportación de Datos")
col1, col2 = st.columns(2)
with col1:
    if st.button("📄 Generar Reporte PDF"):
        st.success("✅ Reporte generado (simulado)")
with col2:
    if st.button("📧 Enviar Reporte"):
        st.success("📬 Reporte enviado (simulado)")

# ---- PIE DE PÁGINA ----
st.markdown("""
<hr>
<div style='text-align:center; color:gray; font-size:0.85rem;'>
    © 2025 Sistema NOM-035 + LEAN 2.0 • 📧 contacto@lean2institute.org • 📞 Soporte: (663) 558-2452
</div>
""", unsafe_allow_html=True)
