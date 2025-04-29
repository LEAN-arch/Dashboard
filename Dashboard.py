import streamlit as st
import pandas as pd
import plotly.express as px
from io import BytesIO
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import pdfkit

# Page configuration with larger title and logo at the top
st.set_page_config(page_title="Dashboard Corporativo: Cumplimiento NOM-035 y Evolución LEAN 2.0", layout="wide")

# Header with Logo and Title
st.image("logo_empresa.png", width=150)  # Company logo
st.markdown("<h1 style='text-align: center; font-size: 36px; font-weight: bold; color: #2d3e50;'>🏛️ Dashboard Corporativo: Cumplimiento NOM-035 y Evolución LEAN 2.0</h1>", unsafe_allow_html=True)
st.subheader("Monitoreo Estratégico de Bienestar, Cultura y Mejora Continua")
st.caption("Última actualización: Abril 2025")
st.markdown("---")

# Sidebar Filters - Card Layout for better UX
st.sidebar.header("🔎 Filtros Dinámicos")
st.sidebar.markdown("<div style='background-color:#f1f1f1; padding: 10px; border-radius: 8px;'>", unsafe_allow_html=True)
departamentos = st.sidebar.multiselect("Selecciona Departamentos:", ['Producción', 'Calidad', 'Logística', 'Administración', 'Ventas'], default=['Producción', 'Calidad', 'Logística', 'Administración', 'Ventas'])
st.sidebar.markdown("</div>", unsafe_allow_html=True)

# KPIs Section: Adding Cards for KPIs with color codes
st.header("📊 KPIs Estratégicos")

# Function for color-coding based on KPI performance
def semaforo(kpi_value):
    if kpi_value > 90:
        return "🟢", "green"
    elif 70 <= kpi_value <= 89:
        return "🟡", "yellow"
    else:
        return "🔴", "red"

# Display KPIs with better spacing and clear icons
col1, col2, col3 = st.columns(3)
with col1:
    kpi_1 = 92
    icon, color = semaforo(kpi_1)
    st.markdown(f"<div style='padding: 20px; background-color:{color}; border-radius: 8px; text-align: center; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);'><h3>{icon} Cumplimiento NOM-035</h3><p style='font-size: 28px; font-weight: bold;'>{kpi_1}%</p></div>", unsafe_allow_html=True)

with col2:
    kpi_2 = 80
    icon, color = semaforo(kpi_2)
    st.markdown(f"<div style='padding: 20px; background-color:{color}; border-radius: 8px; text-align: center; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);'><h3>{icon} Cumplimiento en Calidad</h3><p style='font-size: 28px; font-weight: bold;'>{kpi_2}%</p></div>", unsafe_allow_html=True)

with col3:
    kpi_3 = 70
    icon, color = semaforo(kpi_3)
    st.markdown(f"<div style='padding: 20px; background-color:{color}; border-radius: 8px; text-align: center; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);'><h3>{icon} Cumplimiento en Logística</h3><p style='font-size: 28px; font-weight: bold;'>{kpi_3}%</p></div>", unsafe_allow_html=True)

# Enhanced Graph for KPIs Visualization with Tooltips
fig1 = px.bar(pd.DataFrame({
    'Departamento': ['Producción', 'Calidad', 'Logística', 'Administración', 'Ventas'],
    'Evaluaciones Aplicadas (%)': [95, 90, 85, 80, 75]
}), x='Evaluaciones Aplicadas (%)', y='Departamento', orientation='h', color='Departamento', 
    title="Evaluaciones Aplicadas por Departamento",
    labels={'Evaluaciones Aplicadas (%)': 'Cumplimiento (%)', 'Departamento': 'Departamentos'})
fig1.update_layout(template="plotly_dark", showlegend=False)
st.plotly_chart(fig1, use_container_width=True)

# Adding Esquema de KPIs Visualizados Table
st.subheader("📋 Esquema de KPIs Visualizados")

kpi_table = pd.DataFrame({
    'Indicador': ['Evaluaciones Aplicadas', 'Protocolos Implementados', 'Acciones Preventivas vs Correctivas', 'Reportes Psicosociales', 
                  'Índice de Bienestar Organizacional', 'Adopción Herramientas LEAN 2.0', 'Participación en Programas de Bienestar', 
                  'Índice de Equidad Laboral', 'Reducción de Ausentismo por Estrés'],
    'Tipo de Gráfico': ['Gauge Chart', 'KPI Card', 'Bar Chart', 'Line Chart', 'KPI Card + Line Chart', 'KPI Card', 'Donut Chart', 'KPI Card', 'Line Chart'],
    'Frecuencia de Actualización': ['Trimestral', 'Anual', 'Trimestral', 'Mensual', 'Semestral', 'Trimestral', 'Semestral', 'Anual', 'Trimestral']
})

st.dataframe(kpi_table, use_container_width=True)

# Alerts Section: Critical KPI Alerts
st.markdown("<div style='background-color:#ffeeba; padding: 20px; border-radius: 8px; margin-top: 20px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);'>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; font-weight: bold;'>🚨 Alerta: KPI Crítico en Cumplimiento NOM-035</h4>", unsafe_allow_html=True)
st.markdown("**Se ha detectado que el KPI de Cumplimiento ha caído por debajo del 90%. Valor actual: 85%.**", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# Footer Section: Clean design with company info
st.markdown("""
    <div style="background-color:#2e3b4e; color:white; text-align:center; padding: 10px; font-size: 14px; border-radius: 8px;">
        <p>📧 Contáctanos: soporte@empresa.com | 📍 Visítanos en: Calle Ejemplo 123, Tijuana</p>
        <p>© 2025 Empresa XYZ - Todos los derechos reservados.</p>
    </div>
""", unsafe_allow_html=True)

# Additional sections (Optional for export to PDF, etc.)
# Example: Generate a PDF Report

def generate_pdf():
    html_content = """
    <html>
        <head><title>Reporte LEAN 2.0</title></head>
        <body>
            <h1>Reporte de Cumplimiento LEAN 2.0 y NOM-035</h1>
            <p><b>Fecha:</b> 25 de Abril de 2025</p>
            <p><b>Resumen:</b> Evaluación de KPIs y alertas del sistema.</p>
            <h3>KPIs:</h3>
            <ul>
                <li>Cumplimiento NOM-035: 92%</li>
                <li>Cumplimiento en Calidad: 80%</li>
                <li>Cumplimiento en Logística: 70%</li>
            </ul>
        </body>
    </html>
    """
    pdf = pdfkit.from_string(html_content, False)
    return pdf

# Download Report Button
if st.button('Generar Reporte PDF'):
    pdf_data = generate_pdf()
    st.download_button(label="Descargar Reporte PDF", data=pdf_data, file_name="reporte_lean_2_0.pdf", mime="application/pdf")

