import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, recall_score, precision_score

# 1. Configuración inicial
st.set_page_config(page_title="Simulador de Costos - Vuelos", layout="wide")
st.title("✈️ Dashboard de Costos y Umbrales: Cancelación de Vuelos")

# 2. Cargar los datos exportados
@st.cache_data
def cargar_datos():
    # Asegúrate de que este archivo esté en la misma carpeta que app.py en GitHub
    return pd.read_csv('datos_dashboard_vuelos.csv')

df = cargar_datos()
y_test = df['y_real']
probabilidades = df['probabilidad']

# 3. Panel Lateral: Controles Interactivos
st.sidebar.header("🕹️ Controles de Costos")
costo_fp = st.sidebar.slider("Costo Falso Positivo (FP) - Falsa Alarma", min_value=100, max_value=5000, value=500, step=100)
costo_fn = st.sidebar.slider("Costo Falso Negativo (FN) - Cancelación Sorpresa", min_value=5000, max_value=50000, value=15000, step=1000)

st.sidebar.markdown("---")
st.sidebar.header("🎚️ Ajuste de Umbral")
umbral_manual = st.sidebar.slider("Mueve el umbral manualmente", min_value=0.01, max_value=0.99, value=0.50, step=0.01)

# 4. Cálculos Matemáticos
# Umbral Teórico
umbral_teorico = costo_fp / (costo_fp + costo_fn)

# Curva de Costos y Umbral Empírico
umbrales = np.linspace(0.01, 0.99, 100)
costos_totales = []

for u in umbrales:
    y_pred_temp = (probabilidades >= u).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_temp).ravel()
    costo = (fp * costo_fp) + (fn * costo_fn)
    costos_totales.append(costo)

umbral_empirico = umbrales[np.argmin(costos_totales)]
costo_minimo_empirico = min(costos_totales)

# Cálculos para el Umbral Manual (el que mueve el usuario)
y_pred_manual = (probabilidades >= umbral_manual).astype(int)
tn_m, fp_m, fn_m, tp_m = confusion_matrix(y_test, y_pred_manual).ravel()
costo_manual = (fp_m * costo_fp) + (fn_m * costo_fn)
recall_m = recall_score(y_test, y_pred_manual, zero_division=0)
precision_m = precision_score(y_test, y_pred_manual, zero_division=0)

# Costo del Umbral 0.5 (Base) para calcular el ahorro
y_pred_05 = (probabilidades >= 0.5).astype(int)
tn_05, fp_05, fn_05, tp_05 = confusion_matrix(y_test, y_pred_05).ravel()
costo_05 = (fp_05 * costo_fp) + (fn_05 * costo_fn)
ahorro = costo_05 - costo_manual

# 5. Interfaz Principal (Métricas)
col1, col2, col3, col4 = st.columns(4)
col1.metric("Costo Total (Umbral Manual)", f"${costo_manual:,.0f}")
col2.metric("Ahorro vs Umbral 0.5", f"${ahorro:,.0f}", delta=f"${ahorro:,.0f}", delta_color="normal")
col3.metric("Recall", f"{recall_m:.1%}")
col4.metric("Precisión", f"{precision_m:.1%}")

st.markdown("---")

# 6. Gráficos
col_izq, col_der = st.columns(2)

with col_izq:
    st.subheader("Matriz de Confusión (Umbral Manual)")
    st.write(f"**FP (Falsa alarma):** {fp_m} vuelos (Costo: ${fp_m * costo_fp:,.0f})")
    st.write(f"**FN (Sorpresa):** {fn_m} vuelos (Costo: ${fn_m * costo_fn:,.0f})")
    
    fig_matriz, ax_matriz = plt.subplots(figsize=(5, 4))
    sns.heatmap([[tn_m, fp_m], [fn_m, tp_m]], annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax_matriz)
    ax_matriz.set_xlabel('Predicción')
    ax_matriz.set_ylabel('Realidad')
    ax_matriz.set_xticklabels(['Normal (0)', 'Cancelado (1)'])
    ax_matriz.set_yticklabels(['Normal (0)', 'Cancelado (1)'])
    st.pyplot(fig_matriz)

with col_der:
    st.subheader("Curva de Costo Total vs Umbral")
    fig_curva, ax_curva = plt.subplots(figsize=(6, 4))
    ax_curva.plot(umbrales, costos_totales, color='navy', label='Costo Total')
    ax_curva.axvline(x=0.5, color='red', linestyle='--', label='Umbral 0.5')
    ax_curva.axvline(x=umbral_empirico, color='green', linestyle='--', label=f'Óptimo ({umbral_empirico:.2f})')
    ax_curva.axvline(x=umbral_manual, color='orange', linestyle='-', linewidth=2, label=f'Manual ({umbral_manual:.2f})')
    ax_curva.set_xlabel('Umbral de Decisión')
    ax_curva.set_ylabel('Costo Total ($)')
    ax_curva.legend()
    st.pyplot(fig_curva)

st.markdown(f"**💡 Insight del Modelo:** El Umbral Teórico matemático es **{umbral_teorico:.3f}**. El Umbral Empírico óptimo encontrado en la curva es **{umbral_empirico:.3f}**.")