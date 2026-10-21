# Importamos Streamlit, la librería principal que convierte este script en una página web interactiva
import streamlit as st
# Importamos Pandas para manejar los datos (leer el CSV y manipular columnas)
import pandas as pd
# Importamos NumPy para operaciones matemáticas rápidas, como crear listas de números y buscar mínimos
import numpy as np
# Importamos Matplotlib y Seaborn, que son nuestras herramientas para dibujar los gráficos (curvas y matrices)
import matplotlib.pyplot as plt
import seaborn as sns
# Importamos desde Scikit-Learn las métricas para evaluar el rendimiento de nuestro modelo de clasificación
from sklearn.metrics import confusion_matrix, recall_score, precision_score

# ==========================================
# 1. Configuración inicial de la página
# ==========================================
# st.set_page_config ajusta cómo se ve la pestaña del navegador y usa todo el ancho de la pantalla (layout="wide")
st.set_page_config(page_title="Simulador de Costos - Vuelos", layout="wide")
# st.title pone el título principal en grande en la parte superior de la página web
st.title("✈️ Dashboard de Costos y Umbrales: Cancelación de Vuelos")

# ==========================================
# 2. Cargar los datos exportados
# ==========================================
# Usamos el decorador @st.cache_data para que Streamlit guarde los datos en la memoria caché.
# Esto es vital porque evita que el archivo CSV se vuelva a leer desde cero cada vez que movemos un botón.
@st.cache_data
def cargar_datos():
    # Leemos el archivo CSV que exportamos desde nuestro Colab, el cual contiene la realidad y la predicción
    # Asegúrate de que este archivo esté en la misma carpeta que app.py en GitHub
    return pd.read_csv('datos_dashboard_vuelos.csv')

# Llamamos a la función para guardar los datos en la variable 'df' (nuestro DataFrame)
df = cargar_datos()
# Separamos la columna con lo que pasó realmente (0 = normal, 1 = cancelado)
y_test = df['y_real']
# Separamos la columna con las probabilidades matemáticas (de 0.0 a 1.0) que calculó nuestro LightGBM
probabilidades = df['probabilidad']

# ==========================================
# 3. Panel Lateral: Controles Interactivos
# ==========================================
# st.sidebar pone los elementos en el menú lateral izquierdo de la página
st.sidebar.header("🕹️ Controles de Costos")

# Creamos un slider (barra deslizable) para que el usuario defina cuánto le cuesta a la empresa una falsa alarma (FP)
costo_fp = st.sidebar.slider("Costo Falso Positivo (FP) - Falsa Alarma", min_value=100, max_value=5000, value=500, step=100)
# Creamos otro slider para el costo de una cancelación sorpresa (FN), que suele ser mucho más cara
costo_fn = st.sidebar.slider("Costo Falso Negativo (FN) - Cancelación Sorpresa", min_value=5000, max_value=50000, value=15000, step=1000)

# Dibujamos una línea separadora en el menú lateral
st.sidebar.markdown("---")
# Agregamos una sección para que el gerente (usuario) pueda mover el umbral a mano y ver qué pasa
st.sidebar.header("🎚️ Ajuste de Umbral")
umbral_manual = st.sidebar.slider("Mueve el umbral manualmente", min_value=0.01, max_value=0.99, value=0.50, step=0.01)

# ==========================================
# 4. Cálculos Matemáticos
# ==========================================
# Calculamos el Umbral Teórico usando la fórmula estricta del profesor: CFP / (CFP + CFN)
umbral_teorico = costo_fp / (costo_fp + costo_fn)

# Para armar la curva y buscar el Umbral Empírico, creamos una lista de 100 posibles umbrales (desde 0.01 hasta 0.99)
umbrales = np.linspace(0.01, 0.99, 100)
# Creamos una lista vacía donde iremos guardando cuánta plata se pierde en cada uno de esos 100 umbrales
costos_totales = []

# Iteramos (probamos) uno por uno los 100 umbrales
for u in umbrales:
    # Si la probabilidad del modelo es mayor o igual al umbral 'u', lo marcamos como cancelado (1), si no, normal (0)
    y_pred_temp = (probabilidades >= u).astype(int)
    # Generamos la matriz de confusión y extraemos los Verdaderos Negativos, Falsos Positivos, Falsos Negativos y Verdaderos Positivos
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_temp).ravel()
    # Calculamos el costo en dólares de equivocarnos con este umbral específico
    costo = (fp * costo_fp) + (fn * costo_fn)
    # Guardamos el resultado en nuestra lista
    costos_totales.append(costo)

# Buscamos matemáticamente cuál de los 100 umbrales nos dio el menor costo posible (el óptimo)
umbral_empirico = umbrales[np.argmin(costos_totales)]
# Guardamos la cantidad de dinero mínimo que se pierde usando ese umbral óptimo
costo_minimo_empirico = min(costos_totales)

# Ahora hacemos el mismo cálculo, pero SOLO para el umbral que el usuario está moviendo con el slider (Umbral Manual)
y_pred_manual = (probabilidades >= umbral_manual).astype(int)
tn_m, fp_m, fn_m, tp_m = confusion_matrix(y_test, y_pred_manual).ravel()
costo_manual = (fp_m * costo_fp) + (fn_m * costo_fn)
# Calculamos las métricas de Recall (qué porcentaje de los cancelados atrapamos) y Precisión (qué tan certeras son nuestras alarmas)
recall_m = recall_score(y_test, y_pred_manual, zero_division=0)
precision_m = precision_score(y_test, y_pred_manual, zero_division=0)

# Para demostrarle a la gerencia cuánto ahorramos, calculamos el costo asumiendo un umbral clásico y ciego de 0.5
y_pred_05 = (probabilidades >= 0.5).astype(int)
tn_05, fp_05, fn_05, tp_05 = confusion_matrix(y_test, y_pred_05).ravel()
costo_05 = (fp_05 * costo_fp) + (fn_05 * costo_fn)
# El ahorro es simplemente el costo catastrófico del umbral 0.5 menos el costo de nuestro umbral manual
ahorro = costo_05 - costo_manual

# ==========================================
# 5. Interfaz Principal (Métricas en Pantalla)
# ==========================================
# Dividimos la pantalla principal en 4 columnas iguales para poner nuestros indicadores (tarjetas)
col1, col2, col3, col4 = st.columns(4)
# Tarjeta 1: Muestra el costo total en dólares del umbral que el usuario eligió
col1.metric("Costo Total (Umbral Manual)", f"${costo_manual:,.0f}")
# Tarjeta 2: Muestra la plata que estamos ahorrando. 'delta' le pone una flechita verde hacia arriba si es positivo
col2.metric("Ahorro vs Umbral 0.5", f"${ahorro:,.0f}", delta=f"${ahorro:,.0f}", delta_color="normal")
# Tarjeta 3 y 4: Muestran los porcentajes de Recall y Precisión
col3.metric("Recall", f"{recall_m:.1%}")
col4.metric("Precisión", f"{precision_m:.1%}")

# Dibujamos una línea horizontal para separar los números de los gráficos
st.markdown("---")

# ==========================================
# 6. Gráficos Interactivos
# ==========================================
# Dividimos la pantalla en 2 columnas grandes para poner los gráficos uno al lado del otro
col_izq, col_der = st.columns(2)

# Entramos a la columna izquierda
with col_izq:
    # Título del gráfico izquierdo
    st.subheader("Matriz de Confusión (Umbral Manual)")
    # Imprimimos en texto cuántos errores hubieron y cuánta plata nos costó cada tipo de error
    st.write(f"**FP (Falsa alarma):** {fp_m} vuelos (Costo: ${fp_m * costo_fp:,.0f})")
    st.write(f"**FN (Sorpresa):** {fn_m} vuelos (Costo: ${fn_m * costo_fn:,.0f})")
    
    # Preparamos el lienzo del gráfico con Matplotlib
    fig_matriz, ax_matriz = plt.subplots(figsize=(5, 4))
    # Usamos Seaborn para dibujar un 'heatmap' (mapa de calor) que representa la matriz de confusión visualmente
    sns.heatmap([[tn_m, fp_m], [fn_m, tp_m]], annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax_matriz)
    # Le ponemos nombre a los ejes (X = lo que predijo el modelo, Y = lo que pasó en realidad)
    ax_matriz.set_xlabel('Predicción')
    ax_matriz.set_ylabel('Realidad')
    ax_matriz.set_xticklabels(['Normal (0)', 'Cancelado (1)'])
    ax_matriz.set_yticklabels(['Normal (0)', 'Cancelado (1)'])
    # Le ordenamos a Streamlit que muestre el gráfico en la página
    st.pyplot(fig_matriz)

# Entramos a la columna derecha
with col_der:
    # Título del gráfico derecho
    st.subheader("Curva de Costo Total vs Umbral")
    # Preparamos el lienzo del gráfico
    fig_curva, ax_curva = plt.subplots(figsize=(6, 4))
    # Dibujamos la línea azul oscura que muestra cómo cambia la plata perdida a medida que movemos el umbral de 0 a 1
    ax_curva.plot(umbrales, costos_totales, color='navy', label='Costo Total')
    # Dibujamos una línea roja punteada para marcar dónde está el umbral aburrido y clásico de 0.5
    ax_curva.axvline(x=0.5, color='red', linestyle='--', label='Umbral 0.5')
    # Dibujamos una línea verde punteada para marcar matemáticamente cuál es el mejor punto de todos (el más barato)
    ax_curva.axvline(x=umbral_empirico, color='green', linestyle='--', label=f'Óptimo ({umbral_empirico:.2f})')
    # Dibujamos una línea naranja sólida que se mueve en vivo según lo que el usuario elija en el slider
    ax_curva.axvline(x=umbral_manual, color='orange', linestyle='-', linewidth=2, label=f'Manual ({umbral_manual:.2f})')
    # Ponemos nombres a los ejes X e Y, y agregamos la leyenda para que se entiendan los colores
    ax_curva.set_xlabel('Umbral de Decisión')
    ax_curva.set_ylabel('Costo Total ($)')
    ax_curva.legend()
    # Le ordenamos a Streamlit que muestre el gráfico en la página
    st.pyplot(fig_curva)

# Finalmente, agregamos un texto destacado en la parte inferior resumiendo los descubrimientos clave
st.markdown(f"**💡 Insight del Modelo:** El Umbral Teórico matemático es **{umbral_teorico:.3f}**. El Umbral Empírico óptimo encontrado en la curva es **{umbral_empirico:.3f}**.")
