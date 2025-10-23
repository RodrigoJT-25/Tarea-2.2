import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression

# Título y descripción
st.title("Predicción con Regresión Lineal Simple")
st.write("Aplicación interactiva para entrenar un modelo de regresión lineal y visualizar las predicciones.")
st.write("Selecciona la variable dependiente (Y) y la variable independiente (X).")

# Cargar datos
st.subheader("Cargar datos")
uploaded_file = st.file_uploader("Sube un archivo CSV con tus datos", type=["csv"])

if uploaded_file is not None:
    # Leer datos
    data = pd.read_csv(uploaded_file)
    st.write("Vista previa de los datos:")
    st.dataframe(data.head())

    # Seleccionar columnas directamente
    x_col = st.selectbox("Selecciona la variable independiente (X)", data.columns, index=0)
    y_col = st.selectbox("Selecciona la variable dependiente (Y)", data.columns, index=1 if len(data.columns) > 1 else 0)

    if x_col == y_col:
        st.error("X y Y deben ser columnas distintas.")
        st.stop()

    # Entrenamiento del modelo
    X = data[[x_col]].values  # 2D
    y = data[y_col].values    # 1D

    model = LinearRegression()
    model.fit(X, y)
    y_hat = model.predict(X)

    # Parámetros del modelo
    m = float(model.coef_[0])
    b = float(model.intercept_)
    r2 = float(r2_score(y, y_hat))

    st.success("Modelo entrenado correctamente")

    # Mostrar ecuación
    st.markdown("Ecuación del modelo:")
    st.latex(r"Y = %.2fX \; + \; %.2f" % (m, b))

    # Calcular y mostrar el R^2
    st.markdown("Mostrar el R^2")
    st.caption("Coeficiente de determinación (R²)")
    st.metric(label="R²", value=f"{r2:.4f}")
    st.latex(r"R^2 = %.4f" % r2)

    # Predicción interactiva
    st.markdown("Realiza una predicción")
    x_min, x_max = float(np.min(X)), float(np.max(X))
    rango = max(1e-6, (x_max - x_min))
    entrada = st.number_input(
        f"Introduce un valor para {x_col}:",
        value=round(x_min + rango/2, 2),
        step=0.1,
        format="%.2f"
    )
    pred = float(model.predict(np.array([[entrada]]))[0])
    st.info(f"🔷 **Predicción para {x_col} = {entrada:.2f}: {y_col} ≈ {pred:.2f}**")

    # Visualización del modelo
    st.markdown("Visualización del modelo")

    x_line = np.linspace(x_min, x_max, 200).reshape(-1, 1)
    y_line = model.predict(x_line)

    fig, ax = plt.subplots()
    ax.scatter(X, y, label="Datos reales", alpha=0.9)
    ax.plot(x_line, y_line, label="Línea de regresión")
    ax.scatter([entrada], [pred], s=90, label="Predicción")
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.legend(loc="upper left")
    st.pyplot(fig)

else:
    st.info("Sube un archivo CSV para continuar.")