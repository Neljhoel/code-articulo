import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import differential_evolution
from sklearn.neural_network import MLPRegressor
from pyswarms.single import GlobalBestPSO
import matplotlib.pyplot as plt

# Configuración de página
st.set_page_config(
    page_title="Software de Optimización Híbrida",
    page_icon="📊",
    layout="wide"
)

# Estilos CSS personalizados
st.markdown(
    """
    <style>
    body {
        background-color: #F5F5F5; /* Fondo claro */
    }
    .main-title {
        font-size: 48px;
        font-weight: bold;
        color: #2E86C1; /* Azul oscuro */
        text-align: center;
        margin-bottom: 20px;
    }
    .section-title {
        font-size: 24px;
        font-weight: bold;
        color: #1B4F72; /* Texto oscuro */
        margin-bottom: 10px;
    }
    .sidebar-style {
        background-color: #D6EAF8; /* Azul claro */
        padding: 15px;
        border-radius: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Título principal
st.markdown('<div class="main-title">Software de Optimización Híbrida</div>', unsafe_allow_html=True)

# Funciones de optimización
def optimize_ga_column(data_column):
    def objective_function(x):
        return np.sum((data_column - x) ** 2)
    bounds = [(0, max(data_column)) for _ in data_column]
    result = differential_evolution(objective_function, bounds, maxiter=50)
    return result.x

def optimize_ann_column(data_column):
    X = np.arange(len(data_column)).reshape(-1, 1)
    y = data_column
    model = MLPRegressor(max_iter=300, random_state=42)
    model.fit(X, y)
    predictions = model.predict(X)
    return predictions

def optimize_pso_column(data_column):
    def objective_function(x):
        return np.sum((data_column - x) ** 2)
    lb = [0] * len(data_column)
    ub = [max(data_column)] * len(data_column)
    optimizer = GlobalBestPSO(n_particles=30, dimensions=len(data_column), options={'c1': 0.5, 'c2': 0.3, 'w': 0.9})
    best_cost, best_pos = optimizer.optimize(objective_function, iters=50)
    return best_pos

# Sidebar: opciones principales
st.sidebar.title("Menú de Navegación")
st.sidebar.markdown("### Carga de datos")
uploaded_file = st.sidebar.file_uploader("Sube un archivo Excel o CSV", type=["csv", "xlsx"])

# Contenido principal
if uploaded_file:
    # Leer archivo
    if uploaded_file.name.endswith(".csv"):
        data = pd.read_csv(uploaded_file)
    else:
        data = pd.read_excel(uploaded_file)
    
    st.sidebar.markdown("### Métodos de optimización")
    method = st.sidebar.selectbox(
        "Selecciona el método:",
        ["Sin Optimización (Original)", "GA + PSO", "ANN + PSO", "GA + ACO"]
    )

    if st.sidebar.button("Ejecutar Optimización"):
        st.markdown('<div class="section-title">Datos cargados:</div>', unsafe_allow_html=True)
        st.dataframe(data, use_container_width=True)

        # Seleccionar columnas numéricas
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        input_data = data[numeric_columns].to_numpy()

        # Optimización
        original_totals = input_data.sum(axis=0)  # Totales originales por columna
        optimized_columns = []
        percentage_improvement = []

        if method != "Sin Optimización (Original)":
            for col_idx, column_name in enumerate(numeric_columns):
                column_data = input_data[:, col_idx]
                if method == "GA + PSO":
                    optimized_col = (optimize_ga_column(column_data) + optimize_pso_column(column_data)) / 2
                elif method == "ANN + PSO":
                    optimized_col = (optimize_ann_column(column_data) + optimize_pso_column(column_data)) / 2
                elif method == "GA + ACO":
                    optimized_col = (optimize_ga_column(column_data) + optimize_pso_column(column_data)) / 2

                optimized_columns.append(optimized_col)
                original_sum = column_data.sum()
                optimized_sum = optimized_col.sum()
                improvement = ((original_sum - optimized_sum) / original_sum) * 100
                percentage_improvement.append(improvement)

            # Resultados optimizados
            result = np.column_stack(optimized_columns)
            result_df = pd.DataFrame(result, columns=numeric_columns)

            st.markdown(f'<div class="section-title">Resultados optimizados con {method}:</div>', unsafe_allow_html=True)
            st.dataframe(result_df, use_container_width=True)

            # Resumen
            optimized_totals = result_df.sum(axis=0)
            summary_df = pd.DataFrame({
                "Columna": numeric_columns,
                "Total Original": original_totals,
                "Total Optimizado": optimized_totals,
                "Mejora (%)": percentage_improvement
            })
            st.markdown('<div class="section-title">Resumen de la optimización:</div>', unsafe_allow_html=True)
            st.dataframe(summary_df, use_container_width=True)

            # Resultado global
            total_original = original_totals.sum()
            total_optimized = optimized_totals.sum()
            total_improvement = ((total_original - total_optimized) / total_original) * 100

            st.markdown('<div class="section-title">Resultado Global:</div>', unsafe_allow_html=True)
            st.write(f"**Total Original:** {total_original:.2f}")
            st.write(f"**Total Optimizado:** {total_optimized:.2f}")
            st.write(f"**Mejora Global:** {total_improvement:.2f}%")

            # Gráficas
            st.markdown('<div class="section-title">Gráficas comparativas:</div>', unsafe_allow_html=True)
            for col_idx, column_name in enumerate(numeric_columns):
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(input_data[:, col_idx], label="Original", color="blue", marker="o")
                ax.plot(result[:, col_idx], label="Optimizado", color="green", marker="x")
                ax.set_title(f"Optimización de {column_name}")
                ax.legend()
                st.pyplot(fig)
        else:
            st.markdown('<div class="section-title">Datos originales:</div>', unsafe_allow_html=True)
            st.dataframe(data[numeric_columns], use_container_width=True)
else:
    st.markdown('<div class="section-title">Por favor, carga un archivo de datos para continuar.</div>', unsafe_allow_html=True)
