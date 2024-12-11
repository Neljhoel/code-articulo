import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import differential_evolution
from sklearn.neural_network import MLPRegressor
from pyswarms.single import GlobalBestPSO
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

# Configuración de página
st.set_page_config(
    page_title="Optimización Híbrida - Supermercados",
    page_icon="📊",
    layout="wide"
)

# Estilos CSS
st.markdown(
    """
    <style>
    body {
        background-color: #E74C3C; /* Fondo rojo */
    }
    .main-title {
        font-size: 40px;
        font-weight: bold;
        color: #FFFFFF; /* Título en blanco */
        text-align: center;
        margin-bottom: 20px;
    }
    .section-title {
        font-size: 22px;
        font-weight: bold;
        color: #FFFFFF; /* Subtítulos en blanco */
        margin-bottom: 15px;
    }
    .result-box {
        border: 2px solid #16A085;
        border-radius: 10px;
        padding: 15px;
        background-color: #ECF0F1;
    }
    </style>
    """, unsafe_allow_html=True
)

# Funciones de optimización
def optimize_ga(data):
    def objective_function(x):
        return np.sum((data - x) ** 2)
    bounds = [(0, max(data)) for _ in data]
    result = differential_evolution(objective_function, bounds, maxiter=50)
    return result.x

def optimize_ann(data):
    X = np.arange(len(data)).reshape(-1, 1)
    y = data
    model = MLPRegressor(max_iter=300, random_state=42)
    model.fit(X, y)
    predictions = model.predict(X)
    return predictions

def optimize_pso(data):
    def objective_function(x):
        return np.sum((data - x) ** 2)
    lb = [0] * len(data)
    ub = [max(data)] * len(data)
    optimizer = GlobalBestPSO(n_particles=30, dimensions=len(data), options={'c1': 0.5, 'c2': 0.3, 'w': 0.9})
    best_cost, best_pos = optimizer.optimize(objective_function, iters=50)
    return best_pos

def optimize_aco(data):
    n_variables = len(data)
    n_ants = 10
    max_iters = 50
    pheromones = np.ones((n_variables, n_ants))
    best_solution = None
    best_cost = float('inf')
    for _ in range(max_iters):
        solutions = []
        costs = []
        for _ in range(n_ants):
            solution = np.random.uniform(0, max(data), n_variables)
            cost = np.sum((data - solution) ** 2)
            solutions.append(solution)
            costs.append(cost)
        best_idx = np.argmin(costs)
        best_solution = solutions[best_idx]
        best_cost = min(best_cost, costs[best_idx])
        pheromones += np.outer(best_solution, 1 / np.array(costs))
    return best_solution

def hybrid_ga_pso(data):
    ga_result = optimize_ga(data)
    pso_result = optimize_pso(data)
    return (ga_result + pso_result) / 2

def hybrid_ann_pso(data):
    ann_result = optimize_ann(data)
    pso_result = optimize_pso(data)
    return (ann_result + pso_result) / 2

def hybrid_ga_aco(data):
    ga_result = optimize_ga(data)
    aco_result = optimize_aco(data)
    return (ga_result + aco_result) / 2

# Título principal
st.markdown('<div class="main-title">Optimización Híbrida - Gestión de Supermercados</div>', unsafe_allow_html=True)

# Subir archivo
st.markdown('<div class="section-title">1. Subir archivo de datos</div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("Sube un archivo Excel o CSV", type=["csv", "xlsx"])

if uploaded_file:
    # Leer archivo
    if uploaded_file.name.endswith(".csv"):
        data = pd.read_csv(uploaded_file)
    else:
        data = pd.read_excel(uploaded_file)
    
    st.markdown('<div class="section-title">2. Datos cargados</div>', unsafe_allow_html=True)
    st.write(data)

    # Seleccionar solo columnas numéricas
    numeric_columns = data.select_dtypes(include=[np.number]).columns
    input_data = data[numeric_columns].to_numpy()

    # Dividir página en dos columnas
    col1, col2 = st.columns([1, 3])

    with col1:
        st.markdown('<div class="section-title">3. Seleccionar optimización</div>', unsafe_allow_html=True)
        selected_method = st.selectbox(
            "Selecciona el método de optimización:",
            ["Mostrar resultado original", "GA + PSO", "ANN + PSO", "GA + ACO"]
        )
        
        if st.button("Ejecutar"):
            if selected_method == "Mostrar resultado original":
                st.markdown('<div class="section-title">Resultados originales</div>', unsafe_allow_html=True)
                st.write(f"Suma total original: {np.sum(input_data)}")
            else:
                if selected_method == "GA + PSO":
                    result = np.apply_along_axis(hybrid_ga_pso, 1, input_data)
                elif selected_method == "ANN + PSO":
                    result = np.apply_along_axis(hybrid_ann_pso, 1, input_data)
                elif selected_method == "GA + ACO":
                    result = np.apply_along_axis(hybrid_ga_aco, 1, input_data)

                st.markdown(f'<div class="section-title">Resultados: {selected_method}</div>', unsafe_allow_html=True)
                st.write(f"Suma total optimizada: {np.sum(result)}")
                st.write("Resultados por variable:")
                st.dataframe(pd.DataFrame(result, columns=numeric_columns))

    with col2:
        st.markdown('<div class="section-title">4. Gráfica de resultados</div>', unsafe_allow_html=True)
        if uploaded_file:
            plt.figure(figsize=(12, 6))
            plt.plot(np.sum(input_data, axis=1), label="Original", marker='o')
            if selected_method != "Mostrar resultado original":
                plt.plot(np.sum(result, axis=1), label="Optimizado", marker='x')
            plt.legend()
            plt.title("Comparación de resultados")
            st.pyplot(plt)
