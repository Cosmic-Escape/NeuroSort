import sys
import os
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd

from src.dataset_generator import generate_dataset
from src.predictor import predict_best_algorithm
from src.benchmark import benchmark_algorithms

# ----------------------------
# Page Config & Styling
# ----------------------------
st.set_page_config(
    page_title="NeuroSort Dashboard",
    layout="wide",
)

st.markdown(
    """
    <style>
    /* Global font */
    html, body, [class*="css"]  {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }

    /* Card-like sections */
    .stCard {
        border-radius: 12px;
        border: 1px solid #E6E6E6;
        box-shadow: 2px 2px 12px rgba(0,0,0,0.05);
        padding: 15px;
        margin-bottom: 20px;
        transition: all 0.3s ease-in-out;
    }

    /* Highlight metric cards */
    .stMetric {
        transition: transform 0.3s ease-in-out, box-shadow 0.3s ease-in-out;
    }
    .stMetric:hover {
        transform: scale(1.03);
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
    }

    </style>
    """, unsafe_allow_html=True
)

# ----------------------------
# Header
# ----------------------------
st.title("NeuroSort — ML-Powered Sorting Selector")
st.markdown("<h4>Adaptive algorithm selection using machine learning</h4>", unsafe_allow_html=True)

# ----------------------------
# Sidebar: Dataset Configuration
# ----------------------------
st.sidebar.header("Dataset Configuration")
size = st.sidebar.slider("Dataset Size", 10, 6000, 500, step=10)
distribution = st.sidebar.selectbox(
    "Distribution",
    ["uniform", "normal", "exponential"]
)
run_button = st.sidebar.button("Run NeuroSort")

# ----------------------------
# Main App Logic
# ----------------------------
if run_button:
    # Dataset generation
    with st.spinner("Generating dataset..."):
        data = generate_dataset(size=size, distribution=distribution)

    st.markdown("---")

    # Placeholders for progressive updates
    prediction_placeholder = st.empty()
    metrics_placeholder = st.empty()
    results_placeholder = st.empty()
    charts_placeholder = st.empty()

    # ==========================
    # ML Prediction FIRST
    # ==========================
    ml_start = time.perf_counter()
    predicted_algo, predicted_runtimes = predict_best_algorithm(data)
    ml_end = time.perf_counter()
    ml_time = ml_end - ml_start

    with prediction_placeholder.container():
        st.markdown('<div class="stCard">', unsafe_allow_html=True)
        st.subheader("ML Prediction")
        st.write(f"**Predicted Fastest Algorithm:** {predicted_algo}")
        st.metric("ML Inference Time (s)", f"{ml_time:.6f}")
        # Convert all predicted runtimes to float
        pred_df = pd.DataFrame(
            list(predicted_runtimes.items()),
            columns=["Algorithm", "Predicted Runtime (s)"]
        )
        pred_df["Predicted Runtime (s)"] = pd.to_numeric(pred_df["Predicted Runtime (s)"], errors='coerce')

        # Sort by predicted runtime
        pred_df = pred_df.sort_values("Predicted Runtime (s)")

        # Display in Streamlit (format floats to 6 decimals)
        st.dataframe(pred_df.style.format({"Predicted Runtime (s)": "{:.6f}"}))
            
    # Benchmark AFTER
    # ==========================
    with st.spinner("Running full benchmark..."):
        bench_start = time.perf_counter()
        actual_best, timings = benchmark_algorithms(data)
        bench_end = time.perf_counter()
        benchmark_time = bench_end - bench_start

    speedup = benchmark_time / ml_time if ml_time > 0 else 0

    # Update metrics section
    with metrics_placeholder.container():
        st.markdown('<div class="stCard">', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Benchmark Time (s)", f"{benchmark_time:.6f}", help="Time taken to compute all algorithms")
        with col2:
            st.metric("Speedup vs Benchmark", f"{speedup:.2f}x", help="How much faster ML prediction is vs full evaluation")
        with col3:
            st.metric("Prediction Accuracy", "✔️" if predicted_algo == actual_best else "❌", help="Correctness of predicted fastest algorithm")
        st.markdown('</div>', unsafe_allow_html=True)

    # Show results
    with results_placeholder.container():
        st.markdown('<div class="stCard">', unsafe_allow_html=True)
        st.subheader("Validation Result")
        st.write(f"**Actual Fastest Algorithm:** {actual_best}")
        if predicted_algo == actual_best:
            st.success("Prediction Correct")
        else:
            st.error("Prediction Mismatch")
        st.markdown('</div>', unsafe_allow_html=True)

    # Predicted vs Actual Runtimes Chart
    combined_df = pd.DataFrame({
        "Algorithm": list(predicted_runtimes.keys()),
        "Predicted Runtime": list(predicted_runtimes.values()),
        "Actual Runtime": [timings[algo] for algo in predicted_runtimes.keys()]
    }).sort_values("Actual Runtime")

    with charts_placeholder.container():
        st.markdown('<div class="stCard">', unsafe_allow_html=True)
        st.subheader("Predicted vs Actual Runtimes")
        st.bar_chart(combined_df.set_index("Algorithm"))
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.info("NeuroSort predicts the optimal algorithm before exhaustive evaluation completes.")