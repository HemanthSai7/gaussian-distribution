import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import minimize
import pandas as pd

# Set page config
st.set_page_config(page_title="Gaussian Distribution & Overfitting Demo", layout="wide")

st.title("Gaussian Distribution & Overfitting in ML")
st.markdown("Interactive demonstration of concepts from PRML Chapter 1")

# Sidebar for navigation
page = st.sidebar.selectbox("Select Demo", 
    ["Gaussian Distribution Basics", 
     "Maximum Likelihood Bias", 
     "Polynomial Curve Fitting",
     "Probabilistic Curve Fitting",
     "Regularized Curve Fitting"])

if page == "Gaussian Distribution Basics":
    st.header("1.2.4 The Gaussian Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Parameters")
        mu = st.slider("Mean (μ)", -5.0, 5.0, 0.0, 0.1)
        sigma = st.slider("Standard Deviation (σ)", 0.1, 5.0, 1.0, 0.1)
        
        st.latex(r"N(x|\mu, \sigma^2) = \frac{1}{(2\pi\sigma^2)^{1/2}} \exp\left\{-\frac{1}{2\sigma^2}(x-\mu)^2\right\}")
    
    with col2:
        st.subheader("Gaussian Distribution Plot")
        x = np.linspace(mu - 4*sigma, mu + 4*sigma, 1000)
        y = norm.pdf(x, mu, sigma)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(x, y, 'b-', linewidth=2, label=f'N({mu:.1f}, {sigma:.1f}²)')
        ax.fill_between(x, y, alpha=0.3)
        ax.axvline(mu, color='r', linestyle='--', label=f'Mean = {mu:.1f}')
        ax.axvline(mu - sigma, color='g', linestyle='--', alpha=0.5)
        ax.axvline(mu + sigma, color='g', linestyle='--', alpha=0.5, label=f'±σ = ±{sigma:.1f}')
        ax.set_xlabel('x')
        ax.set_ylabel('p(x)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

elif page == "Maximum Likelihood Bias":
    st.header("Maximum Likelihood Bias in Variance Estimation")
    st.markdown("This demonstrates how ML systematically underestimates the true variance")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Simulation Parameters")
        true_mu = st.slider("True Mean", -2.0, 2.0, 0.0, 0.1)
        true_sigma = st.slider("True Std Dev", 0.5, 3.0, 1.0, 0.1)
        n_samples = st.slider("Number of Samples (N)", 2, 100, 10, 1)
        n_experiments = st.slider("Number of Experiments", 100, 1000, 500, 100)
        
        if st.button("Run Simulation"):
            # Run multiple experiments
            ml_means = []
            ml_vars = []
            unbiased_vars = []
            
            for _ in range(n_experiments):
                # Generate random samples
                samples = np.random.normal(true_mu, true_sigma, n_samples)
                
                # ML estimates
                ml_mean = np.mean(samples)
                ml_var = np.var(samples, ddof=0)  # ML estimate
                unbiased_var = np.var(samples, ddof=1)  # Unbiased estimate
                
                ml_means.append(ml_mean)
                ml_vars.append(ml_var)
                unbiased_vars.append(unbiased_var)
            
            # Store results in session state
            st.session_state.ml_means = ml_means
            st.session_state.ml_vars = ml_vars
            st.session_state.unbiased_vars = unbiased_vars
            st.session_state.true_var = true_sigma**2
            st.session_state.n_samples_used = n_samples
        
        # Results section below parameters
        if 'ml_vars' in st.session_state:
            st.markdown("---")  # Separator line
            st.subheader("Results")
            
            # Calculate averages
            avg_ml_var = np.mean(st.session_state.ml_vars)
            avg_unbiased_var = np.mean(st.session_state.unbiased_vars)
            true_var = st.session_state.true_var
            n_samples_used = st.session_state.n_samples_used
            expected_ml_var = (n_samples_used - 1) / n_samples_used * true_var
            
            # Display metrics
            col3, col4, col5, col6 = st.columns(4)
            with col3:
                st.metric("Average ML Mean", f"{np.mean(st.session_state.ml_means):.4f}")
            with col4:
                st.metric("Average Unbiased Mean", f"{np.mean(st.session_state.unbiased_vars):.4f}")
            with col5:
                st.metric("True Mean", f"{true_mu:.4f}")
            with col6:
                st.metric("Expected ML Variance", f"{expected_ml_var:.4f}",
                         f"{(expected_ml_var - true_var) / true_var * 100:.1f}%")
            
            # Bias factor
            st.info(f"Bias Factor: (N-1)/N = {n_samples_used-1}/{n_samples_used} = {(n_samples_used-1)/n_samples_used:.3f}")
    
    with col2:
        if 'ml_vars' in st.session_state:
            st.subheader("Variance Distribution")
            
            # Get values for plotting
            true_var = st.session_state.true_var
            n_samples_used = st.session_state.n_samples_used
            expected_ml_var = (n_samples_used - 1) / n_samples_used * true_var
            
            # Histogram
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.hist(st.session_state.ml_vars, bins=30, alpha=0.5, label='ML Variance', density=True)
            ax.hist(st.session_state.unbiased_vars, bins=30, alpha=0.5, label='Unbiased Variance', density=True)
            ax.axvline(true_var, color='r', linestyle='--', linewidth=2, label='True Variance')
            ax.axvline(expected_ml_var, color='g', linestyle='--', linewidth=2, label='Expected ML Variance')
            ax.set_xlabel('Variance Estimate', fontsize=12)
            ax.set_ylabel('Density', fontsize=12)
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3)
            ax.set_title(f'Distribution of Variance Estimates (N={n_samples_used})', fontsize=14)
            st.pyplot(fig)

elif page == "Polynomial Curve Fitting":
    st.header("Polynomial Curve Fitting and Overfitting")
    
    # Generate true function
    def true_function(x):
        return np.sin(2 * np.pi * x)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Parameters")
        n_data_points = st.slider("Number of Data Points", 5, 50, 15, 1)
        noise_level = st.slider("Noise Level", 0.0, 0.5, 0.2, 0.05)
        polynomial_degree = st.slider("Polynomial Degree (M)", 0, 15, 3, 1)
        
        if st.button("Generate New Data"):
            np.random.seed(None)  # Random seed
            x_train = np.random.uniform(0, 1, n_data_points)
            y_train = true_function(x_train) + np.random.normal(0, noise_level, n_data_points)
            st.session_state.x_train = x_train
            st.session_state.y_train = y_train
    
    # Initialize data if not exists
    if 'x_train' not in st.session_state:
        np.random.seed(42)
        x_train = np.random.uniform(0, 1, n_data_points)
        y_train = true_function(x_train) + np.random.normal(0, noise_level, n_data_points)
        st.session_state.x_train = x_train
        st.session_state.y_train = y_train
    
    with col2:
        st.subheader("Polynomial Fit")
        
        # Fit polynomial
        X_train = np.vander(st.session_state.x_train, polynomial_degree + 1, increasing=True)
        w = np.linalg.lstsq(X_train, st.session_state.y_train, rcond=None)[0]
        
        # Plot
        x_plot = np.linspace(0, 1, 200)
        X_plot = np.vander(x_plot, polynomial_degree + 1, increasing=True)
        y_pred = X_plot @ w
        y_true = true_function(x_plot)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(x_plot, y_true, 'g-', linewidth=2, label='True Function')
        ax.plot(x_plot, y_pred, 'r-', linewidth=2, label=f'Polynomial (M={polynomial_degree})')
        ax.scatter(st.session_state.x_train, st.session_state.y_train, 
                  c='blue', s=50, alpha=0.8, edgecolors='black', label='Training Data')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_ylim(-1.5, 1.5)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_title(f'Polynomial Degree M = {polynomial_degree}')
        st.pyplot(fig)
        
        # Calculate training error
        y_train_pred = X_train @ w
        train_rmse = np.sqrt(np.mean((st.session_state.y_train - y_train_pred)**2))
        st.metric("Training RMSE", f"{train_rmse:.4f}")

elif page == "Probabilistic Curve Fitting":
    st.header("Probabilistic View of Curve Fitting")
    st.latex(r"p(t|x,\mathbf{w},\beta) = N(t|y(x,\mathbf{w}), \beta^{-1})")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Parameters")
        n_data_points = st.slider("Number of Data Points", 5, 50, 20, 1)
        true_noise = st.slider("True Noise (σ)", 0.1, 0.5, 0.2, 0.05)
        polynomial_degree = st.slider("Polynomial Degree", 0, 9, 3, 1)
        show_uncertainty = st.checkbox("Show Predictive Distribution", True)
        
        if st.button("Generate Data"):
            np.random.seed(None)
            x_train = np.random.uniform(0, 1, n_data_points)
            y_train = np.sin(2 * np.pi * x_train) + np.random.normal(0, true_noise, n_data_points)
            st.session_state.prob_x_train = x_train
            st.session_state.prob_y_train = y_train
    
    # Initialize data
    if 'prob_x_train' not in st.session_state:
        np.random.seed(42)
        x_train = np.random.uniform(0, 1, n_data_points)
        y_train = np.sin(2 * np.pi * x_train) + np.random.normal(0, true_noise, n_data_points)
        st.session_state.prob_x_train = x_train
        st.session_state.prob_y_train = y_train
    
    with col2:
        st.subheader("Maximum Likelihood Fit")
        
        # Fit polynomial and estimate noise
        X_train = np.vander(st.session_state.prob_x_train, polynomial_degree + 1, increasing=True)
        w_ml = np.linalg.lstsq(X_train, st.session_state.prob_y_train, rcond=None)[0]
        
        # Estimate noise variance (beta^-1)
        y_train_pred = X_train @ w_ml
        residuals = st.session_state.prob_y_train - y_train_pred
        sigma_ml = np.sqrt(np.mean(residuals**2))
        beta_ml = 1 / (sigma_ml**2)
        
        # Plot
        x_plot = np.linspace(0, 1, 200)
        X_plot = np.vander(x_plot, polynomial_degree + 1, increasing=True)
        y_mean = X_plot @ w_ml
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot uncertainty bands if requested
        if show_uncertainty:
            y_std = np.sqrt(1 / beta_ml)
            ax.fill_between(x_plot, y_mean - 2*y_std, y_mean + 2*y_std, 
                           alpha=0.3, color='red', label='±2σ predictive')
        
        ax.plot(x_plot, np.sin(2 * np.pi * x_plot), 'g-', linewidth=2, label='True Function')
        ax.plot(x_plot, y_mean, 'r-', linewidth=2, label=f'ML Fit (M={polynomial_degree})')
        ax.scatter(st.session_state.prob_x_train, st.session_state.prob_y_train, 
                  c='blue', s=50, alpha=0.8, edgecolors='black', label='Training Data')
        
        ax.set_xlabel('x')
        ax.set_ylabel('t')
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        
        # Display estimated parameters
        col3, col4 = st.columns(2)
        with col3:
            st.metric("ML Noise Estimate (σ)", f"{sigma_ml:.3f}")
        with col4:
            st.metric("True Noise (σ)", f"{true_noise:.3f}")

elif page == "Regularized Curve Fitting":
    st.header("Regularized Curve Fitting (MAP Estimation)")
    st.latex(r"E(\mathbf{w}) = \frac{\beta}{2}\sum_{n=1}^{N}\{y(x_n,\mathbf{w})-t_n\}^2 + \frac{\alpha}{2}\mathbf{w}^T\mathbf{w}")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Parameters")
        n_data_points = st.slider("Data Points", 10, 50, 15, 1)
        noise_level = st.slider("Noise", 0.1, 0.5, 0.3, 0.05)
        polynomial_degree = st.slider("Degree (M)", 0, 15, 9, 1)
        log_lambda = st.slider("log₁₀(λ)", -8.0, 2.0, -3.0, 0.5)
        regularization = 10**log_lambda
        
        if st.button("New Data"):
            np.random.seed(None)
            x_train = np.random.uniform(0, 1, n_data_points)
            y_train = np.sin(2 * np.pi * x_train) + np.random.normal(0, noise_level, n_data_points)
            st.session_state.reg_x_train = x_train
            st.session_state.reg_y_train = y_train
    
    # Initialize
    if 'reg_x_train' not in st.session_state:
        np.random.seed(42)
        x_train = np.random.uniform(0, 1, n_data_points)
        y_train = np.sin(2 * np.pi * x_train) + np.random.normal(0, noise_level, n_data_points)
        st.session_state.reg_x_train = x_train
        st.session_state.reg_y_train = y_train
    
    with col2:
        st.subheader("Regularized Fit")
        
        # Fit with regularization
        X_train = np.vander(st.session_state.reg_x_train, polynomial_degree + 1, increasing=True)
        
        # Ridge regression (L2 regularization)
        XtX = X_train.T @ X_train
        Xty = X_train.T @ st.session_state.reg_y_train
        w_reg = np.linalg.solve(XtX + regularization * np.eye(polynomial_degree + 1), Xty)
        
        # Plot
        x_plot = np.linspace(0, 1, 200)
        X_plot = np.vander(x_plot, polynomial_degree + 1, increasing=True)
        y_pred = X_plot @ w_reg
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(x_plot, np.sin(2 * np.pi * x_plot), 'g-', linewidth=2, label='True Function')
        ax.plot(x_plot, y_pred, 'r-', linewidth=2, label=f'Regularized (λ={regularization:.1e})')
        ax.scatter(st.session_state.reg_x_train, st.session_state.reg_y_train, 
                  c='blue', s=50, alpha=0.8, edgecolors='black', label='Training Data')
        ax.set_xlabel('x')
        ax.set_ylabel('t')
        ax.set_ylim(-1.5, 1.5)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_title(f'M = {polynomial_degree}, λ = {regularization:.1e}')
        st.pyplot(fig)
        
        # Metrics
        train_pred = X_train @ w_reg
        train_rmse = np.sqrt(np.mean((st.session_state.reg_y_train - train_pred)**2))
        weight_norm = np.linalg.norm(w_reg)
        
        col3, col4 = st.columns(2)
        with col3:
            st.metric("Training RMSE", f"{train_rmse:.4f}")
        with col4:
            st.metric("||w||²", f"{weight_norm:.2f}")

# Add information footer
st.markdown("---")
st.markdown("### Key Concepts Demonstrated:")
st.markdown("""
- **Gaussian Distribution**: Fundamental probability distribution with mean μ and variance σ²
- **Maximum Likelihood Bias**: ML estimation systematically underestimates variance by factor (N-1)/N
- **Overfitting**: High-degree polynomials fit training data perfectly but generalize poorly
- **Probabilistic Curve Fitting**: View regression as estimating conditional distribution p(t|x)
- **Regularization**: Adding penalty term α||w||² prevents overfitting (equivalent to MAP with Gaussian prior)
""")
