# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.special import gamma
# from scipy.integrate import odeint

# # Function to compute the Caputo fractional derivative
# def caputo_derivative(f, t, alpha, order=1):
#     # Fractional derivative using Caputo definition
#     def integrand(tau):
#         return (f(tau) / (t - tau) ** (alpha + 1)) / gamma(1 - alpha)
    
#     t_values = np.linspace(0, t, 100)
#     integral = np.trapz([integrand(tau) for tau in t_values], t_values)
#     return integral

# # Define the COVID-19 infection model
# def covid_model(y, t, alpha, beta, gamma_rate, delta):
#     S, I, R = y
#     dS_dt = -beta * S * I
#     dI_dt = beta * S * I - gamma_rate * I
#     dR_dt = gamma_rate * I

#     # Apply Caputo fractional derivative to the differential equations
#     caputo_I = caputo_derivative(lambda tau: I, t, alpha)
    
#     return [dS_dt, dI_dt + caputo_I, dR_dt]

# # Initial conditions
# S0 = 0.99  # Initial susceptible population (99%)
# I0 = 0.01  # Initial infected population (1%)
# R0 = 0.0   # Initial recovered population (0%)
# y0 = [S0, I0, R0]

# # Time points where we want to solve the system
# t = np.linspace(0, 100, 1000)

# # Model parameters
# alpha = 0.7  # Fractional order for Caputo derivative
# beta = 0.3   # Transmission rate
# gamma_rate = 0.1  # Recovery rate
# delta = 0.05  # Death rate

# # Solve the system using odeint
# solution = odeint(covid_model, y0, t, args=(alpha, beta, gamma_rate, delta))

# # Extract solutions
# S, I, R = solution.T

# # Plot results
# plt.figure(figsize=(10, 6))
# plt.plot(t, S, label='Susceptible', color='blue')
# plt.plot(t, I, label='Infected', color='red')
# plt.plot(t, R, label='Recovered', color='green')
# plt.title('COVID-19 Propagation Model with Caputo Fractional Derivative')
# plt.xlabel('Time')
# plt.ylabel('Population Proportion')
# plt.legend()
# plt.grid()
# plt.show()

from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# SIR model differential equations
def sir_model(y, t, beta, gamma):
    S, I, R = y
    dS_dt = -beta * S * I
    dI_dt = beta * S * I - gamma * I
    dR_dt = gamma * I
    return [dS_dt, dI_dt, dR_dt]

# Solve the SIR model
def solve_sir(S0, I0, R0, beta, gamma, t):
    y0 = [S0, I0, R0]
    return odeint(sir_model, y0, t, args=(beta, gamma))

# Compare SIR model to Caputo Derivative and calculate MSE
def compare_models(country):
    # Load country data
    country_data = get_country_data(country)
    
    # Extract total cases (actual data)
    total_cases = country_data['total_cases'].fillna(0).values
    time_days = np.linspace(0, len(total_cases), len(total_cases))
    
    # Normalize data for SIR model
    N = country_data['population'].iloc[0]  # Population size
    I0 = total_cases[0]  # Initial infected
    R0 = 0  # Initial recovered
    S0 = N - I0  # Initial susceptible
    
    # Parameters for SIR model (to be adjusted for fitting)
    beta = 0.3  # Infection rate (can be optimized)
    gamma = 0.1  # Recovery rate (can be optimized)
    
    # Solve the SIR model
    t = np.linspace(0, len(total_cases), len(total_cases))
    SIR_solution = solve_sir(S0, I0, R0, beta, gamma, t)
    
    # Extract infected population from SIR solution
    SIR_infected = SIR_solution[:, 1]
    
    # Caputo derivative solution (already calculated)
    caputo_solution = caputo_derivatives[country]
    
    # Calculate MSE for SIR model
    mse_sir = mean_squared_error(total_cases, SIR_infected)
    
    # Calculate MSE for Caputo Derivative model
    mse_caputo = mean_squared_error(total_cases, caputo_solution)
    
    # Print the MSE for both models
    print(f"MSE for SIR Model (Country: {country}): {mse_sir}")
    print(f"MSE for Caputo Derivative Model (Country: {country}): {mse_caputo}")
    
    # Plot the comparison
    plt.figure(figsize=(10, 6))
    plt.plot(country_data['date'], total_cases, label='Actual Data', color='black', linestyle='--')
    plt.plot(country_data['date'], SIR_infected, label='SIR Model Infected', color='green')
    plt.plot(country_data['date'], caputo_solution, label=f'Caputo Derivative (alpha={alpha})', color='blue')
    plt.title(f'COVID-19 Cases Comparison for {country}')
    plt.xlabel('Date')
    plt.ylabel('Number of Cases')
    plt.legend()
    plt.grid()
    plt.show()

# Example: Compare for India
compare_models('India')