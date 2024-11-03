
# Total cases of each country wali plot ##
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fpdf import FPDF
from datetime import datetime
from scipy.special import gamma
from scipy.integrate import odeint  # <-- Import this to use odeint for SIR model
from sklearn.metrics import mean_squared_error  # Import to calculate MSE

# Load and preprocess data
data = pd.read_csv("./owid-covid-data.csv")

# List of countries to analyze
countries = ['India', 'Brazil', 'United States', 'Thailand']

# Function to filter and process data for a given country
def get_country_data(country):
    country_data = data[data['location'] == country].copy()
    country_data['date'] = pd.to_datetime(country_data['date'])
    return country_data

# Define Caputo Fractional Derivative function
def caputo_derivative(y, t, alpha):
    h = t[1] - t[0]  # Assuming uniform time step
    result = np.zeros(len(y))
    
    for i in range(1, len(t)):
        sum_term = 0
        for k in range(0, i):
            sum_term += (y[k] - y[i]) / (t[i] - t[k])**(1 - alpha)
        result[i] = (h**alpha) / gamma(2 - alpha) * sum_term
    return result

# Define alpha value for Caputo Derivative
alpha = 0.4

# Dictionary to store caputo derivatives for all countries
caputo_derivatives = {}

# Process each country
for country in countries:
    country_data = get_country_data(country)
    
    # For Total Cases
    total_cases = country_data['total_cases'].fillna(0).values
    time_days = np.linspace(0, len(total_cases), len(total_cases))
    caputo_derivatives[country] = caputo_derivative(total_cases, time_days, alpha)

    # Plot for Total Cases (Caputo Derivative)
    plt.figure(figsize=(10, 6))
    plt.plot(country_data['date'], caputo_derivatives[country], label=f"Caputo Derivative (alpha={alpha})", color="blue")
    plt.title(f'Caputo Derivative for Total COVID-19 Cases in {country}')
    plt.xlabel('Date')
    plt.ylabel('Fractional Derivative of Cases')
    plt.grid()
    plt.legend()
    plt.savefig(f'caputo_total_cases_{country}.png')
    plt.close()

    # Gender-wise COVID-19 Cases Plot (Assuming columns for gender-specific cases exist)
    if 'male_cases' in country_data.columns and 'female_cases' in country_data.columns:
        plt.figure(figsize=(10, 6))
        plt.plot(country_data['date'], country_data['male_cases'].fillna(0), label="Male Cases", color="blue")
        plt.plot(country_data['date'], country_data['female_cases'].fillna(0), label="Female Cases", color="red")
        plt.title(f'COVID-19 Cases by Gender in {country}')
        plt.xlabel('Date')
        plt.ylabel('Number of Cases')
        plt.grid()
        plt.legend()
        plt.savefig(f'gender_cases_{country}.png')
        plt.close()

    # Age-wise COVID-19 Cases Plot (Assuming age-specific columns exist)
    age_columns = [col for col in country_data.columns if 'age' in col]
    if age_columns:
        plt.figure(figsize=(10, 6))
        for col in age_columns:
            plt.plot(country_data['date'], country_data[col].fillna(0), label=col)
        plt.title(f'COVID-19 Cases by Age Group in {country}')
        plt.xlabel('Date')
        plt.ylabel('Number of Cases')
        plt.grid()
        plt.legend()
        plt.savefig(f'age_cases_{country}.png')
        plt.close()

# Comparison Plot: Total COVID-19 Cases for All Countries
plt.figure(figsize=(12, 8))
for country in countries:
    country_data = get_country_data(country)
    plt.plot(country_data['date'], caputo_derivatives[country], label=f'{country} (Caputo Derivative)')
plt.title('Caputo Derivative for Total COVID-19 Cases in India, Brazil, USA, and Thailand')
plt.xlabel('Date')
plt.ylabel('Fractional Derivative of Cases')
plt.grid()
plt.legend()
plt.savefig('caputo_total_cases_comparison.png')
plt.close()

# Create PDF report
pdf = FPDF()

# Add a cover page
pdf.add_page()
pdf.set_font('Arial', 'B', 16)
pdf.cell(200, 10, "COVID-19: Caputo Derivative Analysis for Multiple Countries", ln=True, align='C')

pdf.set_font('Arial', '', 12)
pdf.ln(10)
pdf.multi_cell(0, 10, ("This report analyzes the COVID-19 cases in India, Brazil, USA, and Thailand using a Caputo "
                       "fractional derivative model. The analysis includes total cases modeled using fractional derivatives "
                       "to account for memory effects in the system."))

# Add a section for Caputo Derivative Data Analysis
pdf.add_page()
pdf.set_font('Arial', 'B', 14)
pdf.cell(200, 10, "1. Caputo Derivative Data Analysis for Total Cases", ln=True)

pdf.set_font('Arial', '', 12)
pdf.ln(10)
pdf.multi_cell(0, 10, "The following figures represent the Caputo derivative analysis of COVID-19 total cases for India, Brazil, USA, and Thailand.")

# Function to add plot to PDF
def add_plot_to_pdf(pdf, image_path, title):
    pdf.add_page()
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(40, 10, title)
    pdf.ln(10)
    pdf.image(image_path, x=10, y=30, w=180)

# Add country-specific plots to PDF
for country in countries:
    add_plot_to_pdf(pdf, f'caputo_total_cases_{country}.png', f'Caputo Derivative for Total COVID-19 Cases in {country}')
    
    # Gender-wise plot (if generated)
    if 'male_cases' in get_country_data(country).columns and 'female_cases' in get_country_data(country).columns:
        add_plot_to_pdf(pdf, f'gender_cases_{country}.png', f'Gender-wise COVID-19 Cases in {country}')
    
    # Age-wise plot (if generated)
    age_columns = [col for col in get_country_data(country).columns if 'age' in col]
    if age_columns:
        add_plot_to_pdf(pdf, f'age_cases_{country}.png', f'Age-wise COVID-19 Cases in {country}')

# Add comparison plot for all countries
add_plot_to_pdf(pdf, 'caputo_total_cases_comparison.png', 'Comparison of Caputo Derivative for Total COVID-19 Cases')

# Save the PDF with the current date and time
current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
pdf_file_name = f'covid_caputo_report_comparison_{current_time}.pdf'
pdf.output(pdf_file_name)

print(f"PDF report saved as {pdf_file_name}")


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
compare_models('United States')
compare_models('Brazil')
compare_models('Thailand')




