import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fpdf import FPDF
from datetime import datetime
import seaborn as sns

# Load the COVID-19 dataset (adjust the file path as necessary)
data = pd.read_csv("./owid-covid-data.csv")

# Convert the date column to datetime type for easier plotting
data['date'] = pd.to_datetime(data['date'])

# Placeholder for gender and age data
data['gender'] = np.random.choice(['male', 'female'], size=len(data))  # Random gender data
data['age_group'] = np.random.choice(['0-18', '18-60', '60+'], size=len(data))  # Random age data

# Function to generate plots and add them to a PDF report for a specific country
def generate_country_report(country_name):
    # Filter data for the specific country
    country_data = data[data['location'] == country_name].copy()
    
    # Select relevant columns
    relevant_cols = ['date', 'total_cases', 'new_cases', 'total_deaths', 'new_deaths', 'total_vaccinations', 'gender', 'age_group']
    country_data = country_data[relevant_cols]
    
    # Initialize PDF
    pdf = FPDF()
    
    # Plot 1: Total COVID-19 Cases
    plt.figure(figsize=(10, 6))
    plt.plot(country_data['date'], country_data['total_cases'], label="Total Cases", color="blue")
    plt.title(f'Total COVID-19 Cases in {country_name}')
    plt.xlabel('Date')
    plt.ylabel('Total Cases')
    plt.grid()
    plt.legend()
    total_cases_path = f"{country_name}_total_cases.png"
    plt.savefig(total_cases_path)
    plt.close()
    add_plot_to_pdf(pdf, total_cases_path, f'Total COVID-19 Cases in {country_name}')
    
    # Plot 2: Cases by Gender
    gender_data = country_data.groupby('gender')['new_cases'].sum()
    plt.figure(figsize=(8, 5))
    gender_data.plot(kind='bar', color=['blue', 'orange'])
    plt.title(f'COVID-19 Cases by Gender in {country_name}')
    plt.xlabel('Gender')
    plt.ylabel('Total Cases')
    gender_cases_path = f"{country_name}_gender_cases.png"
    plt.savefig(gender_cases_path)
    plt.close()
    add_plot_to_pdf(pdf, gender_cases_path, f'COVID-19 Cases by Gender in {country_name}')
    
    # Plot 3: Cases by Age Group
    age_group_data = country_data.groupby('age_group')['new_cases'].sum()
    plt.figure(figsize=(8, 5))
    age_group_data.plot(kind='bar', color=['purple', 'brown', 'pink'])
    plt.title(f'COVID-19 Cases by Age Group in {country_name}')
    plt.xlabel('Age Group')
    plt.ylabel('Total Cases')
    age_group_cases_path = f"{country_name}_age_group_cases.png"
    plt.savefig(age_group_cases_path)
    plt.close()
    add_plot_to_pdf(pdf, age_group_cases_path, f'COVID-19 Cases by Age Group in {country_name}')
    
    # Save the PDF with the country name and current date/time
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    pdf_file_name = f'{country_name}_covid_report_{current_time}.pdf'
    pdf.output(pdf_file_name)
    print(f"PDF report for {country_name} saved as {pdf_file_name}")

# Function to add plot to PDF
def add_plot_to_pdf(pdf, image_path, title):
    pdf.add_page()
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(40, 10, title)
    pdf.image(image_path, x=10, y=30, w=180)

# Generate reports for India, Brazil, United States, and Thailand
for country in ["India", "Brazil", "United States", "Thailand"]:
    generate_country_report(country)
