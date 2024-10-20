# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from fpdf import FPDF
# from datetime import datetime
# import seaborn as sns
# from scipy.special import gamma

# # Load and preprocess data
# # url = "https://covid.ourworldindata.org/data/owid-covid-data.csv"
# data = pd.read_csv("./owid-covid-data.csv")

# # Filter data for India
# india_data = data[data['location'] == 'India'].copy()

# # Convert the date column to datetime type
# india_data['date'] = pd.to_datetime(india_data['date'])

# # Preprocessing Data (Assume gender and age distribution are available or calculated)
# india_data['gender'] = np.random.choice(['male', 'female'], size=len(india_data))  # Placeholder
# india_data['age_group'] = np.random.choice(['0-18', '18-60', '60+'], size=len(india_data))  # Placeholder

# # Define Caputo Fractional Derivative function
# def caputo_derivative(y, t, alpha):
#     """
#     Approximate Caputo fractional derivative of order alpha.
#     :param y: Function values at times t
#     :param t: Time values
#     :param alpha: Fractional order (0 < alpha <= 1)
#     """
#     h = t[1] - t[0]  # Assuming uniform time step
#     result = np.zeros(len(y))
    
#     for i in range(1, len(t)):
#         sum_term = 0
#         for k in range(0, i):
#             sum_term += (y[k] - y[i]) / (t[i] - t[k])**(1 - alpha)
#         result[i] = (h**alpha) / gamma(2 - alpha) * sum_term
#     return result

# # Applying the Caputo Derivative to model COVID-19 cases in India
# total_cases = india_data['total_cases'].fillna(0).values
# time_days = np.linspace(0, len(total_cases), len(total_cases))

# # Simulate the Caputo derivative with alpha = 0.9 (for example)
# alpha = 0.9
# caputo_derivative_cases = caputo_derivative(total_cases, time_days, alpha)

# # Plot 1: Total COVID-19 Cases
# plt.figure(figsize=(10, 6))
# plt.plot(india_data['date'], india_data['total_cases'], label="Total Cases", color="blue")
# plt.title('Total COVID-19 Cases in India')
# plt.xlabel('Date')
# plt.ylabel('Total Cases')
# plt.grid()
# plt.legend()
# plt.savefig('total_cases.png')
# plt.close()

# # Plot 2: Total Vaccinations
# plt.figure(figsize=(10, 6))
# plt.plot(india_data['date'], india_data['total_vaccinations'], label="Total Vaccinations", color="green")
# plt.title('Total COVID-19 Vaccinations in India')
# plt.xlabel('Date')
# plt.ylabel('Total Vaccinations')
# plt.grid()
# plt.legend()
# plt.savefig('total_vaccinations.png')
# plt.close()

# # Plot 3: Gender-based cases
# gender_data = india_data.groupby('gender')['new_cases'].sum()
# plt.figure(figsize=(8, 5))
# gender_data.plot(kind='bar', color=['blue', 'orange'])
# plt.title('COVID-19 Cases by Gender in India')
# plt.xlabel('Gender')
# plt.ylabel('Total Cases')
# plt.savefig('gender_cases.png')
# plt.close()

# # Plot 4: age_group based cases
# age_group_data = india_data.groupby('age_group')['new_cases'].sum()
# plt.figure(figsize=(8, 5))
# age_group_data.plot(kind='bar', color=['purple', 'brown', 'pink'])
# plt.title('COVID-19 Cases by Age Group in India')
# plt.xlabel('Age Group')
# plt.ylabel('Total Cases')
# plt.savefig('age_group_cases.png')
# plt.close()

# # Plot 5: Caputo Fractional Derivative Cases
# plt.figure(figsize=(10, 6))
# plt.plot(india_data['date'], caputo_derivative_cases, label=f"Caputo Fractional Derivative (alpha={alpha})", color="red")
# plt.title('Caputo Fractional Derivative for COVID-19 Cases in India')
# plt.xlabel('Date')
# plt.ylabel('Fractional Derivative')
# plt.grid()
# plt.legend()
# plt.savefig('caputo_derivative_cases.png')
# plt.close()

# # Create PDF report
# pdf = FPDF()

# # Add a cover page
# pdf.add_page()
# pdf.set_font('Arial', 'B', 16)
# pdf.cell(200, 10, "COVID-19 in India: Data Analysis and Caputo Fractional Derivative Model", ln=True, align='C')

# pdf.set_font('Arial', '', 12)
# pdf.ln(10)
# pdf.multi_cell(0, 10, ("This report analyzes the COVID-19 cases in India, "
#                        "using a Caputo fractional derivative model to capture the memory "
#                        "effects and predict future case trends. The following sections "
#                        "provide insights into the data and the corresponding mathematical model."))

# # Add a section for COVID-19 Data Analysis
# pdf.add_page()
# pdf.set_font('Arial', 'B', 14)
# pdf.cell(200, 10, "1. COVID-19 Data Analysis", ln=True)

# pdf.set_font('Arial', '', 12)
# pdf.ln(10)
# pdf.multi_cell(0, 10, ("The following figures represent the analysis of COVID-19 data in India, "
#                        "including total cases, vaccinations, cases by gender, and age group."))

# def add_plot_to_pdf(pdf, image_path, title):
#     pdf.add_page()
#     pdf.set_font('Arial', 'B', 16)
#     pdf.cell(40, 10, title)
#     pdf.ln(10)
#     pdf.image(image_path, x=10, y=30, w=180)

# # Adding plots to PDF
# add_plot_to_pdf(pdf, 'total_cases.png', 'Total COVID-19 Cases in India')
# add_plot_to_pdf(pdf, 'total_vaccinations.png', 'Total COVID-19 Vaccinations in India')
# add_plot_to_pdf(pdf, 'gender_cases.png', 'COVID-19 Cases by Gender in India')
# add_plot_to_pdf(pdf, 'age_group_cases.png', 'COVID-19 Cases by Age Group in India')

# # Add Caputo model analysis
# pdf.add_page()
# pdf.set_font('Arial', 'B', 14)
# pdf.cell(200, 10, "2. Caputo Fractional Derivative Model", ln=True)

# pdf.set_font('Arial', '', 12)
# pdf.ln(10)
# pdf.multi_cell(0, 10, ("The Caputo fractional derivative is used to model the dynamics of COVID-19 cases. "
#                        "It considers the memory effects of previous infections to predict future case growth."
#                        "\n\nThe fractional order alpha was set to 0.9, representing a subdiffusion process, "
#                        "and the resulting curve shows the predicted rate of change in cases."))

# add_plot_to_pdf(pdf, 'caputo_derivative_cases.png', 'Caputo Fractional Derivative of COVID-19 Cases in India')

# # Save the PDF with the current date and time
# current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
# pdf_file_name = f'covid_report_{current_time}.pdf'
# pdf.output(pdf_file_name)

# print(f"PDF report saved as {pdf_file_name}")










import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fpdf import FPDF
from datetime import datetime
from scipy.special import gamma

# Load and preprocess data
data = pd.read_csv("./owid-covid-data.csv")

# Filter data for India
india_data = data[data['location'] == 'India'].copy()

# Convert the date column to datetime type
india_data['date'] = pd.to_datetime(india_data['date'])

# Preprocessing Data (Assume gender and age distribution are available or calculated)
india_data['gender'] = np.random.choice(['male', 'female'], size=len(india_data))  # Placeholder
india_data['age_group'] = np.random.choice(['0-18', '18-60', '60+'], size=len(india_data))  # Placeholder

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

# Applying the Caputo Derivative to model different variables
alpha = 0.9  # You can change this value

# For Total Cases
total_cases = india_data['total_cases'].fillna(0).values
time_days = np.linspace(0, len(total_cases), len(total_cases))
caputo_derivative_cases = caputo_derivative(total_cases, time_days, alpha)

# For Total Vaccinations
total_vaccinations = india_data['total_vaccinations'].fillna(0).values
caputo_derivative_vaccinations = caputo_derivative(total_vaccinations, time_days, alpha)

# For Gender-based New Cases
gender_grouped = india_data.groupby(['gender', india_data['date'].dt.date])['new_cases'].sum().unstack().fillna(0)
caputo_derivative_gender = {gender: caputo_derivative(gender_grouped.loc[gender].values, time_days[:len(gender_grouped.columns)], alpha)
                            for gender in gender_grouped.index}

# For age_group based New Cases
age_group_grouped = india_data.groupby(['age_group', india_data['date'].dt.date])['new_cases'].sum().unstack().fillna(0)
caputo_derivative_age_group = {age_group: caputo_derivative(age_group_grouped.loc[age_group].values, time_days[:len(age_group_grouped.columns)], alpha)
                               for age_group in age_group_grouped.index}

# Plot 1: Caputo Derivative for Total COVID-19 Cases
plt.figure(figsize=(10, 6))
plt.plot(india_data['date'], caputo_derivative_cases, label=f"Caputo Derivative (alpha={alpha})", color="blue")
plt.title('Caputo Derivative for Total COVID-19 Cases in India')
plt.xlabel('Date')
plt.ylabel('Fractional Derivative of Cases')
plt.grid()
plt.legend()
plt.savefig('caputo_total_cases.png')
plt.close()

# Plot 2: Caputo Derivative for Total Vaccinations
plt.figure(figsize=(10, 6))
plt.plot(india_data['date'], caputo_derivative_vaccinations, label=f"Caputo Derivative (alpha={alpha})", color="green")
plt.title('Caputo Derivative for Total Vaccinations in India')
plt.xlabel('Date')
plt.ylabel('Fractional Derivative of Vaccinations')
plt.grid()
plt.legend()
plt.savefig('caputo_total_vaccinations.png')
plt.close()

# Plot 3: Caputo Derivative for Gender-based Cases
plt.figure(figsize=(10, 6))
for gender, derivative in caputo_derivative_gender.items():
    plt.plot(india_data['date'][:len(derivative)], derivative, label=f"{gender.capitalize()} (Caputo Derivative)")
plt.title('Caputo Derivative for COVID-19 Cases by Gender in India')
plt.xlabel('Date')
plt.ylabel('Fractional Derivative of Cases by Gender')
plt.grid()
plt.legend()
plt.savefig('caputo_gender_cases.png')
plt.close()

# Plot 4: Caputo Derivative for age_group based Cases
plt.figure(figsize=(10, 6))
for age_group, derivative in caputo_derivative_age_group.items():
    plt.plot(india_data['date'][:len(derivative)], derivative, label=f"{age_group} (Caputo Derivative)")
plt.title('Caputo Derivative for COVID-19 Cases by Age Group in India')
plt.xlabel('Date')
plt.ylabel('Fractional Derivative of Cases by Age Group')
plt.grid()
plt.legend()
plt.savefig('caputo_age_group_cases.png')
plt.close()

# Create PDF report
pdf = FPDF()

# Add a cover page
pdf.add_page()
pdf.set_font('Arial', 'B', 16)
pdf.cell(200, 10, "COVID-19 in India: Caputo Derivative Analysis", ln=True, align='C')

pdf.set_font('Arial', '', 12)
pdf.ln(10)
pdf.multi_cell(0, 10, ("This report analyzes the COVID-19 cases in India using a Caputo fractional derivative model. "
                       "The analysis includes total cases, vaccinations, cases by gender, and age group, all modeled "
                       "using fractional derivatives to consider the memory effects in the system."))

# Add a section for Caputo Derivative Data Analysis
pdf.add_page()
pdf.set_font('Arial', 'B', 14)
pdf.cell(200, 10, "1. Caputo Derivative Data Analysis", ln=True)

pdf.set_font('Arial', '', 12)
pdf.ln(10)
pdf.multi_cell(0, 10, ("The following figures represent the Caputo derivative analysis of COVID-19 data in India, "
                       "including total cases, vaccinations, cases by gender, and age group."))

def add_plot_to_pdf(pdf, image_path, title):
    pdf.add_page()
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(40, 10, title)
    pdf.ln(10)
    pdf.image(image_path, x=10, y=30, w=180)

# Adding plots to PDF
add_plot_to_pdf(pdf, 'caputo_total_cases.png', 'Caputo Derivative for Total COVID-19 Cases')
add_plot_to_pdf(pdf, 'caputo_total_vaccinations.png', 'Caputo Derivative for Total Vaccinations')
add_plot_to_pdf(pdf, 'caputo_gender_cases.png', 'Caputo Derivative for COVID-19 Cases by Gender')
add_plot_to_pdf(pdf, 'caputo_age_group_cases.png', 'Caputo Derivative for COVID-19 Cases by Age Group')

# Save the PDF with the current date and time
current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
pdf_file_name = f'covid_caputo_report_{current_time}.pdf'
pdf.output(pdf_file_name)

print(f"PDF report saved as {pdf_file_name}")


# Load and preprocess data
data = pd.read_csv("./owid-covid-data.csv")

# Filter data for Brazil
brazil_data = data[data['location'] == 'Brazil'].copy()

# Convert the date column to datetime type
brazil_data['date'] = pd.to_datetime(brazil_data['date'])

# Preprocessing Data (Assume gender and age distribution are available or calculated)
brazil_data['gender'] = np.random.choice(['male', 'female'], size=len(brazil_data))  # Placeholder
brazil_data['age_group'] = np.random.choice(['0-18', '18-60', '60+'], size=len(brazil_data))  # Placeholder

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

# Applying the Caputo Derivative to model different variables
alpha = 0.9  # You can change this value

# For Total Cases
total_cases = brazil_data['total_cases'].fillna(0).values
time_days = np.linspace(0, len(total_cases), len(total_cases))
caputo_derivative_cases = caputo_derivative(total_cases, time_days, alpha)

# For Total Vaccinations
total_vaccinations = brazil_data['total_vaccinations'].fillna(0).values
caputo_derivative_vaccinations = caputo_derivative(total_vaccinations, time_days, alpha)

# For Gender-based New Cases
gender_grouped = brazil_data.groupby(['gender', brazil_data['date'].dt.date])['new_cases'].sum().unstack().fillna(0)
caputo_derivative_gender = {gender: caputo_derivative(gender_grouped.loc[gender].values, time_days[:len(gender_grouped.columns)], alpha)
                            for gender in gender_grouped.index}

# For age_group based New Cases
age_group_grouped = brazil_data.groupby(['age_group', brazil_data['date'].dt.date])['new_cases'].sum().unstack().fillna(0)
caputo_derivative_age_group = {age_group: caputo_derivative(age_group_grouped.loc[age_group].values, time_days[:len(age_group_grouped.columns)], alpha)
                               for age_group in age_group_grouped.index}

# Plot 1: Caputo Derivative for Total COVID-19 Cases
plt.figure(figsize=(10, 6))
plt.plot(brazil_data['date'], caputo_derivative_cases, label=f"Caputo Derivative (alpha={alpha})", color="blue")
plt.title('Caputo Derivative for Total COVID-19 Cases in Brazil')
plt.xlabel('Date')
plt.ylabel('Fractional Derivative of Cases')
plt.grid()
plt.legend()
plt.savefig('caputo_total_cases_brazil.png')
plt.close()

# Plot 2: Caputo Derivative for Total Vaccinations
plt.figure(figsize=(10, 6))
plt.plot(brazil_data['date'], caputo_derivative_vaccinations, label=f"Caputo Derivative (alpha={alpha})", color="green")
plt.title('Caputo Derivative for Total Vaccinations in Brazil')
plt.xlabel('Date')
plt.ylabel('Fractional Derivative of Vaccinations')
plt.grid()
plt.legend()
plt.savefig('caputo_total_vaccinations_brazil.png')
plt.close()

# Plot 3: Caputo Derivative for Gender-based Cases
plt.figure(figsize=(10, 6))
for gender, derivative in caputo_derivative_gender.items():
    plt.plot(brazil_data['date'][:len(derivative)], derivative, label=f"{gender.capitalize()} (Caputo Derivative)")
plt.title('Caputo Derivative for COVID-19 Cases by Gender in Brazil')
plt.xlabel('Date')
plt.ylabel('Fractional Derivative of Cases by Gender')
plt.grid()
plt.legend()
plt.savefig('caputo_gender_cases_brazil.png')
plt.close()

# Plot 4: Caputo Derivative for age_group based Cases
plt.figure(figsize=(10, 6))
for age_group, derivative in caputo_derivative_age_group.items():
    plt.plot(brazil_data['date'][:len(derivative)], derivative, label=f"{age_group} (Caputo Derivative)")
plt.title('Caputo Derivative for COVID-19 Cases by Age Group in Brazil')
plt.xlabel('Date')
plt.ylabel('Fractional Derivative of Cases by Age Group')
plt.grid()
plt.legend()
plt.savefig('caputo_age_group_cases_brazil.png')
plt.close()

# Create PDF report for Brazil
pdf = FPDF()

# Add a cover page
pdf.add_page()
pdf.set_font('Arial', 'B', 16)
pdf.cell(200, 10, "COVID-19 in Brazil: Caputo Derivative Analysis", ln=True, align='C')

pdf.set_font('Arial', '', 12)
pdf.ln(10)
pdf.multi_cell(0, 10, ("This report analyzes the COVID-19 cases in Brazil using a Caputo fractional derivative model. "
                       "The analysis includes total cases, vaccinations, cases by gender, and age group, all modeled "
                       "using fractional derivatives to consider the memory effects in the system."))

# Add a section for Caputo Derivative Data Analysis
pdf.add_page()
pdf.set_font('Arial', 'B', 14)
pdf.cell(200, 10, "1. Caputo Derivative Data Analysis", ln=True)

pdf.set_font('Arial', '', 12)
pdf.ln(10)
pdf.multi_cell(0, 10, ("The following figures represent the Caputo derivative analysis of COVID-19 data in Brazil, "
                       "including total cases, vaccinations, cases by gender, and age group."))

# Add the plots to the PDF report
def add_plot_to_pdf(pdf, image_path, title):
    pdf.add_page()
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(40, 10, title)
    pdf.ln(10)
    pdf.image(image_path, x=10, y=30, w=180)

# Adding plots to PDF
add_plot_to_pdf(pdf, 'caputo_total_cases_brazil.png', 'Caputo Derivative for Total COVID-19 Cases')
add_plot_to_pdf(pdf, 'caputo_total_vaccinations_brazil.png', 'Caputo Derivative for Total Vaccinations')
add_plot_to_pdf(pdf, 'caputo_gender_cases_brazil.png', 'Caputo Derivative for COVID-19 Cases by Gender')
add_plot_to_pdf(pdf, 'caputo_age_group_cases_brazil.png', 'Caputo Derivative for COVID-19 Cases by Age Group')

# Save the PDF with the current date and time
current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
pdf_file_name = f'covid_caputo_report_brazil_{current_time}.pdf'
pdf.output(pdf_file_name)

print(f"PDF report saved as {pdf_file_name}")



# Load and preprocess data for the United States
data = pd.read_csv("./owid-covid-data.csv")

# Filter data for United States
us_data = data[data['location'] == 'United States'].copy()

# Convert the date column to datetime type
us_data['date'] = pd.to_datetime(us_data['date'])

# Preprocessing Data (Assume gender and age distribution are available or calculated)
us_data['gender'] = np.random.choice(['male', 'female'], size=len(us_data))  # Placeholder
us_data['age_group'] = np.random.choice(['0-18', '18-60', '60+'], size=len(us_data))  # Placeholder

# Define Caputo Fractional Derivative function (as already defined)
def caputo_derivative(y, t, alpha):
    h = t[1] - t[0]  # Assuming uniform time step
    result = np.zeros(len(y))
    
    for i in range(1, len(t)):
        sum_term = 0
        for k in range(0, i):
            sum_term += (y[k] - y[i]) / (t[i] - t[k])**(1 - alpha)
        result[i] = (h**alpha) / gamma(2 - alpha) * sum_term
    return result

# Applying the Caputo Derivative to model different variables
alpha = 0.9  # You can change this value

# For Total Cases
total_cases = us_data['total_cases'].fillna(0).values
time_days = np.linspace(0, len(total_cases), len(total_cases))
caputo_derivative_cases = caputo_derivative(total_cases, time_days, alpha)

# For Total Vaccinations
total_vaccinations = us_data['total_vaccinations'].fillna(0).values
caputo_derivative_vaccinations = caputo_derivative(total_vaccinations, time_days, alpha)

# For Gender-based New Cases
gender_grouped = us_data.groupby(['gender', us_data['date'].dt.date])['new_cases'].sum().unstack().fillna(0)
caputo_derivative_gender = {gender: caputo_derivative(gender_grouped.loc[gender].values, time_days[:len(gender_grouped.columns)], alpha)
                            for gender in gender_grouped.index}

# For age_group based New Cases
age_group_grouped = us_data.groupby(['age_group', us_data['date'].dt.date])['new_cases'].sum().unstack().fillna(0)
caputo_derivative_age_group = {age_group: caputo_derivative(age_group_grouped.loc[age_group].values, time_days[:len(age_group_grouped.columns)], alpha)
                               for age_group in age_group_grouped.index}

# Plot 1: Caputo Derivative for Total COVID-19 Cases
plt.figure(figsize=(10, 6))
plt.plot(us_data['date'], caputo_derivative_cases, label=f"Caputo Derivative (alpha={alpha})", color="blue")
plt.title('Caputo Derivative for Total COVID-19 Cases in United States')
plt.xlabel('Date')
plt.ylabel('Fractional Derivative of Cases')
plt.grid()
plt.legend()
plt.savefig('caputo_us_total_cases.png')
plt.close()

# Plot 2: Caputo Derivative for Total Vaccinations
plt.figure(figsize=(10, 6))
plt.plot(us_data['date'], caputo_derivative_vaccinations, label=f"Caputo Derivative (alpha={alpha})", color="green")
plt.title('Caputo Derivative for Total Vaccinations in United States')
plt.xlabel('Date')
plt.ylabel('Fractional Derivative of Vaccinations')
plt.grid()
plt.legend()
plt.savefig('caputo_us_total_vaccinations.png')
plt.close()

# Plot 3: Caputo Derivative for Gender-based Cases
plt.figure(figsize=(10, 6))
for gender, derivative in caputo_derivative_gender.items():
    plt.plot(us_data['date'][:len(derivative)], derivative, label=f"{gender.capitalize()} (Caputo Derivative)")
plt.title('Caputo Derivative for COVID-19 Cases by Gender in United States')
plt.xlabel('Date')
plt.ylabel('Fractional Derivative of Cases by Gender')
plt.grid()
plt.legend()
plt.savefig('caputo_us_gender_cases.png')
plt.close()

# Plot 4: Caputo Derivative for Age Group-based Cases
plt.figure(figsize=(10, 6))
for age_group, derivative in caputo_derivative_age_group.items():
    plt.plot(us_data['date'][:len(derivative)], derivative, label=f"{age_group} (Caputo Derivative)")
plt.title('Caputo Derivative for COVID-19 Cases by Age Group in United States')
plt.xlabel('Date')
plt.ylabel('Fractional Derivative of Cases by Age Group')
plt.grid()
plt.legend()
plt.savefig('caputo_us_age_group_cases.png')
plt.close()

# Create PDF report for United States
pdf = FPDF()

# Add a cover page
pdf.add_page()
pdf.set_font('Arial', 'B', 16)
pdf.cell(200, 10, "COVID-19 in United States: Caputo Derivative Analysis", ln=True, align='C')

pdf.set_font('Arial', '', 12)
pdf.ln(10)
pdf.multi_cell(0, 10, ("This report analyzes the COVID-19 cases in the United States using a Caputo fractional derivative model. "
                       "The analysis includes total cases, vaccinations, cases by gender, and age group, all modeled "
                       "using fractional derivatives to consider the memory effects in the system."))

# Add a section for Caputo Derivative Data Analysis
pdf.add_page()
pdf.set_font('Arial', 'B', 14)
pdf.cell(200, 10, "1. Caputo Derivative Data Analysis", ln=True)

pdf.set_font('Arial', '', 12)
pdf.ln(10)
pdf.multi_cell(0, 10, ("The following figures represent the Caputo derivative analysis of COVID-19 data in the United States, "
                       "including total cases, vaccinations, cases by gender, and age group."))

# Function to add plots to PDF
def add_plot_to_pdf(pdf, image_path, title):
    pdf.add_page()
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(40, 10, title)
    pdf.ln(10)
    pdf.image(image_path, x=10, y=30, w=180)

# Adding plots to the PDF for United States
add_plot_to_pdf(pdf, 'caputo_us_total_cases.png', 'Caputo Derivative for Total COVID-19 Cases')
add_plot_to_pdf(pdf, 'caputo_us_total_vaccinations.png', 'Caputo Derivative for Total Vaccinations')
add_plot_to_pdf(pdf, 'caputo_us_gender_cases.png', 'Caputo Derivative for COVID-19 Cases by Gender')
add_plot_to_pdf(pdf, 'caputo_us_age_group_cases.png', 'Caputo Derivative for COVID-19 Cases by Age Group')

# Save the PDF with the current date and time
current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
pdf_file_name = f'covid_caputo_us_report_{current_time}.pdf'
pdf.output(pdf_file_name)

print(f"PDF report saved as {pdf_file_name}")



# Load and preprocess data for Thailand
data = pd.read_csv("./owid-covid-data.csv")

# Filter data for Thailand
thailand_data = data[data['location'] == 'Thailand'].copy()

# Convert the date column to datetime type
thailand_data['date'] = pd.to_datetime(thailand_data['date'])

# Preprocessing Data (Assume gender and age distribution are available or calculated)
thailand_data['gender'] = np.random.choice(['male', 'female'], size=len(thailand_data))  # Placeholder
thailand_data['age_group'] = np.random.choice(['0-18', '18-60', '60+'], size=len(thailand_data))  # Placeholder

# Define Caputo Fractional Derivative function (as already defined)
def caputo_derivative(y, t, alpha):
    h = t[1] - t[0]  # Assuming uniform time step
    result = np.zeros(len(y))
    
    for i in range(1, len(t)):
        sum_term = 0
        for k in range(0, i):
            sum_term += (y[k] - y[i]) / (t[i] - t[k])**(1 - alpha)
        result[i] = (h**alpha) / gamma(2 - alpha) * sum_term
    return result

# Applying the Caputo Derivative to model different variables
alpha = 0.9  # You can change this value

# For Total Cases
total_cases = thailand_data['total_cases'].fillna(0).values
time_days = np.linspace(0, len(total_cases), len(total_cases))
caputo_derivative_cases = caputo_derivative(total_cases, time_days, alpha)

# For Total Vaccinations
total_vaccinations = thailand_data['total_vaccinations'].fillna(0).values
caputo_derivative_vaccinations = caputo_derivative(total_vaccinations, time_days, alpha)

# For Gender-based New Cases
gender_grouped = thailand_data.groupby(['gender', thailand_data['date'].dt.date])['new_cases'].sum().unstack().fillna(0)
caputo_derivative_gender = {gender: caputo_derivative(gender_grouped.loc[gender].values, time_days[:len(gender_grouped.columns)], alpha)
                            for gender in gender_grouped.index}

# For age_group based New Cases
age_group_grouped = thailand_data.groupby(['age_group', thailand_data['date'].dt.date])['new_cases'].sum().unstack().fillna(0)
caputo_derivative_age_group = {age_group: caputo_derivative(age_group_grouped.loc[age_group].values, time_days[:len(age_group_grouped.columns)], alpha)
                               for age_group in age_group_grouped.index}

# Plot 1: Caputo Derivative for Total COVID-19 Cases
plt.figure(figsize=(10, 6))
plt.plot(thailand_data['date'], caputo_derivative_cases, label=f"Caputo Derivative (alpha={alpha})", color="blue")
plt.title('Caputo Derivative for Total COVID-19 Cases in Thailand')
plt.xlabel('Date')
plt.ylabel('Fractional Derivative of Cases')
plt.grid()
plt.legend()
plt.savefig('caputo_thailand_total_cases.png')
plt.close()

# Plot 2: Caputo Derivative for Total Vaccinations
plt.figure(figsize=(10, 6))
plt.plot(thailand_data['date'], caputo_derivative_vaccinations, label=f"Caputo Derivative (alpha={alpha})", color="green")
plt.title('Caputo Derivative for Total Vaccinations in Thailand')
plt.xlabel('Date')
plt.ylabel('Fractional Derivative of Vaccinations')
plt.grid()
plt.legend()
plt.savefig('caputo_thailand_total_vaccinations.png')
plt.close()

# Plot 3: Caputo Derivative for Gender-based Cases
plt.figure(figsize=(10, 6))
for gender, derivative in caputo_derivative_gender.items():
    plt.plot(thailand_data['date'][:len(derivative)], derivative, label=f"{gender.capitalize()} (Caputo Derivative)")
plt.title('Caputo Derivative for COVID-19 Cases by Gender in Thailand')
plt.xlabel('Date')
plt.ylabel('Fractional Derivative of Cases by Gender')
plt.grid()
plt.legend()
plt.savefig('caputo_thailand_gender_cases.png')
plt.close()

# Plot 4: Caputo Derivative for Age Group-based Cases
plt.figure(figsize=(10, 6))
for age_group, derivative in caputo_derivative_age_group.items():
    plt.plot(thailand_data['date'][:len(derivative)], derivative, label=f"{age_group} (Caputo Derivative)")
plt.title('Caputo Derivative for COVID-19 Cases by Age Group in Thailand')
plt.xlabel('Date')
plt.ylabel('Fractional Derivative of Cases by Age Group')
plt.grid()
plt.legend()
plt.savefig('caputo_thailand_age_group_cases.png')
plt.close()

# Create PDF report for Thailand
pdf = FPDF()

# Add a cover page
pdf.add_page()
pdf.set_font('Arial', 'B', 16)
pdf.cell(200, 10, "COVID-19 in Thailand: Caputo Derivative Analysis", ln=True, align='C')

pdf.set_font('Arial', '', 12)
pdf.ln(10)
pdf.multi_cell(0, 10, ("This report analyzes the COVID-19 cases in Thailand using a Caputo fractional derivative model. "
                       "The analysis includes total cases, vaccinations, cases by gender, and age group, all modeled "
                       "using fractional derivatives to consider the memory effects in the system."))

# Add a section for Caputo Derivative Data Analysis
pdf.add_page()
pdf.set_font('Arial', 'B', 14)
pdf.cell(200, 10, "1. Caputo Derivative Data Analysis", ln=True)

pdf.set_font('Arial', '', 12)
pdf.ln(10)
pdf.multi_cell(0, 10, ("The following figures represent the Caputo derivative analysis of COVID-19 data in Thailand, "
                       "including total cases, vaccinations, cases by gender, and age group."))

# Function to add plots to PDF
def add_plot_to_pdf(pdf, image_path, title):
    pdf.add_page()
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(40, 10, title)
    pdf.ln(10)
    pdf.image(image_path, x=10, y=30, w=180)

# Adding plots to the PDF for Thailand
add_plot_to_pdf(pdf, 'caputo_thailand_total_cases.png', 'Caputo Derivative for Total COVID-19 Cases')
add_plot_to_pdf(pdf, 'caputo_thailand_total_vaccinations.png', 'Caputo Derivative for Total Vaccinations')
add_plot_to_pdf(pdf, 'caputo_thailand_gender_cases.png', 'Caputo Derivative for COVID-19 Cases by Gender')
add_plot_to_pdf(pdf, 'caputo_thailand_age_group_cases.png', 'Caputo Derivative for COVID-19 Cases by Age Group')

# Save the PDF with the current date and time
current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
pdf_file_name = f'covid_caputo_thailand_report_{current_time}.pdf'
pdf.output(pdf_file_name)

print(f"PDF report saved as {pdf_file_name}")



# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from fpdf import FPDF
# from datetime import datetime
# from scipy.special import gamma

# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from fpdf import FPDF
# from datetime import datetime
# from scipy.special import gamma
# from scipy.integrate import odeint  # <-- Import this to use odeint for SIR model
# from sklearn.metrics import mean_squared_error  # Import to calculate MSE

# # Load and preprocess data
# data = pd.read_csv("./owid-covid-data.csv")

# # List of countries to analyze
# countries = ['India', 'Brazil', 'United States', 'Thailand']

# # Function to filter and process data for a given country
# def get_country_data(country):
#     country_data = data[data['location'] == country].copy()
#     country_data['date'] = pd.to_datetime(country_data['date'])
#     return country_data

# # Define Caputo Fractional Derivative function
# def caputo_derivative(y, t, alpha):
#     h = t[1] - t[0]  # Assuming uniform time step
#     result = np.zeros(len(y))
    
#     for i in range(1, len(t)):
#         sum_term = 0
#         for k in range(0, i):
#             sum_term += (y[k] - y[i]) / (t[i] - t[k])**(1 - alpha)
#         result[i] = (h**alpha) / gamma(2 - alpha) * sum_term
#     return result

# # Define alpha value for Caputo Derivative
# alpha = 0.9

# # Dictionary to store caputo derivatives for all countries
# caputo_derivatives = {}

# # Process each country
# for country in countries:
#     country_data = get_country_data(country)
    
#     # For Total Cases
#     total_cases = country_data['total_cases'].fillna(0).values
#     time_days = np.linspace(0, len(total_cases), len(total_cases))
#     caputo_derivatives[country] = caputo_derivative(total_cases, time_days, alpha)

#     # Plot for Total Cases (Caputo Derivative)
#     plt.figure(figsize=(10, 6))
#     plt.plot(country_data['date'], caputo_derivatives[country], label=f"Caputo Derivative (alpha={alpha})", color="blue")
#     plt.title(f'Caputo Derivative for Total COVID-19 Cases in {country}')
#     plt.xlabel('Date')
#     plt.ylabel('Fractional Derivative of Cases')
#     plt.grid()
#     plt.legend()
#     plt.savefig(f'caputo_total_cases_{country}.png')
#     plt.close()

#     # Gender-wise COVID-19 Cases Plot (Assuming columns for gender-specific cases exist)
#     if 'male_cases' in country_data.columns and 'female_cases' in country_data.columns:
#         plt.figure(figsize=(10, 6))
#         plt.plot(country_data['date'], country_data['male_cases'].fillna(0), label="Male Cases", color="blue")
#         plt.plot(country_data['date'], country_data['female_cases'].fillna(0), label="Female Cases", color="red")
#         plt.title(f'COVID-19 Cases by Gender in {country}')
#         plt.xlabel('Date')
#         plt.ylabel('Number of Cases')
#         plt.grid()
#         plt.legend()
#         plt.savefig(f'gender_cases_{country}.png')
#         plt.close()

#     # Age-wise COVID-19 Cases Plot (Assuming age-specific columns exist)
#     age_columns = [col for col in country_data.columns if 'age' in col]
#     if age_columns:
#         plt.figure(figsize=(10, 6))
#         for col in age_columns:
#             plt.plot(country_data['date'], country_data[col].fillna(0), label=col)
#         plt.title(f'COVID-19 Cases by Age Group in {country}')
#         plt.xlabel('Date')
#         plt.ylabel('Number of Cases')
#         plt.grid()
#         plt.legend()
#         plt.savefig(f'age_cases_{country}.png')
#         plt.close()

# # Comparison Plot: Total COVID-19 Cases for All Countries
# plt.figure(figsize=(12, 8))
# for country in countries:
#     country_data = get_country_data(country)
#     plt.plot(country_data['date'], caputo_derivatives[country], label=f'{country} (Caputo Derivative)')
# plt.title('Caputo Derivative for Total COVID-19 Cases in India, Brazil, USA, and Thailand')
# plt.xlabel('Date')
# plt.ylabel('Fractional Derivative of Cases')
# plt.grid()
# plt.legend()
# plt.savefig('caputo_total_cases_comparison.png')
# plt.close()

# # Create PDF report
# pdf = FPDF()

# # Add a cover page
# pdf.add_page()
# pdf.set_font('Arial', 'B', 16)
# pdf.cell(200, 10, "COVID-19: Caputo Derivative Analysis for Multiple Countries", ln=True, align='C')

# pdf.set_font('Arial', '', 12)
# pdf.ln(10)
# pdf.multi_cell(0, 10, ("This report analyzes the COVID-19 cases in India, Brazil, USA, and Thailand using a Caputo "
#                        "fractional derivative model. The analysis includes total cases modeled using fractional derivatives "
#                        "to account for memory effects in the system."))

# # Add a section for Caputo Derivative Data Analysis
# pdf.add_page()
# pdf.set_font('Arial', 'B', 14)
# pdf.cell(200, 10, "1. Caputo Derivative Data Analysis for Total Cases", ln=True)

# pdf.set_font('Arial', '', 12)
# pdf.ln(10)
# pdf.multi_cell(0, 10, "The following figures represent the Caputo derivative analysis of COVID-19 total cases for India, Brazil, USA, and Thailand.")

# # Function to add plot to PDF
# def add_plot_to_pdf(pdf, image_path, title):
#     pdf.add_page()
#     pdf.set_font('Arial', 'B', 16)
#     pdf.cell(40, 10, title)
#     pdf.ln(10)
#     pdf.image(image_path, x=10, y=30, w=180)

# # Add country-specific plots to PDF
# for country in countries:
#     add_plot_to_pdf(pdf, f'caputo_total_cases_{country}.png', f'Caputo Derivative for Total COVID-19 Cases in {country}')
    
#     # Gender-wise plot (if generated)
#     if 'male_cases' in get_country_data(country).columns and 'female_cases' in get_country_data(country).columns:
#         add_plot_to_pdf(pdf, f'gender_cases_{country}.png', f'Gender-wise COVID-19 Cases in {country}')
    
#     # Age-wise plot (if generated)
#     age_columns = [col for col in get_country_data(country).columns if 'age' in col]
#     if age_columns:
#         add_plot_to_pdf(pdf, f'age_cases_{country}.png', f'Age-wise COVID-19 Cases in {country}')

# # Add comparison plot for all countries
# add_plot_to_pdf(pdf, 'caputo_total_cases_comparison.png', 'Comparison of Caputo Derivative for Total COVID-19 Cases')

# # Save the PDF with the current date and time
# current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
# pdf_file_name = f'covid_caputo_report_comparison_{current_time}.pdf'
# pdf.output(pdf_file_name)

# print(f"PDF report saved as {pdf_file_name}")


# # SIR model differential equations
# def sir_model(y, t, beta, gamma):
#     S, I, R = y
#     dS_dt = -beta * S * I
#     dI_dt = beta * S * I - gamma * I
#     dR_dt = gamma * I
#     return [dS_dt, dI_dt, dR_dt]

# # Solve the SIR model
# def solve_sir(S0, I0, R0, beta, gamma, t):
#     y0 = [S0, I0, R0]
#     return odeint(sir_model, y0, t, args=(beta, gamma))

# # Compare SIR model to Caputo Derivative and calculate MSE
# def compare_models(country):
#     # Load country data
#     country_data = get_country_data(country)
    
#     # Extract total cases (actual data)
#     total_cases = country_data['total_cases'].fillna(0).values
#     time_days = np.linspace(0, len(total_cases), len(total_cases))
    
#     # Normalize data for SIR model
#     N = country_data['population'].iloc[0]  # Population size
#     I0 = total_cases[0]  # Initial infected
#     R0 = 0  # Initial recovered
#     S0 = N - I0  # Initial susceptible
    
#     # Parameters for SIR model (to be adjusted for fitting)
#     beta = 0.3  # Infection rate (can be optimized)
#     gamma = 0.1  # Recovery rate (can be optimized)
    
#     # Solve the SIR model
#     t = np.linspace(0, len(total_cases), len(total_cases))
#     SIR_solution = solve_sir(S0, I0, R0, beta, gamma, t)
    
#     # Extract infected population from SIR solution
#     SIR_infected = SIR_solution[:, 1]
    
#     # Caputo derivative solution (already calculated)
#     caputo_solution = caputo_derivatives[country]
    
#     # Calculate MSE for SIR model
#     mse_sir = mean_squared_error(total_cases, SIR_infected)
    
#     # Calculate MSE for Caputo Derivative model
#     mse_caputo = mean_squared_error(total_cases, caputo_solution)
    
#     # Print the MSE for both models
#     print(f"MSE for SIR Model (Country: {country}): {mse_sir}")
#     print(f"MSE for Caputo Derivative Model (Country: {country}): {mse_caputo}")
    
#     # Plot the comparison
#     plt.figure(figsize=(10, 6))
#     plt.plot(country_data['date'], total_cases, label='Actual Data', color='black', linestyle='--')
#     plt.plot(country_data['date'], SIR_infected, label='SIR Model Infected', color='green')
#     plt.plot(country_data['date'], caputo_solution, label=f'Caputo Derivative (alpha={alpha})', color='blue')
#     plt.title(f'COVID-19 Cases Comparison for {country}')
#     plt.xlabel('Date')
#     plt.ylabel('Number of Cases')
#     plt.legend()
#     plt.grid()
#     plt.show()

# # Example: Compare for India
# compare_models('India')
# compare_models('United States')
# compare_models('Brazil')
# compare_models('Thailand')




