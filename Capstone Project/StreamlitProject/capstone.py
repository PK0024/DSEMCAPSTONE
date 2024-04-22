import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
import streamlit as st
import matplotlib.pyplot as plt

# Load and preprocess the data
data = pd.read_csv('Mental health Depression disorder Data.csv')
data = data[['Entity', 'Year', 'Depression (%)']].dropna()

# Encode countries as categorical numeric values
data['Entity'] = data['Entity'].astype('category').cat.codes

# Prepare the data for training
X = data[['Entity', 'Year']]
y = data['Depression (%)']

# Transform features with Polynomial Features
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

# Model selection - try multiple models
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(),
    'Random Forest': RandomForestRegressor(n_estimators=100)
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    results[name] = mse
    print(f'{name} Mean Squared Error: {mse}')

# Streamlit app interface
st.title('Depression Prevalence Prediction Tool')

# Loading data again for the dropdown to avoid using encoded values
countries = pd.read_csv('Mental health Depression disorder Data.csv')['Entity'].dropna().unique()
country_dict = {country: i for i, country in enumerate(countries)}

# User inputs
selected_countries = st.multiselect('Select Countries', countries, default=countries[0])
selected_years = st.slider('Select Year Range', 1990, 2025, (2010, 2020))
model_choice = st.selectbox('Select Model', list(models.keys()))

# Get predictions
if st.button('Get Predictions'):
    # Prepare data for prediction
    predictions_data = []
    for year in range(selected_years[0], selected_years[1] + 1):
        for country in selected_countries:
            country_code = country_dict[country]
            feature = poly.transform(np.array([[country_code, year]]))
            prediction = models[model_choice].predict(feature)
            predictions_data.append({'Country': country, 'Year': year, 'Prediction': prediction[0]})

    # Convert to DataFrame
    df_predictions = pd.DataFrame(predictions_data)

    # Visualization
    fig, ax = plt.subplots(figsize=(10, 5))
    for country in selected_countries:
        subset = df_predictions[df_predictions['Country'] == country]
        ax.plot(subset['Year'], subset['Prediction'], marker='o', label=country)
    ax.set_title('Depression Prevalence Predictions')
    ax.set_xlabel('Year')
    ax.set_ylabel('Depression Prevalence (%)')
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

# Explain the output
st.markdown("""
This tool allows you to compare the predicted depression rates across different countries over a selected range of years. 
You can choose between different regression models to see how predictions vary.
""")
