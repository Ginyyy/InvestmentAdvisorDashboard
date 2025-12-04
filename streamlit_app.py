import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os

# Read data from CSV in the `data/` folder
data_path = os.path.join("data", "cleaned_properties dataset.csv")
df = pd.read_csv(data_path)
st.set_page_config(layout='wide')
st.markdown('<style>div.block-container{padding-top:1rem;}</style>', unsafe_allow_html=True)

st.title("Smart Property Investment Advisor")
st.write(
    "Let's start building! For help and inspiration, head over to [docs.streamlit.io](https://docs.streamlit.io/)."
)


# --- Data Loading Function (STUBBED FOR UI DEVELOPMENT) ---

def load_data():
    """
    STUB FUNCTION: Returns a hardcoded mock DataFrame to render the Streamlit UI.
    This bypasses the need to load the actual 'cleaned_properties dataset.csv'.
    """
    print("--- INFO: Using hardcoded mock data to render the UI. ---")
    
    # Essential columns required by the charts and analysis
    data = pd.DataFrame({
        'Property_ID': [101, 102, 103, 104, 105, 106],
        'Price_per_sqm': [25000, 35000, 18000, 42000, 22000, 31000],
        'Rental_Yield_percent': [4.8, 3.1, 5.5, 2.5, 4.2, 3.8],
        'ROI_percent': [15.2, 10.5, 18.0, 7.1, 13.9, 11.8],
        'Tier': [
            'Tier 1: Great Buy', 
            'Tier 2: Moderate Buy', 
            'Tier 1: Great Buy', 
            'Tier 4: Poor Investment',
            'Tier 2: Moderate Buy', 
            'Tier 3: Risky/Low Buy'
        ]
    })
    return data.copy()


# --- Simulation Function (Remains the same as before) ---

def calculate_net_worth(
    initial_cash, price, down_payment_perc, loan_rate, loan_term_years, 
    appreciation_rate, market_return_rate, horizon_years
):
    """
    Simulates the net worth trajectories for 'Buy' vs 'Rent and Invest' scenarios.
    
    Returns:
    - buy_net_worth (numpy array)
    - rent_net_worth (numpy array)
    - years_data (list of years)
    """
    years_data = np.arange(horizon_years + 1)
    
    # --- Buy Scenario Calculations ---
    down_payment_amount = price * down_payment_perc
    loan_principal = price - down_payment_amount
    annual_rate = loan_rate
    monthly_rate = annual_rate / 12
    n_payments = loan_term_years * 12
    
    if monthly_rate > 0 and n_payments > 0:
        emi = loan_principal * (monthly_rate * (1 + monthly_rate)**n_payments) / ((1 + monthly_rate)**n_payments - 1)
    else:
        emi = loan_principal / (loan_term_years * 12) if loan_term_years > 0 else 0
        
    annual_mortgage = emi * 12
    remaining_cash_after_down = initial_cash - down_payment_amount
    
    buy_net_worth = np.zeros(horizon_years + 1)
    buy_net_worth[0] = initial_cash
    current_property_value = price
    current_loan_balance = loan_principal
    
    for y in years_data[1:]:
        current_property_value *= (1 + appreciation_rate)
        
        if current_loan_balance > 0 and y <= loan_term_years:
            annual_interest_paid = current_loan_balance * annual_rate
            annual_principal_paid = annual_mortgage - annual_interest_paid
            current_loan_balance = max(0, current_loan_balance - annual_principal_paid)
        
        elif y > loan_term_years:
            current_loan_balance = 0 

        equity = current_property_value - current_loan_balance
        buy_net_worth[y] = equity + remaining_cash_after_down 
    
    # --- Rent & Invest Scenario Calculations ---
    rent_investments = np.zeros(horizon_years + 1)
    rent_investments[0] = initial_cash
    
    MOCK_RENTAL_YIELD = 0.035
    annual_rent = price * MOCK_RENTAL_YIELD
    annual_market_contribution = annual_mortgage 
    
    for y in years_data[1:]:
        rent_investments[y] = rent_investments[y-1] * (1 + market_return_rate)
        rent_investments[y] += annual_market_contribution
        
    rent_net_worth = rent_investments

    return buy_net_worth, rent_net_worth, years_data.tolist()


if __name__ == '__main__':
    # --- Minimal UI to preview dataset and simulation ---

    use_mock = st.checkbox('Use mock data (for UI development)', value=True)
    if use_mock:
        df_display = load_data()
    else:
        try:
            df_display = pd.read_csv(data_path)
        except Exception as e:
            st.error(f'Could not load dataset: {e}')
            st.stop()

    st.subheader('Dataset preview')
    st.write(f'Rows: {df_display.shape[0]}, Columns: {df_display.shape[1]}')
    st.dataframe(df_display.head())

    st.subheader('Tier distribution')
    fig = px.histogram(df_display, x='Tier', title='Tier counts')
    st.plotly_chart(fig, use_container_width=True)

    st.subheader('Buy vs Rent simulation (example)')
    col1, col2 = st.columns(2)
    with col1:
        initial_cash = st.number_input('Initial cash', value=100000.0)
        price = st.number_input('Property price', value=500000.0)
        down_payment = st.slider('Down payment %', min_value=0.05, max_value=0.5, value=0.2, step=0.01)
    with col2:
        loan_rate = st.number_input('Loan annual rate', value=0.04)
        loan_term = st.number_input('Loan term (years)', value=20)
        horizon = st.number_input('Investment horizon (years)', value=30)

    if st.button('Run simulation'):
        buy, rent, years = calculate_net_worth(
            initial_cash, price, down_payment, loan_rate, int(loan_term), 0.03, 0.06, int(horizon)
        )
        sim_df = pd.DataFrame({'Year': years, 'Buy': buy, 'Rent': rent})
        fig2 = px.line(sim_df, x='Year', y=['Buy', 'Rent'], title='Net worth: Buy vs Rent')
        st.plotly_chart(fig2, use_container_width=True)


