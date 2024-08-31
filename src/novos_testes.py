import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#from optimize import fit_data
import optimize
from new_wave import get_transition_points

# Use full page width
st.set_page_config(layout="wide")

# Import data
data = pd.read_csv("../Datasets/italy_regions.csv") 

# Page title and sub header
st.title('Multi-wave modelling and short-term prediction of ICU bed occupancy by patients with Covid-19 in regions of Italy')
st.subheader('Math. Model. Nat. Phenom. 19 (2024) 13 - https://doi.org/10.1051/mmnp/2024012')


col1, col2 = st.columns(2)

with col1:
    # Allow region selection
    cities = ['Lombardia', 'Lazio', 'Campania', 'Veneto', 'Sicilia']
    city_name = st.selectbox('Region Name', cities)
with col2:
    # Allow indicator selection
    indicator = st.selectbox('Pandemic Indicator', ['ICU Admissions'], disabled = True)

with st.spinner('Please wait...'):

    # Filter selected region
    data = data[data['denominazione_regione'] == city_name]

    # Get daily indicator
    daily_data = data['terapia_intensiva']

    # Get cummulative indicator from daily data
    acc_data = []
    for i in range(len(daily_data)):
        acc_data.append(np.sum(daily_data[:i]))

    # time array
    t = np.linspace(0, len(acc_data)-1, len(acc_data))

    normalized_acc_data = acc_data / max(acc_data)

    #scaling_factor = 1000
    scaling_factor = max(acc_data)
    #scaling_factor = 1

    acc_data = np.array(acc_data) / scaling_factor
    daily_data = list(daily_data/ scaling_factor)


    # Initial Conditions

    def initial_cond_0(y_t):
        A0 = 2*max(y_t)
        tp0 = (2/3)*len(y_t)
        delta0 = (1/4)*len(y_t)
        nu0 = 1
        return [A0, tp0, delta0, nu0]

    optimize.initial_cond = initial_cond_0

    def update_cond_nw(A0, tp0):
        return [A0, tp0]
        
    optimize.update_cond = update_cond_nw

    if (city_name == 'Campania'):
        tp_threshold = 3e-6
    elif (city_name == 'Veneto'):
        # new automatic Veneto 2e-6
        tp_threshold = 2e-6
    elif (city_name == 'Sicilia'):
        tp_threshold = 5e-6
    else:
        tp_threshold = 1e-6

    col1, col2 = st.columns([1, 2])

    with col1:    
        # Transition Points
        x_nw, fig = get_transition_points(scaling_factor*acc_data, visual=True, threshold=tp_threshold, indicator = indicator, city_name=city_name)

        st.pyplot(fig)    

    with col2:
        if (city_name == 'Campania' or city_name == 'Sicilia'):
            x_nw = x_nw[1:7] # Campania e Sicilia
        else:
            x_nw = x_nw[:6]

        print('x_nw:', x_nw)

        if (x_nw[-1] != len(acc_data) - 1):
            x_nw.append(len(acc_data) - 1)  

        # utilizando scaling_factor = max(acc_data)

        if (city_name == 'Lombardia'):
            # Manual (old)
            #x_nw = [189, 361, 532, 630, 865, 1144]

            # New automatic Lombardia
            x_nw = [174, 356, 523, 624, 856, 951]
        elif (city_name == 'Lazio'):
            # New automatic Lazio
            x_nw = [187, 373, 514, 609, 839, 950]
        elif (city_name == 'Campania'):
            # New automatic Campania 3e-6 x_nw[1:7]
            x_nw =  [184, 337, 525, 625, 646, 780]
        elif (city_name == 'Veneto'):
            # new automatic Veneto 2e-6
            x_nw =  [183, 371, 506, 612, 800, 845]
        elif (city_name == 'Sicilia'):
            #x_nw = [156, 320, 396, 513, 635, 852, 973] # com ITSE_norm e scaling_factor = 1000

            # Siscilia 5e-6 manual (213 -> 150) com th 5e-6 x_nw[1:7]
            x_nw = [150, 313, 389, 509, 651, 852]


        sig_params, rel_rmse_list, rel_rmse_list_pred, y_m, fig_opt = optimize.fit_data(acc_data, 
                                    daily_data, 
                                    city_name, 
                                    x_nw[:6], 
                                    indicator = indicator, 
                                    n_weeks_pred = 2,
                                    scaling_factor = scaling_factor,
                                    loss_function = 'ITSE',
                                    transpose = False
                                    )
        st.pyplot(fig_opt)

    col1, col2 = st.columns(2)

    with col1:
        df_rmse = pd.DataFrame({
                        'Relative RMSE':rel_rmse_list,
                        'Relative RMSE (Predictions)':rel_rmse_list_pred,
                    })

        st.dataframe(df_rmse, use_container_width=True)

    with col2:
        sig_params_all_reg = []

        wave_idx = 1
        for sig in sig_params:
            sig_dict = {}
            [A, tp, delta, nu] = sig
            sig_dict['Region'] = city_name
            sig_dict['Wave'] = wave_idx
            wave_idx += 1
            sig_dict['A'] = A*scaling_factor
            sig_dict['tp'] = tp
            sig_dict['delta'] = delta
            sig_dict['nu'] = nu
            sig_params_all_reg.append(sig_dict)

        df_sig_params = pd.DataFrame(sig_params_all_reg)

        st.dataframe(df_sig_params, use_container_width=True)

footer="""
<br><br>
<div style='text-align: center;'>
  <p>Developed by Frederico José Ribeiro Pelogia</p>
  <p>Data Source: <a src="https://github.com/pcm-dpc/COVID-19">https://github.com/pcm-dpc/COVID-19</a></p>
</div>
"""
st.markdown(footer, unsafe_allow_html=True)