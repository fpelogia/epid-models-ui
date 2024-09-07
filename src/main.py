import streamlit as st
from streamlit_echarts import st_echarts
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#from optimize import fit_data
import optimize
from new_wave import get_transition_points

# Use full page width
st.set_page_config(
    layout="wide",
    page_title="COVID-19 Modelling",
    page_icon="🏥",
)

# Import data
data = pd.read_csv("../Datasets/italy_regions.csv") 


col1, col2 = st.columns([3, 1])
with col1:
    # Page title and sub header
    st.title('Multi-wave modelling and short-term prediction of ICU bed occupancy by patients with Covid-19 in regions of Italy')
    st.subheader('Math. Model. Nat. Phenom. 19 (2024) 13 - https://doi.org/10.1051/mmnp/2024012')
with col2:
    st.image("../assets/unifesp.png", width=400)

col1, col2 = st.columns(2)

with col1:
    # Allow region selection
    cities = ['Lombardia', 'Lazio', 'Campania', 'Veneto', 'Sicilia']
    city_name = st.selectbox('Region Name', cities)
with col2:
    # Allow indicator selection
    indicator = st.selectbox('Pandemic Indicator', ['ICU Admissions'], disabled = True)


st.session_state.disabled = False


use_transition_points = st.radio(
        "Transition Points",
        ["Same as paper", "Calculated", "User defined"],
        key="visibility",
        label_visibility="visible",
        disabled=st.session_state.disabled,
        horizontal=True,
    )

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

    # Initial Conditions
    col1, col2, col3, col4 = st.columns([1, 1, 2, 3])
    with col1:
        tp_threshold = st.number_input(
            "Threshold",
            disabled=False,
            value=tp_threshold,
            format="%0.1e",
            step=1e-6,
            max_value=1.0,
            min_value=1e-6
        )
    with col2:
        scaling_factor = st.number_input(
            "Scaling Factor",
            disabled=False,
            value=scaling_factor
        )


    # Transition Points
    x_nw, fig, tran_pts_fig_series_data = get_transition_points(scaling_factor*acc_data, visual=True, threshold=tp_threshold, indicator = indicator, city_name=city_name)

    if use_transition_points == 'User defined':
       
        # Allow user to adjust transition points
        # st.sidebar.title("Adjust Transition Points")
        # user_transition_points = [
        #     st.sidebar.slider(f'Transition Point {i+1}', 0, len(tran_pts_fig_series_data['normalized_acc_n_cases']) - 1, value=x)
        #     for i, x in enumerate(x_nw)
        # ]
        # save_btn = st.sidebar.button('Run Model')
        
        # if save_btn:
        #     transition_points = user_transition_points
        # else:
        #     transition_points = x_nw

        # Form to adjust transition points
        with st.sidebar.form(key='transition_points_form'):
            st.title("Adjust Transition Points")
            transition_points = [
                st.slider(f'Transition Point {i+1}', 0, len(tran_pts_fig_series_data['normalized_acc_n_cases']) - 1, value=x)
                for i, x in enumerate(tran_pts_fig_series_data['x_t'])
            ]
            # Submit button to apply changes
            submit_button = st.form_submit_button(label='Apply Changes')



    else:
        transition_points = x_nw

    if use_transition_points == 'Same as paper':
        # Use same transition points as in paper

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

    elif use_transition_points == 'User defined':
        # Use user defined transition points
        x_nw = transition_points

    else: # Calculated
        pass # x_nw already contains the calculated points
    

    with col3:
        # Show transition points being used
        st.text_input(
            "Transition Points:",
            disabled=True,
            value=x_nw
        )

    col1, col2 = st.columns([1, 2])

    with col1:
        tab1, tab2 = st.tabs(['Interactive Chart (Echarts)', 'Regular Chart (Matplotlib)'])

        with tab2:
            # regular matplotlib plot
            st.pyplot(fig)   
            
        with tab1:

            # plotting with streamlit echarts
            options = {
                "title": {
                    "text": f"{city_name}",
                    "left": "center",
                    "top": 10,
                    "textStyle": {"fontSize": 24}
                },
                "grid": [
                    {"top": "15%", "left": "5%", "height": "25%", "width": "90%"},
                    {"top": "45%", "left": "5%", "height": "25%", "width": "90%"},
                    {"top": "75%", "left": "5%", "height": "20%", "width": "90%"}
                ],
                "xAxis": [
                    {"gridIndex": 0, "type": "category", "data": list(range(len(tran_pts_fig_series_data['normalized_acc_n_cases'])))},
                    {"gridIndex": 1, "type": "category", "data": list(range(len(tran_pts_fig_series_data['unf_daily_n_cases'])))},
                    {"gridIndex": 2, "type": "category", "data": list(range(len(tran_pts_fig_series_data['sec_der'])))}
                ],
                "yAxis": [
                    {"gridIndex": 0, "type": "value"},
                    {"gridIndex": 1, "type": "value"},
                    {"gridIndex": 2, "type": "value", "min": -2e-4, "max": 2e-4}
                ],
                "legend" : {
                    "show": True,
                    "position": "bottom",  
                    "top": "5%",
                    "left": "center",
                    "width": "auto",
                    "height": "auto",
                    "data": ["Normalized accumulated number of cases", "New wave transition", "First derivative", "Filtered first derivative", "Second derivative", "Threshold"],                    
                },
                "series": [
                    {
                        "name": "Normalized accumulated number of cases",
                        "type": "line",
                        "xAxisIndex": 0,
                        "yAxisIndex": 0,
                        "data": list(tran_pts_fig_series_data['normalized_acc_n_cases']),
                        "markLine": {
                            "data": [{"xAxis": x} for x in transition_points],
                            "lineStyle": {"type": "dotted"},
                            "symbol": "none",  # Ensure no arrow tips
                            "draggable": True  # Make the lines draggable
                        },
                        "color": "#1f77b4" # default blue (plt)
                    },
                    {
                        "name": "First derivative",
                        "type": "line",
                        "xAxisIndex": 1,
                        "yAxisIndex": 1,
                        "data": list(tran_pts_fig_series_data['unf_daily_n_cases']),
                        "markLine": {
                            "data": [{"xAxis": x} for x in transition_points],
                            "lineStyle": {"type": "dotted"},
                            "symbol": "none",
                            "draggable": True
                        },
                        "color": "#9467bd"
                    },
                    {
                        "name": "Filtered first derivative",
                        "type": "line",
                        "xAxisIndex": 1,
                        "yAxisIndex": 1,
                        "data": list(tran_pts_fig_series_data['daily_n_cases']),
                        "markLine": {
                            "data": [{"xAxis": x} for x in transition_points],
                            "lineStyle": {"type": "dotted"},
                            "symbol": "none",
                            "draggable": True
                        },
                        "color": "#2ca02c" # default green (plt)
                    },
                    {
                        "name": "Second derivative",
                        "type": "line",
                        "xAxisIndex": 2,
                        "yAxisIndex": 2,
                        "data": list(tran_pts_fig_series_data['sec_der']),
                        "markLine": {
                            "data": [{"xAxis": x} for x in transition_points],
                            "lineStyle": {"type": "dotted"},
                            "symbol": "none",
                            "draggable": True
                        },
                        "color": "#ff7f0e" # default orange (plt)
                    },
                    {
                        "name": "Threshold",
                        "type": "line",
                        "xAxisIndex": 2,
                        "yAxisIndex": 2,
                        "data": [-1*tran_pts_fig_series_data['abs_threshold']]*len(tran_pts_fig_series_data['sec_der']),
                        "lineStyle": {"type": "dashed"},
                        "color": "gray"
                    },
                    # Additional series definitions remain the same
                ]
            }

            st_echarts(options=options, height=700, width=500)

    # Right Column - Optimization results - model fit and predictions
    with col2:

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