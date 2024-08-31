import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from optimize import fit_data
from new_wave import get_transition_points
# imports for debugging purposes
import readline
import code

def main():
    # Import data
    data = pd.read_csv("../Datasets/rosario_obitos.csv") 
    city_name = 'Rosario' 

    try:
        #deaths
        acc_data = data.cumulative_deceased
    except(AttributeError):
        # cases
        acc_data = data.total_confirmed

    normalized_acc_data = acc_data / max(acc_data)
    t = np.linspace(0, len(acc_data)-1, len(acc_data))

    normalized_acc_data = normalized_acc_data.tolist()
    daily_data = data.new_deceased.tolist()

    # Transition Points
    if city_name == 'Rosario':
        x_nw = [295, 360, 515] # Manual Rosario
    elif city_name == 'Buenos Aires':
        x_nw = [300, 400, 520] # Manual Buenos Aires
    elif city_name == 'Mendoza':
        x_nw = [315, 400, 510] # Manual Mendoza
    elif city_name == 'Córdoba':
        x_nw = [300, 400, 520] # Manual Córdoba
    elif city_name == 'São José dos Campos':
        #x_nw = [147, 287, 382, 669] #(SJC)
        x_nw = [147, 287, 382, 669] #(SJC)
    elif city_name == 'São Paulo':
        x_nw = [89, 254, 309, 370, 683] # (SP)
    elif city_name == 'Campinas':
        #x_nw  = [94, 146, 269, 314, 377, 469, 607, 677, 741] # Campinas tr 2e-5
        x_nw = [100, 149, 471, 610, 681, 741] # Campinas tr 3e-5
    else:
        x_nw = [300, 400, 520] 

    x_nw_manual = x_nw     
    fit_data(acc_data, daily_data, city_name, x_nw)

    x_nw = get_transition_points(acc_data, visual=True)   
    
    fit_data(acc_data, daily_data, city_name, x_nw)

    print('Manual: ', x_nw_manual)
    print('new_wave_detection: ', x_nw)

    # Just to wait
    plt.show()
    

if __name__ == "__main__":
    main()

#====== Interactive Debug ======
# variables = globals().copy()
# variables.update(locals())
# shell = code.InteractiveConsole(variables)
# shell.interact()
#===============================