from pickletools import stringnl_noescape_pair
import numpy as np

'''
f(t) - Richards Model (Assymetric Sigmoid)
t: time array
A, tp, delta, nu: model parameters
'''
def f_t(t, A, tp, delta, nu ):
    return A / ((1 + nu * np.exp(-1*(t - tp)/(delta)))**(1/nu))

# Vectorized version of f(t)
f_t = np.vectorize(f_t)

'''
f'(t) -Derivative of f(t) with respect to t
'''
def deriv_f_t(t, A, tp, delta, nu ):
    g = lambda x: np.exp(-1*(t - tp)/delta)
    return (A * g(t))/(delta * (1 + nu*g(t))**((nu+1)/nu))

# Vectorized version of df(t)/dt
deriv_f_t = np.vectorize(deriv_f_t)

# ===================================

def model(t, A, tp, delta, nu):
    res = np.zeros(n_days)
    for i in range(n_sig - 1):
        [A_i, tp_i, delta_i, nu_i] = sig_params[i]
        res += f_t(t[:n_days], A_i, tp_i, delta_i, nu_i)

    res += f_t(t[:n_days], A, tp, delta, nu)
    return res

def model_daily(t, A, tp, delta, nu):
    res = np.zeros(n_days)
    for i in range(n_sig - 1):
        [A_i, tp_i, delta_i, nu_i] = sig_params[i]
        res += deriv_f_t(t[:n_days], A_i, tp_i, delta_i, nu_i)

    res += deriv_f_t(t[:n_days], A, tp, delta, nu)
    return res

def share_variables(n_days_p, n_sig_p, sig_params_p):
    global n_days
    global n_sig
    global sig_params
    #print('SHARING: ', n_days_p, n_sig_p, sig_params_p)
    n_days = n_days_p
    n_sig = n_sig_p
    sig_params = sig_params_p
