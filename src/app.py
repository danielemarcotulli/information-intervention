# app.py
import dash
from dash import html, dcc, Input, Output, State
import dash_bootstrap_components as dbc

import numpy as np
import pandas as pd
from scipy.integrate import odeint
import plotly.graph_objs as go

from cmdstanpy import CmdStanModel
import os

import networkx as nx
import random

# compile the Stan models 
bounded_back_model = CmdStanModel(stan_file='bounded_back.stan')
moving_se_sp_model = CmdStanModel(stan_file='moving_se_sp.stan')

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server  

# Define math functions
def sigmoid(x, max_value, k):
    return max_value / (1 + np.exp(-k * x))

def saturation(P, K):
    return 1 - P / K

def model_network_density(time, base_density, peak_density, peak_months):
    month = int((time % 12) + 1)
    if month in peak_months:
        return peak_density
    else:
        return base_density

def influence_practitioner(practitioner, S_D, HS, k, w_d, p_d):
    if practitioner == 'influenced':
        q_d = k * HS * S_D * w_d
    else:
        q_d = 0.6 * p_d
    return q_d

def exponential_diagnosis(form, b, P, K):
    if form == 'exponential':
        w_d = 1 - np.exp(-b * P)
    else:
        w_d = saturation(P, K)
    return w_d

def spread_awareness_network(S, A, D, HS, time_steps):
    # Step 1: Define parameters for the small-world network
    n = int(S)  # Number of nodes in the susceptible population
    k = 4  # Each node is connected to k nearest neighbors in ring topology
    p_rewire = 0.01  # The probability of rewiring each edge

    if n <= 0:
        return 0  # No susceptible individuals to spread awareness

    # Step 2: Generate the small-world network
    G = nx.watts_strogatz_graph(n, k, p_rewire)

    # the number of aware nodes does not exceed the population size (otherwie this throws error)
    initially_aware = min(int(A), n)

    # Step 3: Initialize the nodes' awareness status
    awareness = np.zeros(n, dtype=int)
    if initially_aware > 0:
        aware_nodes = np.random.choice(range(n), size=initially_aware, replace=False)
        awareness[aware_nodes] = 1

    # Step 4: Simulate the spread of awareness over multiple time steps
    for _ in range(int(time_steps)):
        new_awareness = awareness.copy()
        aware_indices = np.where(awareness == 1)[0]
        for node in aware_indices:
            neighbors = list(G.neighbors(node))
            for neighbor in neighbors:
                if awareness[neighbor] == 0:
                    if random.random() < p_rewire:
                        new_awareness[neighbor] = 1
        awareness = new_awareness

    # Return the number of new aware individuals after time_steps
    return np.sum(awareness) - initially_aware

def spread_mode(awareness_mode, S, A, D, HS, lambd, rho, time_steps, D_weight, HS_weight, A_weight, N):
    if awareness_mode == 'small_world':
        new_aware_count = spread_awareness_network(S, A, D, HS, time_steps)
    else:
        # Spread awareness using the sigmoid function
        influence_factor = ((D * D_weight) + (HS * HS_weight) + ((A - HS - D) * A_weight)) / N
        S_A = sigmoid(influence_factor, 0.8, k=0.5)
        new_aware_count = S_A * lambd * S * rho
    return new_aware_count

# Define the differential equation model function
def model(state, time, parameters):
    S, A, HS, D = state
    N = parameters['N']
    alpha = parameters['alpha']
    lambd = parameters['lambda']
    eta = parameters['eta']
    omega = parameters['omega']
    mu = parameters['mu']
    epsilon = parameters['lambda_fade']
    k = parameters['k']
    b = parameters['b']
    practitioner = parameters['practitioner']
    form = parameters['FunctionalForm']
    awareness_mode = parameters['awareness_mode']

    # Additional parameters
    base_density = parameters['base_density']
    peak_density = parameters['peak_density']
    peak_months = parameters['peak_months']
    D_weight = parameters['D_weight']
    HS_weight = parameters['HS_weight']
    A_weight = parameters['A_weight']
    K_saturation = parameters['K_saturation']  # capacity of mental health system
    k_sigmoid = parameters['k_sigmoid']
    max_S_A = parameters['max_S_A']
    max_S_HS = parameters['max_S_HS']
    K_exponential_diagnosis = parameters['K_exponential_diagnosis']  # K in exponential_diagnosis

    # Calculate network density
    rho = model_network_density(time, base_density, peak_density, peak_months)

    # Influence factor
    influence_factor = ((D * D_weight) + (HS * HS_weight) + ((A - HS - D) * A_weight)) / N

    # Sigmoidal effects of state transitions
    S_A = sigmoid(influence_factor, max_S_A, k_sigmoid)
    S_HS = sigmoid(influence_factor, max_S_HS, k_sigmoid)

    # Saturation effect on the rate of diagnoses
    S_D = saturation(D / N, K_saturation)

    # Awareness spread using spread_mode
    r_d = spread_mode(awareness_mode, S, A, D, HS, lambd, rho, time_steps=1,
                      D_weight=D_weight, HS_weight=HS_weight, A_weight=A_weight, N=N)

    # Transition from awareness to help-seeking
    p_d = S_HS * eta * A * rho

    # Willingness to diagnose
    w_d = exponential_diagnosis(form, b, D / N, K_exponential_diagnosis)

    # Influence of practitioner on diagnosis
    q_d = influence_practitioner(practitioner, S_D, HS, k, w_d, p_d)

    # Differential equations for state transitions
    dA = alpha * S * (A / N) + r_d - epsilon * A
    dHS = p_d - omega * HS
    dD = q_d - mu * D
    dS = -dA

    return [dS, dA, dHS, dD]

#### Prevalence calculation function ###
def prevalence_calc(sim_out_df, prevalence_method, Se, Sp, N):
    D = sim_out_df['D'].values
    HS = sim_out_df['HS'].values
    time = sim_out_df['time'].values

    # Ensure diagnosed cases do not exceed help-seeking individuals
    D = np.minimum(D, HS)

    output_df = pd.DataFrame({
        'time': time,
        'clinic_prevalence': np.nan,
        'true_positives': np.nan,
        'false_positives': np.nan,
        'diagnosed_cases': D
    })

    if prevalence_method == 'back-calculate':
        output_df['clinic_prevalence'] = (D / HS - (1 - Sp)) / (Se + Sp - 1)
        output_df['true_positives'] = output_df['clinic_prevalence'] * HS * Se
        output_df['false_positives'] = (1 - output_df['clinic_prevalence']) * HS * (1 - Sp)

    elif prevalence_method == 'simple':
        output_df['clinic_prevalence'] = D / HS
        output_df['true_positives'] = output_df['clinic_prevalence'] * HS * Se
        output_df['false_positives'] = (1 - output_df['clinic_prevalence']) * HS * (1 - Sp)

    elif prevalence_method == 'bayesian_moving_se_sp':
        # Prepare data for Stan model
        stan_data = {
            'N': len(D),
            'D': np.round(D).astype(int),
            'H': np.round(HS).astype(int),
            't': np.arange(1, len(D) + 1),
            'Se': Se,
            'Sp': Sp,
            'trend_scale_se': 0.005,
            'trend_scale_sp': 0.008
        }

        # Run Stan model
        fit = moving_se_sp_model.sample(data=stan_data, chains=4, parallel_chains=4, iter_sampling=2000, iter_warmup=1000)
        output_df['clinic_prevalence'] = fit.stan_variable('apparent_prev').mean(axis=0)
        output_df['true_positives'] = fit.stan_variable('true_positives').mean(axis=0)
        output_df['false_positives'] = fit.stan_variable('false_positives').mean(axis=0)
        output_df['diagnosed_cases'] = fit.stan_variable('diagnosed_cases').mean(axis=0)

    elif prevalence_method == 'bayesian_bounded_back_calculation':
        # Prepare data for Stan model
        stan_data = {
            'N': len(D),
            'D': np.round(D).astype(int),
            'H': np.round(HS).astype(int),
            't': np.arange(1, len(D) + 1),
            'Se': Se,
            'fp_influx_rate_alfa': 1.0,
            'fp_influx_rate_beta': 1.0
        }

        # Run Stan model
        fit = bounded_back_model.sample(data=stan_data, chains=4, parallel_chains=4, iter_sampling=2000, iter_warmup=1000)
        output_df['clinic_prevalence'] = fit.stan_variable('tested_prev_gen').mean(axis=0)
        output_df['true_positives'] = fit.stan_variable('true_positives').mean(axis=0)
        output_df['false_positives'] = fit.stan_variable('false_positives').mean(axis=0)
        output_df['diagnosed_cases'] = D  # Use observed diagnosed cases

    else:
        raise ValueError("Invalid prevalence calculation method.")

    # Ensure all clinic prevalence values are within 0-1
    output_df['clinic_prevalence'] = np.clip(output_df['clinic_prevalence'], 0, 1)

    return output_df

####### Simulation function #####################
def run_simulation(input_params):
    # Initial state
    N = input_params['N']
    A_initial = 0.2 * N
    HS_initial = 0.05 * N
    D_initial = 0.02 * N
    S_initial = N - A_initial - HS_initial - D_initial
    initial_state = [S_initial, A_initial, HS_initial, D_initial]

    # Time steps
    time = np.arange(1, 50)

    ###### Parameters   ###########
    parameters = input_params

    # Add parameters with their default values 
    parameters.setdefault('base_density', 0.5)
    parameters.setdefault('peak_density', 1.0)
    parameters.setdefault('peak_months', [2, 3, 4, 5, 10, 11])
    parameters.setdefault('D_weight', 1.0)
    parameters.setdefault('HS_weight', 1.0)
    parameters.setdefault('A_weight', 1.0)
    parameters.setdefault('K_saturation', 0.3)
    parameters.setdefault('k_sigmoid', 0.5)
    parameters.setdefault('max_S_A', 0.8)
    parameters.setdefault('max_S_HS', 0.6)
    parameters.setdefault('K_exponential_diagnosis', 0.5)

    # Run the simulation
    sim_out = odeint(model, initial_state, time, args=(parameters,))
    sim_out_df = pd.DataFrame(sim_out, columns=['S', 'A', 'HS', 'D'])
    sim_out_df['time'] = time

    # Ensure non-negative values
    sim_out_df[['S', 'A', 'HS', 'D']] = sim_out_df[['S', 'A', 'HS', 'D']].clip(lower=0)

    # Calculate prevalence and true/false positives
    prevalence_results = prevalence_calc(sim_out_df, input_params['prevalence_method'], input_params['Se'], input_params['Sp'], N)

    # Calculate responders and harmed
    prevalence_results['responders'] = prevalence_results['true_positives'] / input_params['nnt']
    prevalence_results['harmed'] = prevalence_results['diagnosed_cases'] / input_params['nnh']

    return sim_out_df, prevalence_results

####### Layout  ########
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("Awareness and Diagnoses Spread Model"), className="text-center")
    ]),
    dbc.Row([
        dbc.Col([
            html.Label('Total population size:'),
            dcc.Input(id='N', type='number', value=1000000, min=100, max=10000000),
            html.Br(),
            html.Label('Rate of awareness transmission per media campaigns (alpha):'),
            dcc.Input(id='alpha', type='number', value=0.05, min=0, max=1, step=0.01),
            html.Br(),
            html.Label('Awareness spread rate due to contact (lambda):'),
            dcc.Input(id='lambda', type='number', value=0.05, min=0, max=1, step=0.01),
            html.Br(),
            html.Label('Rate of seeking help from aware (eta):'),
            dcc.Input(id='eta', type='number', value=0.05, min=0, max=1, step=0.01),
            html.Br(),
            html.Label('Help Seeking recovery rate (omega):'),
            dcc.Input(id='omega', type='number', value=0.01, min=0, max=1, step=0.005),
            html.Br(),
            html.Label('Illness recovery rate (mu):'),
            dcc.Input(id='mu', type='number', value=0.01, min=0, max=1, step=0.005),
            html.Br(),
            html.Label('Rate of awareness fading (lambda_fade):'),
            dcc.Input(id='lambda_fade', type='number', value=0.02, min=0, max=1, step=0.005),
            html.Br(),
            html.Label('Base diagnosis rate (k):'),
            dcc.Input(id='k', type='number', value=0.4, min=0, max=1, step=0.1),
            html.Br(),
            html.Label('Practitioner influence factor (b):'),
            dcc.Input(id='b', type='number', value=1.1, min=0, max=2, step=0.1),
            html.Br(),
            html.Label('Practitioner type:'),
            dcc.Dropdown(
                id='practitioner',
                options=[
                    {'label': 'Influenced', 'value': 'influenced'},
                    {'label': 'Not Influenced', 'value': 'not_influenced'}
                ],
                value='influenced'
            ),
            html.Br(),
            html.Label('Function Form:'),
            dcc.Dropdown(
                id='FunctionalForm',
                options=[
                    {'label': 'Exponential', 'value': 'exponential'},
                    {'label': 'Linear', 'value': 'linear'}
                ],
                value='exponential'
            ),
            html.Br(),
            html.Label('Awareness Mode:'),
            dcc.Dropdown(
                id='awareness_mode',
                options=[
                    {'label': 'Small World Network', 'value': 'small_world'},
                    {'label': 'Sigmoid Function', 'value': 'sigmoid'}
                ],
                value='sigmoid'
            ),
            html.Br(),
            html.Label('Sensitivity (Se):'),
            dcc.Input(id='Se', type='number', value=0.6, min=0, max=1, step=0.05),
            html.Br(),
            html.Label('Specificity (Sp):'),
            dcc.Input(id='Sp', type='number', value=0.6, min=0, max=1, step=0.05),
            html.Br(),
            html.Label('Number needed to treat (nnt):'),
            dcc.Input(id='nnt', type='number', value=8, min=1, max=100, step=1),
            html.Br(),
            html.Label('Number needed to harm (nnh):'),
            dcc.Input(id='nnh', type='number', value=8, min=1, max=100, step=1),
            html.Br(),
            html.Label('Prevalence calculation method:'),
            dcc.Dropdown(
                id='prevalence_method',
                options=[
                    {'label': 'Back-calculate', 'value': 'back-calculate'},
                    {'label': 'Simple', 'value': 'simple'},
                    {'label': 'Bayesian Moving Se Sp', 'value': 'bayesian_moving_se_sp'},
                    {'label': 'Bayesian Bounded Back Calculation', 'value': 'bayesian_bounded_back_calculation'}
                ],
                value='back-calculate'
            ),
            html.Br(),
            html.Button('Run Simulation', id='run-button', n_clicks=0),
        ], width=3),
        dbc.Col([
            dcc.Graph(id='simulation-plot'),
            dcc.Graph(id='prevalence-plot'),
            dcc.Graph(id='harm-plot')
        ], width=9)
    ])
], fluid=True)

##### Callbacks #############
@app.callback(
    [Output('simulation-plot', 'figure'),
     Output('prevalence-plot', 'figure'),
     Output('harm-plot', 'figure')],
    [Input('run-button', 'n_clicks')] +
    [State('N', 'value'),
     State('alpha', 'value'),
     State('lambda', 'value'),
     State('eta', 'value'),
     State('omega', 'value'),
     State('mu', 'value'),
     State('lambda_fade', 'value'),
     State('k', 'value'),
     State('b', 'value'),
     State('practitioner', 'value'),
     State('FunctionalForm', 'value'),
     State('awareness_mode', 'value'),
     State('Se', 'value'),
     State('Sp', 'value'),
     State('nnt', 'value'),
     State('nnh', 'value'),
     State('prevalence_method', 'value')]
)
def update_graph(n_clicks, N, alpha, lambd, eta, omega, mu, lambda_fade, k, b,
                 practitioner, FunctionalForm, awareness_mode, Se, Sp, nnt, nnh, prevalence_method):
    if n_clicks == 0:
        return dash.no_update

    input_params = {
        'N': N,
        'alpha': alpha,
        'lambda': lambd,
        'eta': eta,
        'omega': omega,
        'mu': mu,
        'lambda_fade': lambda_fade,
        'k': k,
        'b': b,
        'practitioner': practitioner,
        'FunctionalForm': FunctionalForm,
        'awareness_mode': awareness_mode,
        'Se': Se,
        'Sp': Sp,
        'nnt': nnt,
        'nnh': nnh,
        'prevalence_method': prevalence_method,
        # Additional parameters with default values - need to change
        'base_density': 0.5,
        'peak_density': 1.0,
        'peak_months': [2, 3, 4, 5, 10, 11],
        'D_weight': 1.0,
        'HS_weight': 1.0,
        'A_weight': 1.0,
        'K_saturation': 0.3,
        'k_sigmoid': 0.5,
        'max_S_A': 0.8,
        'max_S_HS': 0.6,
        'K_exponential_diagnosis': 0.5,
    }

    sim_out_df, prevalence_results = run_simulation(input_params)

    # Simulation plot
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=sim_out_df['time'], y=sim_out_df['S'], mode='lines', name='Susceptible'))
    fig1.add_trace(go.Scatter(x=sim_out_df['time'], y=sim_out_df['A'], mode='lines', name='Aware'))
    fig1.add_trace(go.Scatter(x=sim_out_df['time'], y=sim_out_df['HS'], mode='lines', name='Help Seeking'))
    fig1.add_trace(go.Scatter(x=sim_out_df['time'], y=sim_out_df['D'], mode='lines', name='Diagnosed'))
    fig1.update_layout(title='Dynamics of Awareness and Diagnoses Spread', xaxis_title='Time', yaxis_title='Number of People')

    # Prevalence plot
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=prevalence_results['time'], y=prevalence_results['true_positives'], mode='lines', name='True Positives'))
    fig2.add_trace(go.Scatter(x=prevalence_results['time'], y=prevalence_results['diagnosed_cases'], mode='lines', name='Diagnosed Cases'))
    fig2.add_trace(go.Scatter(x=sim_out_df['time'], y=sim_out_df['HS'], mode='lines', name='Help Seeking'))
    fig2.update_layout(title='True Positives, Diagnosed, and Help Seekers Over Time', xaxis_title='Time', yaxis_title='Number of People')

    # Harm plot
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=prevalence_results['time'], y=prevalence_results['responders'], mode='lines', name='Responders'))
    fig3.add_trace(go.Scatter(x=prevalence_results['time'], y=prevalence_results['harmed'], mode='lines', name='Harmed'))
    fig3.update_layout(title='Dynamics of Responders and Harmed Population', xaxis_title='Time', yaxis_title='Number of People')

    return fig1, fig2, fig3

if __name__ == "__main__":
    app.run_server(debug=True)
