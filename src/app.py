import dash
from dash import dcc
from dash import html
import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from finite_element_model import lateral_pile_analysis
import plotly.io as pio

pio.templates.default = "plotly_white"

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

app.layout = html.Div([
    html.H1('PISA - finite element analysis for laterally loaded piles'),
    html.P('This is a demonstration app, it should not be used in decision making processes without supervision from a qualified engineer. For consultancy services on software implementation or geotechnical analysis, please contact info@offshorewind.io'),

    dbc.Row(
        [
            dbc.Col(html.Div([
                html.Label('Pile Diameter (m)'),
                dbc.Input(id='pile-diameter', type='number', value=8.75),
                html.Label('Pile Embedded Length (m)'),
                dbc.Input(id='pile-embedded-length', type='number', value=35),
                html.Label('Pile wall thickness (mm)'),
                dbc.Input(id='pile-wall-thickness', type='number', value=150),
                html.Label('Number of elements (-)'),
                dbc.Input(id='number-of-elements', type='number', value=20, disabled=True),
                html.Label('Maximum horizontal load (N)'),
                dbc.Input(id='H-max', type='number', value=60000000),
                html.Label('Height of load application (m)'),
                dbc.Input(id='height', type='number', value=-87.5),
                html.Label('Soil type'),
                dbc.Select(id="soil-type", value="sand",
                    options=[
                        {"label": "sand", "value": "sand"},
                        {"label": "clay", "value": "clay"},
                    ],
                ),
            ]), md=3),
            dbc.Col(dcc.Graph(id='graph'), md=9),
        ])
])
@app.callback(
    dash.dependencies.Output('graph', 'figure'),
    [dash.dependencies.Input('pile-diameter', 'value'),
     dash.dependencies.Input('pile-embedded-length', 'value'),
     dash.dependencies.Input('pile-wall-thickness', 'value'),
     dash.dependencies.Input('number-of-elements', 'value'),
     dash.dependencies.Input('H-max', 'value'),
     dash.dependencies.Input('height', 'value'),
     dash.dependencies.Input('soil-type', 'value')])
def update_figure(D, L, t_mm, n_elements, H_max, h, soil_type):

    pile = {
        'D': D,
        't': t_mm / 1000,
        'L': L,
        'n_elements': n_elements,
    }

    soil = {
        'type': soil_type,
        'k_lateral': 1,
        'k_distributed_moment': 0,
        'k_base_shear': 0,
        'k_base_moment': 0,
        'Dr': 0.85,
        'sigma_v': np.array([0, 20, 40, 50, 80, 130, 300, 400]) * 1e3,
        'z': [0, 2, 4, 5, 8, 13, 30, 40],
        'su': np.array([60, 160, 120, 105, 115, 130, 185, 210]) * 1e3,
        'G': np.array([2, 35, 70, 85, 130, 160, 330, 380]) * 1e6
    }

    load = {
        'H_max': H_max,
        'h': h,
        'n_steps': 10
    }

    results = lateral_pile_analysis(pile, soil, load)

    fig = make_subplots(rows=1, cols=6, shared_yaxes=True, subplot_titles=(
        "Displacement", "Rotation", "Bending moment", "Shear force", "Lateral pressure", "Distributed moment"))

    fig.add_trace(go.Scatter(x=results['v_interp'], y=results['z_interp'], name='v', line={'color': 'firebrick'}), 1, 1)
    fig.add_trace(go.Scatter(x=results['phi_interp'], y=results['z_interp'], name='φ', line={'color': 'firebrick'}), 1, 2)
    fig.add_trace(go.Scatter(x=results['M_interp'], y=results['z_interp'], name='M', line={'color': 'firebrick'}), 1, 3)
    fig.add_trace(go.Scatter(x=results['V_interp'], y=results['z_interp'], name='V', line={'color': 'firebrick'}), 1, 4)
    fig.add_trace(go.Scatter(x=results['p_interp'], y=results['z_interp'], name='p', line={'color': 'firebrick'}), 1, 5)
    fig.add_trace(go.Scatter(x=results['m_interp'], y=results['z_interp'], name='m', line={'color': 'firebrick'}), 1, 6)

    fig['layout']['yaxis']['title'] = 'Depth (m)'
    fig['layout']['xaxis']['title'] = 'v (m)'
    fig['layout']['xaxis2']['title'] = 'φ (rad)'
    fig['layout']['xaxis3']['title'] = 'M (Nm)'
    fig['layout']['xaxis4']['title'] = 'V (N)'
    fig['layout']['xaxis5']['title'] = 'p (N/m)'
    fig['layout']['xaxis6']['title'] = 'm (Nm/m)'

    fig.update_yaxes(autorange="reversed")
    fig.update_layout(height=600)

    return fig

if __name__ == '__main__':
    app.run_server(debug=True)
