import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
from classifier import Classifier
import pandas as pd

clf = Classifier()

small_dataset = pd.read_csv("small_dataset.csv")

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
    dcc.Store(id='tmdb-store', storage_type='session'),
    dcc.Store(id='comment-store', storage_type='session'),

    html.H1(['Toxic Comment Classification']),

    html.Div([html.Div([
        html.Div([
            html.H2('Enter your comment here'),
            dcc.Textarea(
                id='comment-input',
                cols=50,
                rows=5,
                placeholder='Type your comment here or paste randomly selected one',
            ),
            html.Br(),
            html.Button(id='predict-button', n_clicks=0, children='PREDICT', style={'color': 'rgb(255, 255, 255)'}),
        ], style={'padding': '10px',
                  'font-size': '22px',
                  'border': 'thick red solid',
                  'color': 'rgb(255, 255, 255)',
                  'backgroundColor': '#536869',
                  'textAlign': 'left',
                  }),

        html.Div([
            html.H2(['Each class probability'], style={'textAlign': 'center'}),
            html.Table([
                html.Thead([html.Td('class', style={'textAlign': 'center'}),
                            html.Td('probability', style={'textAlign': 'center'})]),
                html.Tbody([html.Tr([html.Td(['toxic'], style={'textAlign': 'center'}),
                                     html.Td(id='toxic', style={'textAlign': 'center'})]),
                            html.Tr([html.Td(['severe_toxic'], style={'textAlign': 'center'}),
                                     html.Td(id='severe_toxic', style={'textAlign': 'center'})]),
                            html.Tr([html.Td(['obscene'], style={'textAlign': 'center'}),
                                     html.Td(id='obscene', style={'textAlign': 'center'})]),
                            html.Tr([html.Td(['threat'], style={'textAlign': 'center'}),
                                     html.Td(id='threat', style={'textAlign': 'center'})]),
                            html.Tr([html.Td(['insult'], style={'textAlign': 'center'}),
                                     html.Td(id='insult', style={'textAlign': 'center'})]),
                            html.Tr([html.Td(['identity_hate'], style={'textAlign': 'center'}),
                                     html.Td(id='identity_hate', style={'textAlign': 'center'})])
                            ]),
            ], style={
                'border': 'thin black solid',
                'width': '70%',
                'marginLeft': 90,
            }),
        ]),
    ], className='six columns'),

        html.Div([
            html.Div('Randomly select a comment example'),
            html.Button(id='random-button', n_clicks=0, children='SELECT', style={'color': 'rgb(255, 255, 255)'}),
            html.Div(id='comment', children=[]),
        ], style={'padding': '12px',
                  'font-size': '22px',
                  'border': 'thick red solid',
                  'color': 'rgb(255, 255, 255)',
                  'backgroundColor': '#536869',
                  'textAlign': 'left',
                  },
            className='six columns')], className='twelve columns')
])


@app.callback(Output('tmdb-store', 'data'),
              [Input('random-button', 'n_clicks')],
              [State('tmdb-store', 'data')])
def on_click(n_clicks, data):
    if n_clicks is None:
        raise PreventUpdate
    elif n_clicks == 0:
        data = ""
    elif n_clicks > 0:
        data = small_dataset.sample(1)["comment_text"].item()
    return data


@app.callback(Output('comment', 'children'),
              Input('tmdb-store', 'data'))
def on_data(data):
    return data


@app.callback(Output('comment-store', 'data'),
              [Input('predict-button', 'n_clicks')],
              [State('comment-input', 'value')]
              )
def on_click(n_clicks, value):
    if n_clicks is None:
        raise PreventUpdate
    elif n_clicks == 0:
        data = 'This is good comment'
    elif n_clicks > 0:
        data = str(value)
    return data


@app.callback(
    Output('toxic', 'children'),
    Output('severe_toxic', 'children'),
    Output('obscene', 'children'),
    Output('threat', 'children'),
    Output('insult', 'children'),
    Output('identity_hate', 'children'),
    Input('comment-store', 'data'),
)
def callback_a(text):
    return clf.predict_text(text)


if __name__ == '__main__':
    app.run_server(debug=True)
