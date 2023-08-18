import dash
from dash import dcc
from dash import html 
import dash_bootstrap_components as dbc
import pandas as pd
import uuid

from dash.dependencies import Input, Output
from app import app
from typing import List, Union
from apps.nowcast.real_time_net_delay_nowcast import render as render_surface_summary
from apps.historical_view.real_time_net_delay_historical_view import render as render_historical_graph
from apps.delay.delay_view import render as render_delay
from apps.edct.edct_view import render as render_edct
from apps.apreq.apreq_view import render as render_apreq
# from apps.prediction.real_time_net_delay_prediction import render as render_prediction
# from apps.daily_prediction.real_time_net_delay_prediction_daily import render as render_prediction_daily

navs = [
    "nowcast",
    "historical_view",
    "prediction_sliding_window",
    "prediction_daily",
    "delay",
    "edct",
    "apreq"
]

nav_links = [
    dbc.NavLink(
        m.replace('_', ' ').upper(),
        href=f"/{m}",
        id=f"page-{m}-link",
        active="exact"
    )
    for m in navs
]

sidebar = html.Div(
    children=[
        html.H2(
            children="Real time Network Delay Monitor",
            className="display-5",
        ),

        html.Hr(),

        dbc.Nav(
            nav_links,
            vertical=True,
            pills=True,
        ),
    ],
    style={
        "position": "fixed",
        "top": 0,
        "left": 0,
        "bottom": 0,
        "width": "16rem",
        "padding": "2rem 1rem",
        "backgroundColor": "#f8f9fa",
    },
)

content = html.Div(
    id="page-content",
    style={
        "marginLeft": "18rem",
        "marginRight": "2rem",
        "padding": "2rem 1rem",
    },
)

def serve_layout() -> html.Div:
    return html.Div(
        children=[
            dcc.Location(
                id="url",
            ),
            
            dcc.Store(data=str(uuid.uuid4()), id='session-id'),
            
            dcc.Store(id='store'),

            sidebar,

            content,
        ],
    )
    

app.layout = serve_layout
server = app.server

@app.callback(
    [
        Output(f"page-{m}-link", "active")
        for m in navs
    ],
    [
        Input("url", "pathname"),
    ],
)
def toggle_active_links(
        pathname: str,
) -> List[bool]:
    app.logger.info(f"pathname: {pathname}")
    if pathname == "/":
        return [True if m == "nowcast" else False for m in navs]
    return [pathname == f"/{m}" for m in navs]


@app.callback(
    Output("page-content", "children"),
    [
        Input("url", "pathname"),
    ],
)
def render_page_content(
        pathname: str,
) -> html.Div:
    key = pathname.replace("/", "")
    if (pathname == "/"):
        key = "nowcast"

    if key == 'nowcast':
        return render_surface_summary()
    elif key == 'historical_view':
        return render_historical_graph()
    elif key == 'delay':
        return render_delay()
    elif key == 'edct':
        return render_edct()
    elif key == 'apreq':
        return render_apreq()
    elif key == 'prediction_sliding_window':
        return render_prediction()
    elif key == 'prediction_daily':
        return render_prediction_daily()
    else:
        return html.Div(
            dbc.Container(
            [
                html.H1("404: Not found", className="display-3"),
                html.Hr(),
                html.P(f"The pathname {pathname} was not recognized..."),
            ]
            )
        )

if __name__ =='__main__':
    app.run_server(debug=True)