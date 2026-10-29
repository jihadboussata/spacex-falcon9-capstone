"""Explore the historical course snapshot: python spacex_dash_app.py."""
import argparse
import math
from pathlib import Path

import dash
from dash import Input, Output, dcc, html
import numpy as np
import pandas as pd
import plotly.express as px

from spacex.train import LEGACY_PAYLOAD_FILL

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data/dataset_part_2.csv"
spacex_df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
spacex_df.loc[np.isclose(spacex_df.PayloadMass, LEGACY_PAYLOAD_FILL, rtol=0, atol=1e-7), "PayloadMass"] = np.nan
spacex_df["Landing outcome"] = spacex_df.Class.map({0: "Unsuccessful / no landing", 1: "Successful"})
MAX_PAYLOAD = int(math.ceil(spacex_df.PayloadMass.max()/1000)*1000)
COLORS = {"Successful": "#007f86", "Unsuccessful / no landing": "#ce6c31"}
app = dash.Dash(__name__)
app.title = "Falcon 9 · launch explorer"


def card(label, value):
    return html.Div([html.Div(label, style={"color": "#59677a", "fontSize": "14px"}),
        html.Strong(value, style={"fontSize": "28px"})],
        style={"padding": "20px", "background": "#f0f5f8", "borderRadius": "12px", "flex": "1", "minWidth": "150px"})


app.layout = html.Main([
    html.P("JIHAD BOUSSATA · DATA SCIENCE PORTFOLIO", style={"color": "#007f86", "letterSpacing": "2px", "fontSize": "12px"}),
    html.H1("Falcon 9 launch explorer", style={"fontSize": "38px", "marginBottom": "8px"}),
    html.P("Explore launch-site and payload patterns in the IBM capstone’s 2010–2020 snapshot.", style={"color": "#59677a"}),
    html.Div([card("Launches in snapshot", str(len(spacex_df))), card("Successful landings", f"{spacex_df.Class.mean():.1%}"),
              card("Observed period", "2010–2020")], style={"display": "flex", "flexWrap": "wrap", "gap": "14px", "margin": "28px 0"}),
    html.Label("Launch site", htmlFor="site-dropdown"),
    dcc.Dropdown(id="site-dropdown", options=[{"label": "All launch sites", "value": "ALL"}] +
        [{"label": site, "value": site} for site in sorted(spacex_df.LaunchSite.unique())], value="ALL", clearable=False),
    dcc.Graph(id="success-site-chart"),
    html.Label("Payload mass range (kg)", htmlFor="payload-slider"),
    dcc.RangeSlider(id="payload-slider", min=0, max=MAX_PAYLOAD, step=250, value=[0,MAX_PAYLOAD],
        marks={value:f"{value:,}" for value in range(0,MAX_PAYLOAD+1,4000)}, tooltip={"placement":"bottom","always_visible":False}),
    dcc.Graph(id="success-payload-scatter-chart"),
    html.P("Five unknown payload masses are omitted from the scatter plot. The charts describe historical records; model evaluation is reported separately in results/metrics.json.",
        style={"fontSize": "13px", "color": "#59677a"}),
], style={"maxWidth": "1120px", "margin": "40px auto", "padding": "0 24px", "fontFamily": "Arial, sans-serif", "color": "#16283e"})


def filtered_data(site, payload_range=None):
    frame = spacex_df if site == "ALL" else spacex_df.loc[spacex_df.LaunchSite.eq(site)]
    if payload_range is not None: frame = frame.loc[frame.PayloadMass.between(*payload_range)]
    return frame


@app.callback(Output("success-site-chart","figure"),Input("site-dropdown","value"))
def update_site(site):
    frame = filtered_data(site)
    counts = frame.groupby(["LaunchSite","Landing outcome"]).size().reset_index(name="Launches")
    fig = px.bar(counts,x="LaunchSite",y="Launches",color="Landing outcome",color_discrete_map=COLORS,
        title="Landing outcomes by launch site",labels={"LaunchSite":"Launch site"},template="plotly_white")
    fig.update_layout(legend_title_text="",margin=dict(l=30,r=10,t=60,b=35))
    return fig


@app.callback(Output("success-payload-scatter-chart","figure"),Input("site-dropdown","value"),Input("payload-slider","value"))
def update_scatter(site,payload_range):
    frame = filtered_data(site,payload_range)
    fig = px.scatter(frame,x="PayloadMass",y="FlightNumber",color="Landing outcome",color_discrete_map=COLORS,
        hover_data={"Date":True,"Orbit":True,"LaunchSite":True,"FlightNumber":True},
        labels={"PayloadMass":"Payload mass (kg)","FlightNumber":"Launch sequence number"},
        title=f"Payload and launch history · {len(frame)} observations",template="plotly_white")
    fig.update_layout(legend_title_text="",margin=dict(l=30,r=10,t=60,b=35))
    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port",type=int,default=8050)
    args = parser.parse_args()
    app.run(host="127.0.0.1",port=args.port,debug=False)
