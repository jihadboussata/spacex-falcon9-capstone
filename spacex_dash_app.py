"""Interactive Plotly Dash application for the IBM SpaceX capstone."""
from pathlib import Path

import dash
from dash import Input, Output, dcc, html
import pandas as pd
import plotly.express as px

DATA_PATH = Path(__file__).parent / "data" / "dataset_part_2.csv"
spacex_df = pd.read_csv(DATA_PATH)

app = dash.Dash(__name__)
app.title = "SpaceX Launch Records Dashboard"

app.layout = html.Div([
    html.H1("SpaceX Launch Records Dashboard", style={"textAlign": "center"}),
    dcc.Dropdown(
        id="site-dropdown",
        options=[{"label": "All Sites", "value": "ALL"}] +
                [{"label": site, "value": site} for site in sorted(spacex_df["LaunchSite"].unique())],
        value="ALL", clearable=False,
    ),
    dcc.Graph(id="success-pie-chart"),
    html.P("Payload mass range (kg)"),
    dcc.RangeSlider(
        id="payload-slider", min=0, max=10000, step=500, value=[0, 10000],
        marks={value: str(value) for value in range(0, 10001, 2500)},
    ),
    dcc.Graph(id="success-payload-scatter-chart"),
], style={"maxWidth": "1100px", "margin": "auto", "fontFamily": "Arial"})


@app.callback(Output("success-pie-chart", "figure"), Input("site-dropdown", "value"))
def update_pie(site):
    filtered = spacex_df if site == "ALL" else spacex_df[spacex_df["LaunchSite"] == site]
    counts = filtered["Class"].value_counts().rename_axis("Class").reset_index(name="Count")
    counts["Outcome"] = counts["Class"].map({0: "Unsuccessful", 1: "Successful"})
    return px.pie(counts, values="Count", names="Outcome", color="Outcome",
                  color_discrete_map={"Successful": "#16a34a", "Unsuccessful": "#dc2626"},
                  title=f"Landing outcomes: {'all sites' if site == 'ALL' else site}")


@app.callback(
    Output("success-payload-scatter-chart", "figure"),
    Input("site-dropdown", "value"), Input("payload-slider", "value"),
)
def update_scatter(site, payload_range):
    filtered = spacex_df[spacex_df["PayloadMass"].between(*payload_range)]
    if site != "ALL":
        filtered = filtered[filtered["LaunchSite"] == site]
    return px.scatter(filtered, x="PayloadMass", y="Class", color="BoosterVersion",
                      hover_data=["FlightNumber", "Orbit", "LaunchSite"],
                      labels={"PayloadMass": "Payload mass (kg)", "Class": "Landing outcome"},
                      title="Payload mass and landing outcome")


if __name__ == "__main__":
    app.run(debug=True)
