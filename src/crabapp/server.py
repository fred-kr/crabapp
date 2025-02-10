import io
from multiprocessing.synchronize import Condition
from typing import Any, Literal

import dash_ag_grid as dag
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import polars as pl
import setproctitle
from dash import Dash, Input, Output, State, callback, dcc, html

from crabapp._utils import (
    DataSegment,
    PlotlyTemplates,
    ResultRow,
    SelectedData,
    T_PlotlyTemplate,
    UploadedData,
    parse_contents,
    terminate_when_parent_process_dies,
)

upload_style = {
    "height": "60px",
    "lineHeight": "60px",
    "borderWidth": "1px",
    "borderStyle": "dashed",
    "borderRadius": "5px",
    "textAlign": "center",
    "margin-top": "10px",
}

upload_link_style = {
    "color": "blue",
    "textDecoration": "underline",
    "cursor": "pointer",
}

container_style = {"padding": "10px"}

# -----------------------------------------------------------------------------
# Column definitions for the segment results grid
# -----------------------------------------------------------------------------
segment_grid_columns: list[dict[str, Any]] = [
    {"field": "source_file", "headerName": "source_file", "checkboxSelection": True},
    {"field": "fit_id", "headerName": "fit_id"},
    {"field": "start_index", "headerName": "start_index"},
    {"field": "end_index", "headerName": "end_index"},
    {"field": "slope", "headerName": "slope"},
    {"field": "rsquared", "headerName": "rsquared"},
]


def start_dash(host: str, port: str, server_is_started: Condition) -> None:
    # Set the process title.
    setproctitle.setproctitle("crabapp-dash")
    # When the parent dies, follow along.
    terminate_when_parent_process_dies()

    app = Dash("crabapp", external_stylesheets=[dbc.themes.BOOTSTRAP])

    app.layout = dbc.Container(
        [
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.Row(
                                        [
                                            dbc.Col(
                                                [
                                                    dbc.Label("Skip Rows"),
                                                    dbc.Input(id="input-skip-rows", type="number", value=0, min=0),
                                                ]
                                            ),
                                            dbc.Col(
                                                [
                                                    dbc.Label("Column Separator"),
                                                    dcc.Dropdown(
                                                        id="dropdown-separator",
                                                        options=[
                                                            {"label": "Detect Automatically", "value": "auto"},
                                                            {"label": "Comma (',')", "value": ","},
                                                            {"label": "Semicolon (';')", "value": ";"},
                                                            {"label": "Tab ('\\t')", "value": "\t"},
                                                            {"label": "Pipe ('|')", "value": "|"},
                                                        ],
                                                        value="auto",
                                                    ),
                                                ]
                                            ),
                                        ],
                                    ),
                                    dbc.Row(
                                        [
                                            dbc.Col(
                                                [
                                                    dcc.Upload(
                                                        id="upload-data",
                                                        children=html.Div(
                                                            [
                                                                "Drag and Drop or ",
                                                                html.A("Select File", style=upload_link_style),
                                                            ]
                                                        ),
                                                        multiple=False,
                                                        style=upload_style,
                                                    ),
                                                    dbc.Label("Current File: -", id="label-current-file"),
                                                ]
                                            ),
                                        ]
                                    ),
                                    dbc.Row(
                                        [
                                            dbc.Col(
                                                [
                                                    dbc.Label("X Axis"),
                                                    dcc.Dropdown(
                                                        id="dropdown-x-data",
                                                        placeholder="Select column for x-axis",
                                                    ),
                                                ]
                                            ),
                                            dbc.Col(
                                                [
                                                    dbc.Label("Y Axis"),
                                                    dcc.Dropdown(
                                                        id="dropdown-y-data",
                                                        multi=True,
                                                        placeholder="Select column(s) for y-axis",
                                                    ),
                                                ]
                                            ),
                                        ]
                                    ),
                                    dbc.Row(
                                        [
                                            dbc.Col(
                                                [
                                                    dbc.Label("Plot Style"),
                                                    dcc.Dropdown(
                                                        id="dropdown-plot-template",
                                                        options=list(PlotlyTemplates),
                                                        value="simple_white",
                                                    ),
                                                ],
                                            ),
                                            dbc.Col(
                                                [
                                                    dbc.Label("Secondary Y Style"),
                                                    dcc.Dropdown(
                                                        id="dropdown-secondary-y-style",
                                                        options=[
                                                            {"label": "Color", "value": "color"},
                                                            {"label": "Scatter", "value": "scatter"},
                                                            {"label": "Line", "value": "line"},
                                                        ],
                                                        value="color",
                                                        clearable=False,
                                                    ),
                                                ],
                                            ),
                                            dbc.Col(
                                                dbc.Button("Plot", id="button-make-plot", n_clicks=0),
                                                align="end",
                                            ),
                                            dbc.Col(
                                                dbc.Button(
                                                    "Add linear fit",
                                                    id="button-add-fit",
                                                    n_clicks=0,
                                                ),
                                                align="end",
                                            ),
                                        ],
                                        style={"margin-top": "10px"},
                                    ),
                                ],
                                body=True,
                            ),
                        ]
                    ),
                    dbc.Col(
                        [
                            dbc.ButtonGroup(
                                [
                                    dbc.Button(
                                        "Remove selected result(s)",
                                        id="button-clear-segments",
                                        n_clicks=0,
                                    ),
                                    dbc.Button(
                                        "Export results",
                                        id="button-save-segments",
                                        n_clicks=0,
                                    ),
                                ],
                            ),
                            html.Div(
                                dag.AgGrid(
                                    id="table-segment-results",
                                    columnSize="responsiveSizeToFit",
                                    columnDefs=segment_grid_columns,
                                    rowData=[],
                                    csvExportParams={"fileName": "results.csv"},
                                    dashGridOptions={
                                        "rowSelection": "multiple",
                                        "suppressRowClickSelection": True,
                                        "animateRows": False,
                                        "suppressFieldDotNotation": True,
                                    },
                                ),
                            ),
                        ],
                    ),
                ]
            ),
            dbc.Row(
                dbc.Col(dcc.Graph(id="output-graph"), width=12),
                style={"margin-top": "10px"},
            ),
            dbc.Row(
                dbc.Col(id="output-data-upload", width=12),
                style={"margin-top": "10px"},
            ),
            dcc.Store(id="store-data-upload"),
        ],
        fluid=True,
        style=container_style,
    )

    @callback(
        Output("output-data-upload", "children"),
        Output("store-data-upload", "data"),
        Output("dropdown-x-data", "options"),
        Output("dropdown-y-data", "options"),
        Output("label-current-file", "children"),
        Input("upload-data", "contents"),
        State("upload-data", "filename"),
        State("input-skip-rows", "value"),
        State("dropdown-separator", "value"),
    )
    def update_output(
        content: str | None, name: str, skip_rows: int, separator: str
    ) -> tuple[list[html.Div], UploadedData, list[str], list[str], str]:
        if content is not None:
            div, df = parse_contents(content, name, skip_rows, separator)
            return [div], {"name": name, "data": df.write_json()}, df.columns, df.columns, f"Current File: {name}"
        return [html.Div(["No file uploaded."])], {"name": "", "data": ""}, [], [], "Current File: -"

    @callback(
        Output("output-graph", "figure", allow_duplicate=True),
        Input("button-make-plot", "n_clicks"),
        State("dropdown-plot-template", "value"),
        State("store-data-upload", "data"),
        State("dropdown-x-data", "value"),
        State("dropdown-y-data", "value"),
        State("dropdown-secondary-y-style", "value"),
        prevent_initial_call=True,
    )
    def update_graph(
        n_clicks: int, theme: T_PlotlyTemplate, data: UploadedData, x_col: str, y_cols: list[str], secondary_y_style: Literal["color", "scatter", "line"]
    ) -> go.Figure:
        if not n_clicks or not data:
            return go.Figure()
        df = pl.read_json(io.StringIO(data["data"]))
        DataSegment.set_source(data["name"], df, x_col, y_cols[0], y_cols[1] if len(y_cols) > 1 else None, theme=theme)
        return DataSegment.source_fig

    @callback(
        Output("output-graph", "figure", allow_duplicate=True),
        Output("table-segment-results", "rowData"),
        Input("button-add-fit", "n_clicks"),
        State("output-graph", "selectedData"),
        prevent_initial_call=True,
    )
    def update_segments(n_clicks: int, selected_data: SelectedData | None) -> tuple[go.Figure, list[dict[str, Any]]]:
        if not n_clicks or not selected_data:
            return DataSegment.source_fig, []
        start = selected_data["points"][0]["pointIndex"]
        end = selected_data["points"][-1]["pointIndex"]
        ds = DataSegment(start, end)
        ds.plot()

        result_df = pl.concat([pl.DataFrame(s.result_row()) for s in DataSegment.all_segments])
        return DataSegment.source_fig, result_df.to_dicts()

    @callback(
        Output("output-graph", "figure", allow_duplicate=True),
        Output("table-segment-results", "deleteSelectedRows"),
        Input("button-clear-segments", "n_clicks"),
        State("table-segment-results", "selectedRows"),
        State("dropdown-plot-template", "value"),
        prevent_initial_call=True,
    )
    def clear_segments(
        n_clicks: int, selected_rows: list[ResultRow], theme: T_PlotlyTemplate
    ) -> tuple[go.Figure, bool]:
        current_fits = DataSegment.all_segments.copy()
        # get the start indices from the selected rows and use them to remove the corresponding segments
        for row in selected_rows:
            start = row["start_index"]
            for i, fit in enumerate(current_fits):
                if fit.start_index == start:
                    current_fits.pop(i)
                    break

        DataSegment.make_base_fig(theme=theme)
        for cfit in current_fits:
            ds = DataSegment(cfit.start_index, cfit.end_index)
            ds.plot()

        return DataSegment.source_fig, True

    @callback(
        Output("table-segment-results", "exportDataAsCsv"),
        Input("button-save-segments", "n_clicks"),
        prevent_initial_call=True,
    )
    def save_segments_to_csv(n_clicks: int) -> bool:
        return bool(n_clicks)

    with server_is_started:
        server_is_started.notify()

    # debug cannot be True right now with nuitka: https://github.com/Nuitka/Nuitka/issues/2953
    app.run(debug=False, host=host, port=port)
