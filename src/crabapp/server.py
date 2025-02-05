import io
from multiprocessing.synchronize import Condition
from typing import Any

import dash_ag_grid as dag
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import polars as pl
import setproctitle
from dash import Dash, Input, Output, State, callback, dcc, html

from crabapp._utils import (
    DataSegment,
    DataSegmentDict,
    LayoutOpts,
    PlotlyTemplates,
    ResultRow,
    SelectedData,
    T_PlotlyTemplate,
    UploadedData,
    parse_contents,
)
from crabapp.domino import terminate_when_parent_process_dies

upload_style = {
    # "width": "100%",
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

flex_container_style = {
    "display": "flex",
    "gap": "10px",
    "align-items": "center",
    "margin-top": "10px",
}

container_style = {"padding": "10px"}

# -----------------------------------------------------------------------------
# Column definitions for the segment results grid
# -----------------------------------------------------------------------------
segment_grid_columns: list[dict[str, Any]] = [
    {"field": "source_file", "headerName": "source_file", "checkboxSelection": True},
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

    app = Dash("crabapp", external_stylesheets=[dbc.themes.JOURNAL])

    app.layout = dbc.Container(
        [
            dbc.Row(
                [
                    # Left column: Card with file upload, dropdowns, and buttons
                    dbc.Col(
                        dbc.Card(
                            [
                                # Inputs for setting skip_rows and separator for the data upload
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            [
                                                dbc.Label("Skip Rows"),
                                                dbc.Input(id="skip-rows", type="number", value=0, min=0),
                                            ]
                                        ),
                                        dbc.Col(
                                            [
                                                dbc.Label("Column Separator"),
                                                dcc.Dropdown(
                                                    id="separator",
                                                    options=[
                                                        {"label": "Detect Automatically", "value": "auto"},
                                                        {"label": "Comma (,)", "value": ","},
                                                        {"label": "Semicolon (;)", "value": ";"},
                                                        {"label": "Tab (\\t)", "value": "\t"},
                                                        {"label": "Pipe (|)", "value": "|"},
                                                    ],
                                                    value="auto",
                                                ),
                                            ]
                                        ),
                                    ],
                                ),
                                # File upload section
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
                                                dbc.Label("Current File: -", id="current-file-label"),
                                            ]
                                        ),
                                    ]
                                ),
                                # Dropdowns for selecting x and y columns
                                html.Div(
                                    [
                                        dcc.Dropdown(
                                            id="x-data",
                                            placeholder="Select column for x-axis",
                                            style={"flex": "1"},
                                        ),
                                        dcc.Dropdown(
                                            id="y-data",
                                            multi=True,
                                            placeholder="Select column(s) for y-axis",
                                            style={"flex": "1"},
                                        ),
                                    ],
                                    style=flex_container_style,
                                ),
                                # Dropdown for plot template and control buttons
                                html.Div(
                                    [
                                        dcc.Dropdown(
                                            id="plot-template",
                                            options=list(PlotlyTemplates),
                                            value="simple_white",
                                            style={"flex": "1"},
                                        ),
                                        dbc.Button("Plot", id="plot-button", n_clicks=0),
                                        dbc.Button("Add Segment", id="add-segment-button", n_clicks=0),
                                        dbc.Button("Clear Segments", id="clear-segments-button", n_clicks=0),
                                        dbc.Button("Save Segments", id="save-segments-button", n_clicks=0),
                                    ],
                                    style=flex_container_style,
                                ),
                            ],
                            body=True,
                        ),
                        width=4,
                    ),
                    # Right column: Data grid for segment results
                    dbc.Col(
                        html.Div(
                            dag.AgGrid(
                                id="segment-result-grid",
                                columnSize="responsiveSizeToFit",
                                columnDefs=segment_grid_columns,
                                rowData=[],
                                csvExportParams={"fileName": "results.csv"},
                                dashGridOptions={
                                    "rowSelection": "multiple",
                                    "suppressRowClickSelection": True,
                                    "animateRows": False,
                                },
                            ),
                        ),
                        width=8,
                    ),
                ]
            ),
            # Row for the graph output
            dbc.Row(
                dbc.Col(dcc.Graph(id="output-graph"), width=12),
                style={"margin-top": "10px"},
            ),
            # Row for data upload output (if any)
            dbc.Row(
                dbc.Col(id="output-data-upload", width=12),
                style={"margin-top": "10px"},
            ),
            # Hidden stores for intermediate data
            dcc.Store(id="uploaded-data"),
            dcc.Store(id="data-segments"),
        ],
        fluid=True,
        style=container_style,
    )

    @callback(
        Output("output-data-upload", "children"),
        Output("uploaded-data", "data"),
        Output("x-data", "options"),
        Output("y-data", "options"),
        Output("current-file-label", "children"),
        Input("upload-data", "contents"),
        State("upload-data", "filename"),
        State("skip-rows", "value"),
        State("separator", "value"),
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
        Input("plot-button", "n_clicks"),
        State("plot-template", "value"),
        State("uploaded-data", "data"),
        State("x-data", "value"),
        State("y-data", "value"),
        prevent_initial_call=True,
    )
    def update_graph(
        n_clicks: int, template: T_PlotlyTemplate, data: UploadedData, x_col: str, y_cols: list[str]
    ) -> go.Figure:
        if not n_clicks or not data:
            return go.Figure()
        df = pl.read_json(io.StringIO(data["data"]))
        lopts = LayoutOpts(theme=template)
        DataSegment.set_source(data["name"], df, x_col, y_cols[0], y_cols[1] if len(y_cols) > 1 else None, lopts)
        return DataSegment.source_fig

    @callback(
        Output("data-segments", "data", allow_duplicate=True),
        Output("output-graph", "figure", allow_duplicate=True),
        Output("segment-result-grid", "rowData"),
        Input("add-segment-button", "n_clicks"),
        State("output-graph", "selectedData"),
        prevent_initial_call=True,
    )
    def update_segments(
        n_clicks: int, selected_data: SelectedData | None
    ) -> tuple[list[DataSegmentDict], go.Figure, list[dict[str, Any]]]:
        if not n_clicks or not selected_data:
            return [], DataSegment.source_fig, []
        start = selected_data["points"][0]["pointIndex"]
        end = selected_data["points"][-1]["pointIndex"]
        DataSegment(start, end)

        res_dfs = pl.concat(
            [
                pl.read_json(io.StringIO(s["formatted_results"])).select(
                    pl.lit(DataSegment.source_name).alias("source_file"),
                    pl.lit(s["start_index"]).alias("start_index"),
                    pl.lit(s["end_index"]).alias("end_index"),
                    pl.col("slope").alias("slope"),
                    pl.col("rsquared").alias("rsquared"),
                )
                for s in DataSegment.all_segments
            ]
        )
        return DataSegment.all_segments, DataSegment.source_fig, res_dfs.to_dicts()

    @callback(
        Output("output-graph", "figure", allow_duplicate=True),
        Output("data-segments", "data", allow_duplicate=True),
        Output("segment-result-grid", "deleteSelectedRows"),
        Input("clear-segments-button", "n_clicks"),
        State("segment-result-grid", "selectedRows"),
        prevent_initial_call=True,
    )
    def clear_segments(n_clicks: int, selected_rows: list[ResultRow]) -> tuple[go.Figure, list[DataSegmentDict], bool]:
        current_fits = DataSegment.all_segments.copy()
        # get the start indices from the selected rows and use them to remove the corresponding segments
        for row in selected_rows:
            start = row["start_index"]
            for i, fit in enumerate(current_fits):
                if fit["start_index"] == start:
                    current_fits.pop(i)
                    break

        DataSegment.make_base_fig()
        for cfit in current_fits:
            DataSegment(cfit["start_index"], cfit["end_index"])

        return DataSegment.source_fig, DataSegment.all_segments, True

    @callback(
        Output("segment-result-grid", "exportDataAsCsv"),
        Input("save-segments-button", "n_clicks"),
        prevent_initial_call=True,
    )
    def save_segments_to_csv(n_clicks: int) -> bool:
        return bool(n_clicks)

    with server_is_started:
        server_is_started.notify()

    # debug cannot be True right now with nuitka: https://github.com/Nuitka/Nuitka/issues/2953
    app.run(debug=False, host=host, port=port)
