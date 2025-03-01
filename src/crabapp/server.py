import io
from pathlib import Path
from typing import TYPE_CHECKING, Any

import dash_ag_grid as dag
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import polars as pl
import polars.selectors as cs
import setproctitle
from dash import Dash, Input, Output, State, callback, ctx, dcc, html

from crabapp.utils import (
    DataSegment,
    PlotlyTemplates,
    ResultRow,
    SelectedData,
    T_PlotlyTemplate,
    UploadedData,
    parse_contents,
    terminate_when_parent_process_dies,
)

if TYPE_CHECKING:
    from multiprocessing.synchronize import Condition

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
segment_grid_columns = [
    {"field": "source_file", "headerName": "source_file", "checkboxSelection": True, "cellDataType": "text"},
    {"field": "fit_id", "headerName": "fit_id", "cellDataType": "number"},
    {"field": "start_index", "headerName": "start_index", "cellDataType": "number"},
    {"field": "end_index", "headerName": "end_index", "cellDataType": "number"},
    {"field": "slope", "headerName": "slope", "cellDataType": "number"},
    {"field": "rsquared", "headerName": "rsquared", "cellDataType": "number"},
    {"field": "mean_y2", "headerName": "mean_y2", "cellDataType": "number"},
    {"field": "name_x", "headerName": "name_x", "cellDataType": "text"},
    {"field": "start_x", "headerName": "start_x", "cellDataType": "number"},
    {"field": "end_x", "headerName": "end_x", "cellDataType": "number"},
    {"field": "name_y", "headerName": "name_y", "cellDataType": "text"},
    {"field": "start_y", "headerName": "start_y", "cellDataType": "number"},
    {"field": "end_y", "headerName": "end_y", "cellDataType": "number"},
    {"field": "name_y2", "headerName": "name_y2", "cellDataType": "text"},
    {"field": "start_y2", "headerName": "start_y2", "cellDataType": "number"},
    {"field": "end_y2", "headerName": "end_y2", "cellDataType": "number"},
]

def create_data_input_controls():
    return dbc.Card(
        [
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Label("Skip Rows"),
                            dbc.Input(id="input-skip-rows", type="number", value=57, min=0),
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
                ]
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
                    dbc.Col(
                        dbc.Button(
                            "Clear Data",
                            id="button-clear-data",
                            n_clicks=0,
                            style={"margin-top": "10px"},
                        ),
                        width=3,
                        align="start",
                    ),
                ],
                style={"margin-top": "10px"},
            ),
        ],
        body=True,
    )


def create_axis_controls():
    return dbc.Row(
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
                        placeholder="Select column for y-axis",
                    ),
                ]
            ),
            dbc.Col(
                [
                    dbc.Label("Secondary Y Axis"),
                    dcc.Dropdown(
                        id="dropdown-y2-data",
                        placeholder="Select column for secondary y-axis",
                    ),
                ]
            ),
        ]
    )


def create_plot_controls():
    return dbc.Row(
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
                width=3,
            ),
            dbc.Col(
                dbc.Button("Plot", id="button-make-plot", n_clicks=0),
                align="end",
                width=1,
            ),
            dbc.Col(
                dbc.Button("Add fit", id="button-add-fit", n_clicks=0),
                align="end",
                width=2,
            ),
            dbc.Col(
                dbc.Button(
                    "Remove selected result(s)",
                    id="button-clear-segments",
                    n_clicks=0,
                ),
                align="end",
                width=4,
            ),
            dbc.Col(
                dbc.Button(
                    "Export results",
                    id="button-save-segments",
                    n_clicks=0,
                ),
                align="end",
                width=2,
            ),
        ],
        style={"margin-top": "10px"},
    )


def create_results_table():
    return html.Div(
        dag.AgGrid(
            id="table-segment-results",
            columnSize="autoSize",
            columnDefs=segment_grid_columns,
            rowData=[],
            csvExportParams={"fileName": "results.csv"},
            dashGridOptions={
                "rowSelection": "multiple",
                "animateRows": False,
                "suppressFieldDotNotation": True,
                "suppressHorizontalScroll": False,
            },
        ),
    )


def create_layout():
    return dbc.Container(
        [
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    create_data_input_controls(),
                                    create_axis_controls(),
                                    create_plot_controls(),
                                ],
                                body=True,
                            ),
                        ]
                    ),
                    dbc.Col([create_results_table()]),
                ]
            ),
            dbc.Row(
                dbc.Col(dcc.Graph(id="output-graph", style={"width": "100%"})),
                style={"margin-top": "10px"},
            ),
            dbc.Row(
                dbc.Col(id="output-data-upload"),
                style={"margin-top": "10px"},
            ),
            dcc.Store(id="store-data-upload"),
        ],
        fluid=True,
        style=container_style,
    )

def start_dash(host: str, port: str, server_is_started: "Condition") -> None:
    # Set the process title.
    setproctitle.setproctitle("crabapp-dash")
    # When the parent dies, follow along.
    terminate_when_parent_process_dies()

    app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

    app.layout = create_layout()

    @callback(
        Output("output-data-upload", "children"),
        Output("store-data-upload", "data"),
        Output("dropdown-x-data", "options"),
        Output("dropdown-y-data", "options"),
        Output("dropdown-y2-data", "options"),
        Output("label-current-file", "children"),
        Output("dropdown-x-data", "value"),
        Output("dropdown-y-data", "value"),
        Output("dropdown-y2-data", "value"),
        Input("upload-data", "contents"),
        Input("button-clear-data", "n_clicks"),
        State("upload-data", "filename"),
        State("input-skip-rows", "value"),
        State("dropdown-separator", "value"),
    )
    def update_output(
        content: str | None, n_clicks: int, name: str, skip_rows: int, separator: str
    ) -> tuple[list[html.Div], UploadedData, list[str], list[str], list[str], str, str, str, str]:
        if ctx.triggered_id == "button-clear-data":
            return [html.Div(["No file uploaded."])], {"name": "", "data": ""}, [], [], [], "Current File: -", "", "", ""
        if content is not None:
            div, df = parse_contents(content, name, skip_rows, separator)
            cols = df.columns
            return [div], {"name": name, "data": df.write_json()}, cols, cols, cols, f"Current File: {name}", cols[1], cols[2], cols[-1]
        return [html.Div(["No file uploaded."])], {"name": "", "data": ""}, [], [], [], "Current File: -", "", "", ""

    @callback(
        Output("table-segment-results", "csvExportParams"),
        Input("store-data-upload", "data"),
    )
    def set_export_name(data: UploadedData) -> dict[str, str]:
        if not data or data["name"] == "":
            return {"fileName": "results.csv"}
        return {"fileName": f"{Path(data['name']).stem}_results.csv"}
    
    @callback(
        Output("upload-data", "filename"),
        Output("upload-data", "contents"),
        Output("upload-data", "last_modified"),
        Output("output-graph", "figure", allow_duplicate=True),
        Output("table-segment-results", "rowData", allow_duplicate=True),
        Input("button-clear-data", "n_clicks"),
        prevent_initial_call=True,
    )
    def clear_current_data(n_clicks: int) -> tuple[str, str, int, go.Figure, list[dict[str, Any]]]:
        DataSegment.clear()
        return "", "", 0, go.Figure(), []

    @callback(
        Output("output-graph", "figure", allow_duplicate=True),
        Input("button-make-plot", "n_clicks"),
        State("dropdown-plot-template", "value"),
        State("store-data-upload", "data"),
        State("dropdown-x-data", "value"),
        State("dropdown-y-data", "value"),
        State("dropdown-y2-data", "value"),
        prevent_initial_call=True,
    )
    def update_graph(
        n_clicks: int,
        theme: T_PlotlyTemplate,
        data: UploadedData,
        x_col: str,
        y_col: str,
        y2_col: str | None,
    ) -> go.Figure:
        if not n_clicks or data["data"] == "":
            return go.Figure()
        df = pl.read_json(io.StringIO(data["data"]))
        y2_col = None if y2_col == "" else y2_col
        DataSegment.set_source(data["name"], df, x_col, y_col, y2_col, theme=theme)
        if len(DataSegment.all_segments) > 0:
            for ds in DataSegment.all_segments:
                ds.plot()
        return DataSegment.source_fig

    @callback(
        Output("output-graph", "figure", allow_duplicate=True),
        Output("table-segment-results", "rowData", allow_duplicate=True),
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
        new_fig = ds.plot()
        result_df = pl.concat(pl.DataFrame(s.result_row()) for s in DataSegment.all_segments)
        return new_fig, result_df.to_dicts()

    @callback(
        Output("output-graph", "figure", allow_duplicate=True),
        Output("table-segment-results", "rowData", allow_duplicate=True),
        Input("button-clear-segments", "n_clicks"),
        State("table-segment-results", "selectedRows"),
        State("dropdown-plot-template", "value"),
        prevent_initial_call=True,
    )
    def clear_segments(
        n_clicks: int, selected_rows: list[ResultRow], theme: T_PlotlyTemplate
    ) -> tuple[go.Figure, list[dict[str, Any]]]:
        # get the start indices from the selected rows and use them to remove the corresponding segments
        for row in selected_rows:
            start = row["start_index"]
            for i, fit in enumerate(DataSegment.all_segments):
                if fit.start_index == start:
                    DataSegment.all_segments.pop(i)
                    break

        DataSegment.make_base_fig(theme=theme)

        if not DataSegment.all_segments:
            return DataSegment.source_fig, []
        DataSegment.all_segments.sort(key=lambda s: s.start_index)
        for ds in DataSegment.all_segments:
            ds.plot()

        result_df = pl.concat(pl.DataFrame(s.result_row()) for s in DataSegment.all_segments)
        return DataSegment.source_fig, result_df.to_dicts()

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
