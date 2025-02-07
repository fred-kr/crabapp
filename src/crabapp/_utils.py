import base64
import csv
import io
import multiprocessing.process
import os
import signal
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Literal, NamedTuple, TypedDict

import dash_ag_grid as dag
import plotly.graph_objects as go
import polars as pl
from dash import html
from scipy import stats

type T_PlotlyTemplate = Literal[
    "ggplot2",
    "seaborn",
    "simple_white",
    "plotly",
    "plotly_white",
    "plotly_dark",
    "presentation",
    "xgridoff",
    "ygridoff",
    "gridon",
    "none",
]

PlotlyTemplates = (
    "ggplot2",
    "seaborn",
    "simple_white",
    "plotly",
    "plotly_white",
    "plotly_dark",
    "presentation",
    "xgridoff",
    "ygridoff",
    "gridon",
    "none",
)


class QualitativeColors(NamedTuple):
    """NamedTuple holding the available qualitative color palettes."""

    Plotly = (
        "#636EFA",
        "#EF553B",
        "#00CC96",
        "#AB63FA",
        "#FFA15A",
        "#19D3F3",
        "#FF6692",
        "#B6E880",
        "#FF97FF",
        "#FECB52",
    )
    D3 = "#1F77B4", "#FF7F0E", "#2CA02C", "#D62728", "#9467BD", "#8C564B", "#E377C2", "#7F7F7F", "#BCBD22", "#17BECF"
    G10 = "#3366CC", "#DC3912", "#FF9900", "#109618", "#990099", "#0099C6", "#DD4477", "#66AA00", "#B82E2E", "#316395"
    T10 = "#4C78A8", "#F58518", "#E45756", "#72B7B2", "#54A24B", "#EECA3B", "#B279A2", "#FF9DA6", "#9D755D", "#BAB0AC"
    Alphabet = (
        "#AA0DFE",
        "#3283FE",
        "#85660D",
        "#782AB6",
        "#565656",
        "#1C8356",
        "#16FF32",
        "#F7E1A0",
        "#E2E2E2",
        "#1CBE4F",
        "#C4451C",
        "#DEA0FD",
        "#FE00FA",
        "#325A9B",
        "#FEAF16",
        "#F8A19F",
        "#90AD1C",
        "#F6222E",
        "#1CFFCE",
        "#2ED9FF",
        "#B10DA1",
        "#C075A6",
        "#FC1CBF",
        "#B00068",
        "#FBE426",
        "#FA0087",
    )
    Dark24 = (
        "#2E91E5",
        "#E15F99",
        "#1CA71C",
        "#FB0D0D",
        "#DA16FF",
        "#222A2A",
        "#B68100",
        "#750D86",
        "#EB663B",
        "#511CFB",
        "#00A08B",
        "#FB00D1",
        "#FC0080",
        "#B2828D",
        "#6C7C32",
        "#778AAE",
        "#862A16",
        "#A777F1",
        "#620042",
        "#1616A7",
        "#DA60CA",
        "#6C4516",
        "#0D2A63",
        "#AF0038",
    )
    Light24 = (
        "#FD3216",
        "#00FE35",
        "#6A76FC",
        "#FED4C4",
        "#FE00CE",
        "#0DF9FF",
        "#F6F926",
        "#FF9616",
        "#479B55",
        "#EEA6FB",
        "#DC587D",
        "#D626FF",
        "#6E899C",
        "#00B5F7",
        "#B68E00",
        "#C9FBE5",
        "#FF0092",
        "#22FFA7",
        "#E3EE9E",
        "#86CE00",
        "#BC7196",
        "#7E7DCD",
        "#FC6955",
        "#E48F72",
    )


class LinregressResultDict(TypedDict):
    slope: float
    intercept: float
    rvalue: float
    pvalue: float
    stderr: float
    intercept_stderr: float
    rsquared: float


class LinregressResult(NamedTuple):
    slope: float
    intercept: float
    rvalue: float
    pvalue: float
    stderr: float
    intercept_stderr: float

    def to_dict(self) -> LinregressResultDict:
        return {
            "slope": self.slope,
            "intercept": self.intercept,
            "rvalue": self.rvalue,
            "pvalue": self.pvalue,
            "stderr": self.stderr,
            "intercept_stderr": self.intercept_stderr,
            "rsquared": self.rvalue**2,
        }

    def to_df(self) -> pl.DataFrame:
        return pl.DataFrame([self.to_dict()])


class SelectedPoint(TypedDict):
    curveNumber: int
    pointNumber: int
    pointIndex: int
    x: float
    y: float
    customdata: list[Any]


class SelectedRange(TypedDict):
    x: list[float]
    y: list[float]


class SelectedData(TypedDict):
    points: list[SelectedPoint]
    range: SelectedRange


class UploadedData(TypedDict):
    name: str
    data: str


class ResultRow(TypedDict):
    source_file: str
    start_index: int
    end_index: int
    slope: float
    rsquared: float


@dataclass(slots=True)
class LayoutOpts:
    """
    Dataclass to hold layout options for the Plotly plot.

    Parameters
    ----------
    title : str
        The title of the plot.
    x_label : str
        The label for the x-axis.
    y0_label : str
        The label for the y-axis.
    y1_label : str
        The label for the secondary y-axis (if one exists).
    theme : PlotlyTemplate, optional
        The theme to use for the plot. Defaults to "simple_white".
    colors : Sequence[str], optional
        The colors to use for the fits. Defaults to QualitativeColors.Plotly.
    width : int, optional
        The width of the plot. Defaults to 2100.
    height : int, optional
        The height of the plot. Defaults to 1000.
    font_size : int, optional
        The font size of the annotations in pixels. Defaults to 12.
    secondary_y_type : Literal["color", "scatter", "line"], optional
        The type of the secondary y-axis. Defaults to "line".
    """

    title: str = ""
    x_label: str = "x"
    y0_label: str = "y0"
    y1_label: str | None = "y1"
    theme: T_PlotlyTemplate = "simple_white"
    # colors: Sequence[str] = QualitativeColors.Plotly
    width: int = 2100
    height: int = 1000
    font_size: int = 12
    secondary_y_type: Literal["color", "scatter", "line"] = "line"

    # def set_colors(self, name: Literal["Plotly", "D3", "G10", "T10", "Alphabet", "Dark24", "Light24"]) -> None:
    # self.colors = getattr(QualitativeColors, name)

    def apply_to_fig(self, fig: go.Figure) -> go.Figure:
        return fig.update_layout(
            title=self.title,
            xaxis_title=self.x_label,
            yaxis_title=self.y0_label,
            yaxis2_title=self.y1_label,
            width=self.width,
            height=self.height,
            font_size=self.font_size,
        )


class DataSegmentDict(TypedDict):
    segment_id: str
    start_index: int
    end_index: int
    data: str  # JSON string of df, read with pl.read_json(io.StringIO(data))
    fit_result: LinregressResultDict
    fig: go.Figure
    x_col: str
    y0_col: str
    y1_col: str | None
    name: str
    formatted_results: str


class DataSegment:
    all_segments: ClassVar[list[DataSegmentDict]] = []
    source_name: ClassVar[str] = ""
    source_data: ClassVar[pl.DataFrame] = pl.DataFrame()
    source_fig: ClassVar[go.Figure] = go.Figure()
    x_col: ClassVar[str] = ""
    y0_col: ClassVar[str] = ""
    y1_col: ClassVar[str | None] = None
    _source_set: ClassVar[bool] = False
    _layout_opts: ClassVar[LayoutOpts] = LayoutOpts()

    @classmethod
    def set_source(
        cls,
        source_name: str,
        source_data: pl.DataFrame,
        x_col: str,
        y0_col: str,
        y1_col: str | None = None,
        layout_opts: LayoutOpts | None = None,
    ) -> None:
        cls._layout_opts = layout_opts or cls._layout_opts
        cls.source_name = Path(source_name).stem
        cls.source_data = source_data
        cls.x_col = x_col
        cls.y0_col = y0_col
        cls.y1_col = y1_col
        cls.make_base_fig()
        cls._source_set = True

    @classmethod
    def make_base_fig(cls, theme: T_PlotlyTemplate | None = None) -> None:
        cls.all_segments = []
        point_color = "lightgray" if cls.y1_col is None else cls.source_data.get_column(cls.y1_col)
        fig = go.Figure()
        fig.add_scattergl(
            x=cls.source_data.get_column(cls.x_col),
            y=cls.source_data.get_column(cls.y0_col),
            mode="markers",
            marker=dict(color=point_color, symbol="circle-open-dot", colorscale="Plasma", opacity=0.2, size=3),
        )
        cls.source_fig = fig.update_layout(
            clickmode="event+select",
            template=theme or cls._layout_opts.theme,
            height=cls._layout_opts.height,
            dragmode="select",
            showlegend=False,
        )

    def __init__(self, start_index: int, end_index: int) -> None:
        if not self._source_set:
            raise ValueError("DataSegment must be initialized after calling DataSegment.set_source")
        self.start_index = start_index
        self.end_index = end_index
        self.data = self.source_data.slice(self.start_index, self.end_index - self.start_index + 1)
        res: Any = stats.linregress(self.data.get_column(self.x_col), self.data.get_column(self.y0_col))
        self.fit_result = LinregressResult(
            slope=res.slope,
            intercept=res.intercept,
            rvalue=res.rvalue,
            pvalue=res.pvalue,
            stderr=res.stderr,
            intercept_stderr=res.intercept_stderr,
        )
        self.data = self.data.with_columns(
            pl.lit(self.segment_id).alias("segment_id"),
            (self.fit_result.slope * pl.col(self.x_col) + self.fit_result.intercept).alias("fitted"),
        )
        DataSegment.all_segments.append(self.serialize())
        DataSegment.all_segments.sort(key=lambda s: s["start_index"])

    @property
    def segment_id(self) -> str:
        return f"{self.start_index}-{self.end_index}"

    @property
    def name(self) -> str:
        return f"{self.source_name}_{self.segment_id}"

    @property
    def x_data(self) -> pl.Series:
        return self.data.get_column(self.x_col)

    @property
    def y0_data(self) -> pl.Series:
        return self.data.get_column(self.y0_col)

    @property
    def y1_data(self) -> pl.Series | None:
        return None if self.y1_col is None else self.data.get_column(self.y1_col)

    @property
    def y0_fitted(self) -> pl.Series:
        return self.data.get_column("fitted")

    @property
    def slope(self) -> float:
        return self.fit_result.slope

    @property
    def intercept(self) -> float:
        return self.fit_result.intercept

    @property
    def r_value(self) -> float:
        return self.fit_result.rvalue

    @property
    def p_value(self) -> float:
        return self.fit_result.pvalue

    @property
    def stderr(self) -> float:
        return self.fit_result.stderr

    @property
    def intercept_stderr(self) -> float:
        return self.fit_result.intercept_stderr

    @property
    def r_squared(self) -> float:
        return self.r_value**2

    def plot(self, add: bool = True) -> go.Figure:
        if add:
            self.source_fig.add_scattergl(
                x=self.x_data,
                y=self.y0_fitted,
                mode="lines",
                line=dict(color="red", width=3),
                name=f"Segment {self.segment_id}",
                hoverinfo="name",
            )
            return self.source_fig
        else:
            fig = go.Figure()
            point_color = self.y1_data or "lightgray"
            fig.add_scattergl(
                x=self.x_data,
                y=self.y0_data,
                mode="markers",
                marker=dict(color=point_color),
                name=f"Segment {self.segment_id}, raw values",
                hoverinfo="name",
            )
            fig.add_scattergl(
                x=self.x_data,
                y=self.y0_fitted,
                mode="lines",
                line=dict(color="red", width=3),
                name=f"Segment {self.segment_id}, fitted '{self.y0_col}' values",
                hoverinfo="name",
            )
            return fig

    def serialize(self) -> DataSegmentDict:
        return {
            "segment_id": self.segment_id,
            "start_index": self.start_index,
            "end_index": self.end_index,
            "data": self.data.write_json(),
            "fit_result": self.fit_result.to_dict(),
            "fig": self.plot(),
            "x_col": self.x_col,
            "y0_col": self.y0_col,
            "y1_col": self.y1_col,
            "name": self.name,
            "formatted_results": self.fit_result.to_df().write_json(),
        }


def detect_delimiter(decoded_string: str, skip_rows: int = 0, sample_size: int = 3) -> str:
    """
    Automatically detects the delimiter used in a text file containing tabular data.

    Args:
        decoded_string (str): The content of the file as a decoded string.
        skip_rows (int): The number of rows to skip before attempting to detect the delimiter.
        sample_size (int): The number of rows to sample from the file (starting from the `skip_rows` row).

    Returns:
        str: The detected delimiter character.

    Raises:
        ValueError: If the delimiter cannot be detected.
        FileNotFoundError: If the file does not exist.
    """
    sample_size = max(1, sample_size)

    try:
        with io.StringIO(decoded_string) as file:
            if file.readline() == "":
                raise ValueError("File is empty")

            for _ in range(skip_rows):
                file.readline()

            sample = "\n".join(file.readline() for _ in range(sample_size - 1))
    except Exception as e:
        raise ValueError(f"Error reading file: {str(e)}") from e

    sniffer = csv.Sniffer()

    try:
        dialect = sniffer.sniff(sample)
        return dialect.delimiter
    except csv.Error as e:
        raise ValueError(f"Delimiter detection failed. {str(e)}") from e


def parse_contents(
    contents: str, filename: str, skip_rows: int = 0, separator: str = "auto"
) -> tuple[html.Div, pl.DataFrame]:
    _, content_string = contents.split(",")
    decoded = base64.b64decode(content_string)
    suffix = Path(filename).suffix
    try:
        if any(ext in suffix for ext in {"csv", "txt", "tsv"}):
            content = decoded.decode("utf-8", errors="replace")
            if separator == "auto":
                separator = detect_delimiter(content, skip_rows=skip_rows)
            cleaned_content = "\n".join(
                separator.join(field.strip() for field in line.split(separator)) for line in content.splitlines()
            )
            df = pl.read_csv(io.StringIO(cleaned_content), skip_rows=skip_rows, separator=separator)
        elif "xls" in suffix:
            df = pl.read_excel(io.BytesIO(decoded), read_options={"skip_rows": skip_rows})
        else:
            return html.Div(["There was an error processing this file."]), pl.DataFrame()
    except Exception as e:
        print(e)
        return html.Div(["There was an error processing this file."]), pl.DataFrame()

    return html.Div(
        [
            html.H5(filename),
            dag.AgGrid(
                columnSize="responsiveSizeToFit",
                columnDefs=[{"field": col_name, "headerName": col_name} for col_name in df.columns],
                rowData=df.to_dicts(),
                dashGridOptions={"suppressFieldDotNotation": True},
            ),
        ]
    ), df.with_row_index()


def join_process_and_terminate(process: multiprocessing.process.BaseProcess) -> None:
    """
    Whenever the given process exits, send SIGTERM to self.
    This function is synchronous; for async usage see the other two.
    """
    process.join()
    # sys.exit() raises, killing only the current thread
    # os._exit() is private, and also doesn't allow the thread to gracefully exit
    os.kill(os.getpid(), signal.SIGTERM)


def terminate_when_process_dies(process: multiprocessing.process.BaseProcess) -> None:
    """
    Whenever the given process exits, send SIGTERM to self.
    This function is asynchronous.
    """
    threading.Thread(target=join_process_and_terminate, args=(process,)).start()


def terminate_when_parent_process_dies() -> None:
    """
    Whenever the parent process exits, send SIGTERM to self.
    This function is asynchronous.
    """
    parent_process = multiprocessing.parent_process()
    if parent_process is not None:
        terminate_when_process_dies(parent_process)
