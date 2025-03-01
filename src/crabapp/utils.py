import base64
import csv
import io
import multiprocessing.process
import os
import signal
import threading
from pathlib import Path
from typing import Any, ClassVar, Literal, NamedTuple, TypedDict, cast

import dash_ag_grid as dag
import plotly.graph_objects as go
import polars as pl
import polars.selectors as cs
from dash import html
from plotly.subplots import make_subplots
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

    def to_json(self) -> str:
        return self.to_df().write_json()


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
    fit_id: int
    start_index: int
    end_index: int
    slope: float
    rsquared: float
    mean_y2: float
    name_x: str
    start_x: float | int
    end_x: float | int
    name_y: str
    start_y: float
    end_y: float
    name_y2: str
    start_y2: float
    end_y2: float


class DataSegment:
    all_segments: ClassVar[list["DataSegment"]] = []
    source_name: ClassVar[str] = ""
    source_data: ClassVar[pl.DataFrame] = pl.DataFrame()
    source_fig: ClassVar[go.Figure] = go.Figure()
    x_col: ClassVar[str] = ""
    y_col: ClassVar[str] = ""
    y2_col: ClassVar[str | None] = None
    _source_set: ClassVar[bool] = False

    @classmethod
    def clear(cls) -> None:
        cls.all_segments = []
        cls.source_name = ""
        cls.source_data = pl.DataFrame()
        cls.source_fig = go.Figure()
        cls.x_col = ""
        cls.y_col = ""
        cls.y2_col = None
        cls._source_set = False

    @classmethod
    def set_source(
        cls,
        source_name: str,
        source_data: pl.DataFrame,
        x_col: str,
        y_col: str,
        y2_col: str | None = None,
        theme: T_PlotlyTemplate = "simple_white",
    ) -> None:
        cls.source_name = Path(source_name).stem
        cls.source_data = source_data
        cls.x_col = x_col
        cls.y_col = y_col
        cls.y2_col = y2_col
        cls.make_base_fig(theme)
        cls._source_set = True

    @classmethod
    def make_base_fig(cls, theme: T_PlotlyTemplate = "simple_white") -> None:
        x = cls.source_data.get_column(cls.x_col)
        y0 = cls.source_data.get_column(cls.y_col)
        y1 = cls.source_data.get_column(cls.y2_col) if cls.y2_col is not None else None
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_scattergl(
            x=x,
            y=y0,
            name=cls.y_col,
            mode="markers",
            marker=dict(color="royalblue", symbol="circle-open", opacity=0.2, size=3),
            secondary_y=False,
        )
        fig.update_xaxes(title_text=cls.x_col)
        fig.update_yaxes(rangemode="tozero")
        fig.update_yaxes(title_text=cls.y_col, secondary_y=False)
        if y1 is not None:
            fig.add_scattergl(
                x=x,
                y=y1,
                name=cls.y2_col,
                mode="markers",
                marker=dict(color="crimson", symbol="cross", size=3),
                secondary_y=True,
            )
            fig.update_yaxes(title_text=cls.y2_col, secondary_y=True)

        cls.source_fig = fig.update_layout(
            clickmode="event+select",
            template=theme,
            dragmode="select",
            autosize=True,
            height=600,
        )

    def __init__(self, start_index: int, end_index: int) -> None:
        if not self._source_set:
            raise ValueError("DataSegment must be initialized after calling DataSegment.set_source")
        self.start_index = start_index
        self.end_index = end_index
        self.data = self.source_data.slice(self.start_index, self.end_index - self.start_index + 1)
        res: Any = stats.linregress(self.data.get_column(self.x_col), self.data.get_column(self.y_col))
        self.fit_result = LinregressResult(
            slope=res.slope,
            intercept=res.intercept,
            rvalue=res.rvalue,
            pvalue=res.pvalue,
            stderr=res.stderr,
            intercept_stderr=res.intercept_stderr,
        )
        DataSegment.all_segments.append(self)
        DataSegment.all_segments.sort(key=lambda s: s.start_index)
        self.data = self.data.with_columns(
            pl.lit(self.segment_id).alias("segment_id"),
            (self.fit_result.slope * pl.col(self.x_col) + self.fit_result.intercept).alias("fitted"),
        )

    @property
    def segment_id(self) -> int:
        return DataSegment.all_segments.index(self) + 1

    @property
    def x_data(self) -> pl.Series:
        return self.data.get_column(self.x_col)

    @property
    def y_data(self) -> pl.Series:
        return self.data.get_column(self.y_col)

    @property
    def y2_data(self) -> pl.Series | None:
        return None if self.y2_col is None else self.data.get_column(self.y2_col)

    @property
    def y_fitted(self) -> pl.Series:
        return self.data.get_column("fitted")

    @property
    def y2_mean(self) -> float:
        if self.y2_data is not None:
            return cast(float, self.y2_data.mean())
        else:
            return float("nan")

    def plot(self, add: bool = True) -> go.Figure:
        if add:
            return self.source_fig.add_scattergl(
                x=self.x_data,
                y=self.y_fitted,
                mode="lines",
                line=dict(color="darkorange", width=4),
                name=f"Fit {self.segment_id}",
                hoverinfo="name+text",
                hovertext=f"slope={self.fit_result.slope:.4f}<br>r^2={self.fit_result.rvalue**2:.3f}<br>mean_y2={self.y2_mean:.1f}",
            )
        fig = go.Figure()
        point_color = self.y2_data or "lightgray"
        fig.add_scattergl(
            x=self.x_data,
            y=self.y_data,
            mode="markers",
            marker=dict(color=point_color),
            # name=f"Fit {self.segment_id}",
            # hoverinfo="name",
        )
        return fig.add_scattergl(
            x=self.x_data,
            y=self.y_fitted,
            mode="lines",
            line=dict(color="darkorange", width=4),
            name=f"Fit {self.segment_id}",
            hoverinfo="name+text",
            hovertext=f"slope={self.fit_result.slope:.4f}<br>r^2={self.fit_result.rvalue**2:.3f}<br>mean_y2={self.y2_mean:.1f}",
        )

    def result_row(self) -> ResultRow:
        return {
            "source_file": self.source_name,
            "fit_id": self.segment_id,
            "start_index": self.start_index,
            "end_index": self.end_index,
            "slope": self.fit_result.slope,
            "rsquared": self.fit_result.rvalue**2,
            "mean_y2": self.y2_mean,
            "name_x": self.x_col,
            "start_x": self.x_data.item(0),
            "end_x": self.x_data.item(-1),
            "name_y": self.y_col,
            "start_y": self.y_data.item(0),
            "end_y": self.y_data.item(-1),
            "name_y2": self.y2_col or "-",
            "start_y2": self.y2_data.item(0) if self.y2_data is not None else float("nan"),
            "end_y2": self.y2_data.item(-1) if self.y2_data is not None else float("nan"),
        }


def detect_delimiter(decoded_string: str, skip_rows: int = 0, sample_rows: int = 3) -> str:
    """
    Automatically detects the delimiter used in a text file containing tabular data.

    Args:
        decoded_string (str): The content of the file as a decoded string.
        skip_rows (int): The number of rows to skip before attempting to detect the delimiter.
        sample_rows (int): The number of rows to sample from the file (starting from the `skip_rows` row).

    Returns:
        str: The detected delimiter character.

    Raises:
        ValueError: If the delimiter cannot be detected.
        FileNotFoundError: If the file does not exist.
    """
    sample_rows = max(1, sample_rows)

    try:
        with io.StringIO(decoded_string) as file:
            if file.readline() == "":
                raise ValueError("File is empty")

            for _ in range(skip_rows):
                file.readline()

            sample = "\n".join(file.readline() for _ in range(sample_rows - 1))
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
    """
    Parse the contents of an uploaded file and return a DataFrame.

    This function decodes the base64 encoded string, detects the file type, parses the contents accordingly, and returns
    a DataFrame along with a Dash HTML component for displaying the data.

    Args:
        contents (str): The base64 encoded file contents.
        filename (str): The name of the uploaded file.
        skip_rows (int, optional): Number of rows to skip. Defaults to 0.
        separator (str, optional): Delimiter to use. If "auto", an attempt is made to detect the separator character by
        reading the first few rows of the file. Defaults to "auto".

    Returns:
        tuple[html.Div, pl.DataFrame]: A tuple containing a Dash HTML Div for displaying the data and the parsed
        DataFrame.
    """
    _, content_string = contents.split(",")
    decoded = base64.b64decode(content_string)
    suffix = Path(filename).suffix
    try:
        if suffix in {".csv", ".txt", ".tsv"}:
            content = decoded.decode("utf-8", errors="replace")
            if separator == "auto":
                separator = detect_delimiter(content, skip_rows=skip_rows)
            cleaned_content = "\n".join(
                separator.join(field.strip() for field in line.split(separator)) for line in content.splitlines()
            )
            df = pl.read_csv(io.StringIO(cleaned_content), skip_rows=skip_rows, separator=separator)
        elif suffix in {".xlsx", ".xls"}:
            df = pl.read_excel(io.BytesIO(decoded), read_options={"skip_rows": skip_rows})
        else:
            return html.Div(["Unknown file type."]), pl.DataFrame()
    except Exception as e:
        print(e)
        return html.Div(["There was an error processing this file."]), pl.DataFrame()

    df = df.select(cs.numeric()).with_row_index()
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
    ), df


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
