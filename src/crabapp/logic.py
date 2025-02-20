import base64
import io
from typing import Any
import polars as pl
import polars.selectors as cs
import janitor.polars  # noqa: F401
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from dataclasses import dataclass, field

from crabapp.utils import LinregressResult


class DataManager:
    _instance: "DataManager | None" = None
    
    def __new__(cls) -> "DataManager":
        if cls._instance is None:
            cls._instance = super(DataManager, cls).__new__(cls)
        return cls._instance
        

@dataclass
class LinearFit:
    start_index: int
    end_index: int
    df: pl.DataFrame
    x_col: str
    y_col: str
    y2_col: str | None = field(default=None)
    result: LinregressResult = field(init=False)

    def __post_init__(self) -> None:
        res: Any = stats.linregress(self.df.get_column(self.x_col), self.df.get_column(self.y_col))
        self.result = LinregressResult(
            slope=res.slope,
            intercept=res.intercept,
            rvalue=res.rvalue,
            pvalue=res.pvalue,
            stderr=res.stderr,
            intercept_stderr=res.intercept_stderr,
        )
        self.df = self.df.with_columns((self.result.slope * pl.col(self.x_col) + self.result.intercept).alias("fitted"))

    def result_df(self) -> pl.DataFrame:
        result_dict = {
            "start_index": self.start_index,
            "end_index": self.end_index,
            "slope": self.result.slope,
            "rvalue": self.result.rvalue,
            "mean_y2": self.df.get_column(self.y2_col).mean() if self.y2_col is not None else None,
            "name_x": self.x_col,
            "start_x": self.df.get_column(self.x_col).item(0),
        }

class DataSet:

    @classmethod
    def from_presens(cls, file_name: str, file_content: str, separator: str = ";", skip_rows: int = 57) -> "DataSet":
        utf8_data = base64.b64decode(file_content).decode("utf-8", errors="replace")
        utf8_cleaned = "\n".join(
            separator.join(field.strip() for field in line.split(separator)) for line in utf8_data.splitlines()
        )
        df = (
            pl.scan_csv(io.StringIO(utf8_cleaned), separator=separator, skip_rows=skip_rows, row_index_name="index")
            .select(cs.numeric())
            .clean_names(remove_special=True)
            .collect()
        )
        return cls(Path(file_name).stem, df)

    def __init__(self, name: str, df: pl.DataFrame) -> None:
        self.name = name
        self.df = df
        self.fig = make_subplots(specs=[[{"secondary_y": True}]])
        self.fits: list[LinearFit] = []
        self._x_col = ""
        self._y_col = ""
        self._y2_col: str | None = None

    @property
    def can_add_fit(self) -> bool:
        return self._x_col != "" and self._y_col != ""
    
    @property
    def columns(self) -> list[str]:
        return self.df.columns

    def plot(self, x_col: str, y_col: str, y2_col: str | None = None, theme: str = "simple_white") -> go.Figure:
        self._x_col = x_col
        self._y_col = y_col
        self._y2_col = y2_col
        x = self.df.get_column(x_col)
        y = self.df.get_column(y_col)
        y2 = self.df.get_column(y2_col) if y2_col is not None else None
        self.fig.add_scattergl(
            x=x,
            y=y,
            name=y_col,
            mode="markers",
            marker=dict(color="royalblue", symbol="circle-open", opacity=0.2, size=3),
            secondary_y=False,
        )
        self.fig.update_xaxes(title_text=x_col)
        self.fig.update_yaxes(rangemode="tozero")
        self.fig.update_yaxes(title_text=y_col, secondary_y=False)
        if y2 is not None:
            self.fig.add_scattergl(
                x=x,
                y=y2,
                name=y2_col,
                mode="markers",
                marker=dict(color="crimson", symbol="cross", size=3),
                secondary_y=True,
            )
            self.fig.update_yaxes(title_text=y2_col, secondary_y=True)
        self.fig.update_layout(clickmode="event+select", template=theme, dragmode="select", autosize=True, height=600)
        return self.fig

    def add_fit(self, start_index: int, end_index: int) -> None:
        if not self.can_add_fit:
            raise ValueError("Cannot add a fit to a plot that has not been created yet. Call plot() first.")
        fit_df = self.df.slice(start_index, end_index - start_index + 1)
        if self._y2_col is not None:
            y2_mean = fit_df.get_column(self._y2_col).mean()
            