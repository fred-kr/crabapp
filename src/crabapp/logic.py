import base64
import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import polars as pl
import polars.selectors as cs

import janitor.polars  # noqa: F401 # isort:skip

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

from crabapp.utils import LinregressResult


@dataclass
class LinearFit:
    start_index: int
    end_index: int
    df: pl.DataFrame
    x_name: str
    y_name: str
    y2_name: str | None = field(default=None)
    y2_first: float = field(default=float("nan"))
    y2_last: float = field(default=float("nan"))
    result: LinregressResult = field(init=False)
    x_first: float = field(init=False)
    x_last: float = field(init=False)
    y_first: float = field(init=False)
    y_last: float = field(init=False)

    def __post_init__(self) -> None:
        x_data = self.df.get_column(self.x_name)
        y_data = self.df.get_column(self.y_name)
        self.x_first = x_data.item(0)
        self.x_last = x_data.item(-1)
        self.y_first = y_data.item(0)
        self.y_last = y_data.item(-1)
        if self.y2_name is not None:
            y2_data = self.df.get_column(self.y2_name)
            self.y2_first = y2_data.item(0)
            self.y2_last = y2_data.item(-1)

        res: Any = stats.linregress(x_data, y_data)
        self.result = LinregressResult(
            slope=res.slope,
            intercept=res.intercept,
            rvalue=res.rvalue,
            pvalue=res.pvalue,
            stderr=res.stderr,
            intercept_stderr=res.intercept_stderr,
        )
        self.df = self.df.with_columns(
            (self.result.slope * pl.col(self.x_name) + self.result.intercept).alias("fitted")
        )

    @property
    def x_data(self) -> pl.Series:
        return self.df.get_column(self.x_name)

    @property
    def y_data(self) -> pl.Series:
        return self.df.get_column(self.y_name)

    @property
    def y_fitted(self) -> pl.Series:
        return self.df.get_column("fitted")

    @property
    def y2_mean(self) -> float:
        if self.y2_name is not None:
            return cast(float, self.df.get_column(self.y2_name).mean())
        else:
            return float("nan")

    @property
    def rsquared(self) -> float:
        return self.result.rvalue**2

    def make_result(self, source_file: str, fit_id: int) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "source_file": source_file,
                "fit_id": fit_id,
                "start_index": self.start_index,
                "end_index": self.end_index,
                "slope": self.result.slope,
                "rsquared": self.rsquared,
                "y2_mean": self.y2_mean,
                "x_name": self.x_name,
                "x_first": self.x_first,
                "x_last": self.x_last,
                "y_name": self.y_name,
                "y_first": self.y_first,
                "y_last": self.y_last,
                "y2_name": self.y2_name,
                "y2_first": self.y2_first,
                "y2_last": self.y2_last,
            }
        )


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
        return cls(file_name, df)

    def __init__(self, file_path: str, df: pl.DataFrame) -> None:
        self.file_path = file_path
        self.df = df
        self.fig = make_subplots(specs=[[{"secondary_y": True}]])
        self.fits: list[LinearFit] = []
        self._x_name = ""
        self._y_name = ""
        self._y2_name: str | None = None

    @property
    def columns(self) -> list[str]:
        return self.df.columns

    @property
    def source_file_name(self) -> str:
        return Path(self.file_path).name

    @property
    def source_file_stem(self) -> str:
        return Path(self.file_path).stem

    def plot(self, x_name: str, y_name: str, y2_name: str | None = None, theme: str = "simple_white") -> go.Figure:
        self._x_name = x_name
        self._y_name = y_name
        self._y2_name = y2_name
        x = self.df.get_column(x_name)
        y = self.df.get_column(y_name)
        y2 = self.df.get_column(y2_name) if y2_name is not None else None
        self.fig.add_scattergl(
            x=x,
            y=y,
            name=y_name,
            mode="markers",
            marker=dict(color="royalblue", symbol="circle-open", opacity=0.2, size=3),
            secondary_y=False,
        )
        self.fig.update_xaxes(title_text=x_name)
        self.fig.update_yaxes(rangemode="tozero")
        self.fig.update_yaxes(title_text=y_name, secondary_y=False)
        if y2 is not None:
            self.fig.add_scattergl(
                x=x,
                y=y2,
                name=y2_name,
                mode="markers",
                marker=dict(color="crimson", symbol="cross", size=3),
                secondary_y=True,
            )
            self.fig.update_yaxes(title_text=y2_name, secondary_y=True)
        self.fig.update_layout(clickmode="event+select", template=theme, dragmode="select", autosize=True, height=600)
        return self.fig

    def add_fit(self, start_index: int, end_index: int) -> None:
        if not self._x_name or not self._y_name:
            return
        fit_df = self.df.slice(start_index, end_index - start_index + 1)
        fit = LinearFit(start_index, end_index, fit_df, self._x_name, self._y_name, self._y2_name)
        self.fits.append(fit)
        self.fits.sort(key=lambda fit: fit.start_index)

        self.fig.add_scattergl(
            x=fit.x_data,
            y=fit.y_fitted,
            mode="lines",
            line=dict(color="darkorange", width=4),
            name=f"Fit {self.fits.index(fit) + 1}",
            hoverinfo="name+text",
            hovertext=f"slope={fit.result.slope:.4f}<br>r^2={fit.result.rvalue**2:.3f}<br>y2_mean={fit.y2_mean:.1f}",
        )

    def make_result_table(self) -> pl.DataFrame:
        self.fits.sort(key=lambda fit: fit.start_index)
        return pl.concat(fit.make_result(self.source_file_name, i) for i, fit in enumerate(self.fits, start=1))
