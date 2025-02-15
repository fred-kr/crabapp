import polars as pl


class DataSet:
    def __init__(self, name: str, data: pl.DataFrame, path: str | None = None) -> None:
        self.name = name
        self.data = data
        self.path = path or ""
        
    @property
    def columns(self) -> list[str]:
        return self.data.columns
