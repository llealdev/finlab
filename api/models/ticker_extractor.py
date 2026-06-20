from pydantic import BaseModel, Field


class TickerResult(BaseModel):
    ticker: str = Field(
        pattern="^[A-Z]{1,5}$",
        description="Stock ticker symbol (1-5 uppercase letters)",
    )
