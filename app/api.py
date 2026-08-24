from fastapi import FastAPI
from pydantic import BaseModel, Field, field_validator

from app.agent import run_agent_turn

app = FastAPI(title="Web Search API")


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)

    @field_validator("query")
    @classmethod
    def query_must_not_be_blank(cls, value):
        value = value.strip()
        if not value:
            raise ValueError("query must not be blank")
        return value


@app.post("/search")
def search(request: SearchRequest):
    return run_agent_turn(request.query)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
