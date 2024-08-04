import os
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import asyncio

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator
import uvicorn
from starlette.responses import StreamingResponse

from gpt_researcher import GPTResearcher

# Load environment variables
load_dotenv()

# Set up FastAPI app
app = FastAPI()

# Set up CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins in development
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Set up security
security = HTTPBearer()

# API key (in a real-world scenario, store this securely)
API_KEY = os.getenv("API_KEY")

class Query(BaseModel):
    query: str
    report_type: str = "research_report"
    start_date: datetime | None = None
    end_date: datetime | None = None
    sources: list[str] | None = None

    @validator('start_date', 'end_date', pre=True)
    def parse_date(cls, value):
        if isinstance(value, str):
            try:
                utc_time = datetime.fromisoformat(value.rstrip('Z')).replace(tzinfo=ZoneInfo("UTC"))
                et_time = utc_time.astimezone(ZoneInfo("America/New_York"))
                return et_time
            except ValueError as e:
                raise ValueError("Invalid date format")
        return value

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

async def fetch_report(query: str, report_type: str, sources: list | None = None, start_date: datetime = None, end_date: datetime = None):
    start_time = time.time()
    
    if not start_date:
        start_date = datetime.now(ZoneInfo("America/New_York"))
    
    if not end_date:
        end_date = start_date
    
    date_range = f"from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
    
    date_context = f"Considering only information originally published {date_range} & nothing that mentions being updated, "
    contextualized_query = date_context + query

    researcher = GPTResearcher(query=contextualized_query, report_type=report_type, config_path=None)
    try:
        await researcher.conduct_research()
    except Exception as e:
        yield f"Error during research: {str(e)}\n"
    
    try:
        report = await researcher.write_report()
    except Exception as e:
        yield f"Error writing report: {str(e)}\n"
        report = "Unable to generate full report due to an error. Partial results may be available."
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    yield f"Report:\n{report}\n"
    yield f"Execution time: {execution_time:.2f} seconds\n"

def verify_api_key(credentials: HTTPAuthorizationCredentials = Security(security)):
    if credentials.credentials != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    return credentials.credentials

@app.post("/research")
async def research(query: Query, api_key: str = Depends(verify_api_key)):
    return StreamingResponse(fetch_report(query.query, query.report_type, query.sources, query.start_date, query.end_date))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
