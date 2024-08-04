import os
import sys
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import asyncio
import argparse

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, validator
import uvicorn
from starlette.responses import StreamingResponse

from gpt_researcher import GPTResearcher

# Load environment variables
load_dotenv()

# Set up FastAPI app
app = FastAPI()

# Set up security
security = HTTPBearer()

# API key (in a real-world scenario, store this securely)
API_KEY = os.getenv("API_KEY")

class Query(BaseModel):
    query: str
    report_type: str = "research_report"
    start_date: datetime | None = None
    end_date: datetime | None = None
    sources: list[str] = []

    @validator('start_date', 'end_date', pre=True)
    def parse_date(cls, value):
        if isinstance(value, str):
            try:
                utc_time = datetime.fromisoformat(value.rstrip('Z')).replace(tzinfo=ZoneInfo("UTC"))
                et_time = utc_time.astimezone(ZoneInfo("America/New_York"))
                print(f"Original UTC time: {utc_time}, Converted ET time: {et_time}")
                return et_time
            except ValueError as e:
                print(f"Date parsing error: {e}")
                raise ValueError("Invalid date format")
        return value

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

def get_current_time_et():
    return datetime.now(ZoneInfo("America/New_York"))

def get_referenced_date(query: str, current_date: datetime) -> datetime:
    days = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
    query_lower = query.lower()
   
    for i, day in enumerate(days):
        if day in query_lower:
            days_diff = (current_date.weekday() - i) % 7
            return (current_date - timedelta(days=days_diff)).replace(hour=0, minute=0, second=0, microsecond=0)
   
    if "yesterday" in query_lower:
        return (current_date - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    elif "today" in query_lower:
        return current_date.replace(hour=0, minute=0, second=0, microsecond=0)
   
    return current_date.replace(hour=0, minute=0, second=0, microsecond=0)

async def fetch_report(query: str, report_type: str, sources: list = [], start_date: datetime = None, end_date: datetime = None):
    start_time = time.time()
    
    if not start_date:
        start_date = get_current_time_et()
    
    if not end_date:
        end_date = start_date
    
    date_range = f"from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
    
    date_context = f"Considering only information originally published {date_range} & nothing that mentions being updated, "
    contextualized_query = date_context + query

    researcher = GPTResearcher(query=contextualized_query, report_type=report_type, config_path=None)
    try:
        await researcher.conduct_research()
    except Exception as e:
        print(f"Error during research: {e}")
    
    try:
        report = await researcher.write_report()
    except Exception as e:
        print(f"Error writing report: {e}")
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
    print(f"Received query: {query}")
    return StreamingResponse(fetch_report(query.query, query.report_type, query.sources, query.start_date, query.end_date))

@app.post("/research_direct")
async def research_direct(query: str, report_type: str = "research_report", sources: list = [], start_date: datetime = None, end_date: datetime = None, api_key: str = Depends(verify_api_key)):
    return StreamingResponse(fetch_report(query, report_type, sources, start_date, end_date))

def run_fastapi():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1", help="Host to run the server on")
    parser.add_argument("--port", type=int, default=8000, help="Port to run the server on")
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)

async def run_terminal():
    parser = argparse.ArgumentParser(description="GPT Researcher")
    parser.add_argument("query", type=str, help="Research query")
    parser.add_argument("--report_type", type=str, default="research_report", help="Type of report")
    parser.add_argument("--start_date", type=lambda d: datetime.fromisoformat(d).replace(tzinfo=ZoneInfo("America/New_York")),
                        help="Start date (YYYY-MM-DD HH:MM:SS in America/New_York)")
    parser.add_argument("--end_date", type=lambda d: datetime.fromisoformat(d).replace(tzinfo=ZoneInfo("America/New_York")),
                        help="End date (YYYY-MM-DD HH:MM:SS in America/New_York)")
    parser.add_argument("--sources", nargs='+', default=[], help="List of source URLs")
    args = parser.parse_args()
    
    async for chunk in fetch_report(args.query, args.report_type, args.sources, args.start_date, args.end_date):
        print(chunk, end='')

if __name__ == "__main__":
    if len(sys.argv) > 1:
        asyncio.run(run_terminal())
    else:
        run_fastapi()
