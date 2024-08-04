import os
import sys
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, validator
import uvicorn

from gpt_researcher import GPTResearcher
import asyncio
import argparse

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
    start_date: datetime
    end_date: datetime
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

    @validator('start_date', 'end_date')
    def date_must_be_in_past(cls, v):
        if v > datetime.now(ZoneInfo("America/New_York")):
            raise ValueError("Date must be in the past or present")
        return v

    @validator('start_date', 'end_date')
    def dates_must_be_within_range(cls, v):
        earliest_allowed = datetime.now(ZoneInfo("America/New_York")) - timedelta(days=365*10)  # 10 years ago
        if v < earliest_allowed:
            raise ValueError("Date must not be more than 10 years in the past")
        return v

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

def get_current_time_et():
    return datetime.now(ZoneInfo("America/New_York"))

async def fetch_report(query: str, report_type: str, sources: list, start_date: datetime, end_date: datetime) -> tuple:
    start_time = time.time()
    
    # Ensure start_date is earlier than end_date
    if start_date > end_date:
        start_date, end_date = end_date, start_date

    # Format dates as strings
    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')
    
    # Create a strong date restriction message
    date_restriction = f"IMPORTANT: Only use information from articles and content published between {start_date_str} and {end_date_str} (inclusive). Disregard any information outside this date range, even if it seems relevant. If an article's publication date is not clear, do not use it."
    
    # Modify the query to include the date restriction
    contextualized_query = f"{date_restriction}\n\nQuery: {query}\n\nWhen conducting research and writing the report, continuously verify and mention the publication dates of your sources. Include only information from sources within the specified date range."

    researcher = GPTResearcher(
        query=contextualized_query, 
        report_type=report_type, 
        config_path=None
    )
    
    await researcher.conduct_research()
    report = await researcher.write_report()
    
    # Add a note about the date range to the report
    date_range_note = f"\n\nNote: This report only contains information from sources published between {start_date_str} and {end_date_str}."
    report += date_range_note
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    return report, execution_time

def verify_api_key(credentials: HTTPAuthorizationCredentials = Security(security)):
    if credentials.credentials != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    return credentials.credentials

@app.post("/research")
async def research(query: Query, api_key: str = Depends(verify_api_key)):
    print(f"Received query: {query}")
    report, execution_time = await fetch_report(query.query, query.report_type, query.sources, query.start_date, query.end_date)
    return {"report": report, "start_date": query.start_date, "end_date": query.end_date, "execution_time": execution_time}

@app.post("/research_direct")
async def research_direct(
    query: str, 
    report_type: str = "research_report", 
    sources: list = [], 
    start_date: datetime = Depends(get_current_time_et), 
    end_date: datetime = Depends(get_current_time_et),
    api_key: str = Depends(verify_api_key)
):
    try:
        # Validate dates
        for date in [start_date, end_date]:
            if date > datetime.now(ZoneInfo("America/New_York")):
                raise ValueError("Date must be in the past or present")
            earliest_allowed = datetime.now(ZoneInfo("America/New_York")) - timedelta(days=365*10)
            if date < earliest_allowed:
                raise ValueError("Date must not be more than 10 years in the past")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    report, execution_time = await fetch_report(query, report_type, sources, start_date, end_date)
    return {"report": report, "start_date": start_date, "end_date": end_date, "execution_time": execution_time}

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
                        required=True, help="Start date (YYYY-MM-DD HH:MM:SS in America/New_York)")
    parser.add_argument("--end_date", type=lambda d: datetime.fromisoformat(d).replace(tzinfo=ZoneInfo("America/New_York")),
                        required=True, help="End date (YYYY-MM-DD HH:MM:SS in America/New_York)")
    parser.add_argument("--sources", nargs='+', default=[], help="List of source URLs")
    args = parser.parse_args()
    
    try:
        # Validate dates
        for date in [args.start_date, args.end_date]:
            if date > datetime.now(ZoneInfo("America/New_York")):
                raise ValueError("Date must be in the past or present")
            earliest_allowed = datetime.now(ZoneInfo("America/New_York")) - timedelta(days=365*10)
            if date < earliest_allowed:
                raise ValueError("Date must not be more than 10 years in the past")
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    report, execution_time = await fetch_report(args.query, args.report_type, args.sources, args.start_date, args.end_date)
    print(f"Report:\n{report}")
    print(f"Execution time: {execution_time:.2f} seconds")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        asyncio.run(run_terminal())
    else:
        run_fastapi()
