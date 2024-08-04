import os
import time
import traceback
from datetime import datetime
from zoneinfo import ZoneInfo
import logging

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Depends, Security, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator
import uvicorn
from starlette.responses import StreamingResponse, JSONResponse

from gpt_researcher import GPTResearcher

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
                raise ValueError(f"Invalid date format: {str(e)}")
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
        yield f"Starting research for query: {query}\n"
        await researcher.conduct_research()
    except Exception as e:
        error_msg = f"Error during research: {str(e)}\n"
        error_msg += f"Traceback: {traceback.format_exc()}\n"
        logger.error(error_msg)
        yield error_msg
        return

    try:
        yield "Writing report...\n"
        report = await researcher.write_report()
    except Exception as e:
        error_msg = f"Error writing report: {str(e)}\n"
        error_msg += f"Traceback: {traceback.format_exc()}\n"
        logger.error(error_msg)
        yield error_msg
        report = "Unable to generate full report due to an error. Partial results may be available."

    end_time = time.time()
    execution_time = end_time - start_time
    
    yield f"Report:\n{report}\n"
    yield f"Execution time: {execution_time:.2f} seconds\n"

def verify_api_key(credentials: HTTPAuthorizationCredentials = Security(security)):
    if credentials.credentials != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    return credentials.credentials

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    error_msg = f"An unexpected error occurred: {str(exc)}\n"
    error_msg += f"Traceback: {traceback.format_exc()}\n"
    logger.error(error_msg)
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected error occurred. Please check the server logs for more information."}
    )

@app.post("/research")
async def research(query: Query, api_key: str = Depends(verify_api_key)):
    logger.info(f"Received query: {query}")
    return StreamingResponse(fetch_report(query.query, query.report_type, query.sources, query.start_date, query.end_date))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
