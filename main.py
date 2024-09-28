from fastapi import FastAPI, Form, Depends, HTTPException, status, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import logging
import subprocess
from config import config  # Import the Config class
from langchain_agent import langchain_multiagent_method
from autogen_multiagent import autogen_multiagent_input
from autogen_adaptive import autogen_rag_method
from langchain_adaptive import langchain_rag_method
from langchain_poc import agentic_method
app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO)

# Use Jinja2Templates for serving HTML
templates = Jinja2Templates(directory="templates")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# OAuth2PasswordBearer instance
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def authenticate_user(username: str, password: str):
    if username == config.AUTH_USERNAME and password == config.AUTH_PASSWORD:
        return True
    return False

def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        username, password = token.split(':', 1)
        if authenticate_user(username, password):
            return username
    except ValueError:
        pass
    raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    try:
        return templates.TemplateResponse("index.html", {"request": request})
    except Exception as e:
        logging.error(f"Error loading HTML file: {e}")
        return HTMLResponse(content="Error loading page", status_code=500)
    
class QuerySubmit(BaseModel):
    message:str
    currentUserId:str
    firstClick:bool

class SubmitRequest(BaseModel):
    method: str
    user_input: str

@app.post("/submit")
async def submit(request: SubmitRequest, token: str = Depends(oauth2_scheme)):
    try:
        username, password = token.split(':', 1)
        if not authenticate_user(username, password):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")

        if request.method == "langchain":
            result=langchain_multiagent_method(request.user_input)
        elif request.method == "autogen":
            result = autogen_multiagent_input(request.user_input)
        elif request.method == "langchainrag":
            result=langchain_rag_method(request.user_input)
        elif request.method == "autogenrag":
            result = autogen_rag_method(request.user_input)
        else:
            raise ValueError("Invalid method selected")
        return {"result": result}
    except Exception as e:
        logging.error(f"Error processing input: {e}")
        return {"error": "Internal Server Error"}, 500
    
@app.post("/query_submit")
async def query_submit(request:QuerySubmit,token:str= Depends(oauth2_scheme)):
    try:
        print("Inside try")
        username, password = token.split(':', 1)
        if not authenticate_user(username, password):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
        result=agentic_method(request.message,request.currentUserId,request.firstClick)
        return {"result": result}

    except Exception as e:
        logging.error(f"Error processing input: {e}")
        return {"error": "Internal Server Error"}, 500



@app.get("/get-token")
async def get_token():
    token = f"{config.AUTH_USERNAME}:{config.AUTH_PASSWORD}"
    return JSONResponse(content={"token": token})




