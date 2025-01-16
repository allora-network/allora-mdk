""" FastAPI app to perform inference using a model. """

# conda create --name modelworker python=3.9
# conda activate modelworker
# pip install -r requirements.txt
# uvicorn main:app --reload --port 8000
# feel free to test with curl or using the browser with the URL http://localhost:8000/docs

import importlib
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from prometheus_fastapi_instrumentator import Instrumentator


# Lifespan event to handle initialization
@asynccontextmanager
async def lifespan(fapp: FastAPI):  # pylint: disable=unused-argument
    """
    Lifespan event to handle initialization of the FastAPI app.
    """
    active_model = os.getenv("MODEL")  # Environment variable 'MODEL' for Docker or CLI
    if active_model:
        fapp.state.active_model = active_model
        print(f"Default model set from environment variable: {active_model}")
    else:
        print("No model specified via environment variable. Using default at runtime.")

    # The 'yield' allows any lifespan context to be cleaned up at the end of the lifespan
    yield


# Assign lifespan to the FastAPI app
app = FastAPI(
    lifespan=lifespan,
    title="Allora Worker API",
    description="API for performing inference using various models within the Allora platform.",
    version="1.0.0",
    contact={
        "name": "Allora Support",
        "url": "https://github.com/allora-network/allora-worker/issues",
    },
)

# Add instrumentation to measure response time and other metrics
Instrumentator().instrument(app).expose(app)


# Default route
@app.get("/", include_in_schema=False)
async def base():
    return {"message": "Welcome to the Allora Worker API!"}


# Include routers dynamically
def include_routers(_app: FastAPI, directories: list):
    """
    Dynamically import routers from specified directories and include them in the FastAPI app.
    :param app: FastAPI app instance
    :param directories: List of directories to search for routers (e.g., ['/endpoints/get', '/endpoints/post'])
    """
    for directory in directories:
        module_dir = os.path.join(os.path.dirname(__file__), directory)

        # Iterate over all files in the directory
        for filename in os.listdir(module_dir):
            if filename.endswith(".py") and not filename.startswith("__"):
                module_name = f"{directory.replace('/', '.')}.{filename[:-3]}"
                module = importlib.import_module(module_name)
                # Check if the module has a `router` attribute
                if hasattr(module, "router"):
                    _app.include_router(module.router)


# Automatically include all routers from the 'get' and 'post' directories
include_routers(app, ["endpoints/get", "endpoints/post"])


# place health check endpoint here
@app.get("/healthcheck")
async def healthcheck():
    return {"status": "healthy"}


# Catch-all route for all undefined paths
@app.api_route(
    "/{full_path:path}",
    methods=["GET", "POST", "PUT", "DELETE"],
    include_in_schema=False,
)
async def catch_all(request: Request, full_path: str):
    client_host = request.client.host  # type: ignore  # Get client IP address
    return {
        "error": "The endpoint you are trying to reach does not exist.",
        "path": full_path,
        "method": request.method,
        "client_ip": client_host,
        "message": "Please check the available endpoints.",
    }
