import pandas as pd
from fastapi import APIRouter, Body, HTTPException, Request

from utils.common import load_model_from_config

router = APIRouter()


@router.post("/inference")
async def perform_inference(
    request: Request,
    payload: dict = Body(...),
):
    """
    Perform inference on the given input data passed as JSON.
    The payload should contain the 'topic_id' and the input data for inference.
    Example payload:
    {
        "topic_id": 1,
        "data": {...}
    }
    """

    topic_id = payload.get("topic_id")
    if not topic_id:
        raise HTTPException(
            status_code=400, detail="Topic ID is required in the payload."
        )

    input_data = payload.get("data")
    if not input_data:
        raise HTTPException(
            status_code=400, detail="Input data is required in the payload."
        )

    try:
        # Convert input_data into a DataFrame
        input_data = pd.DataFrame(payload)

        model = load_model_from_config(request.app.state.active_model)
        # Load the model
        model.load()
        # Perform inference
        predictions = model.inference(input_data)
        return {"predictions": predictions.to_dict(orient="records")}
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Post Inference error: {str(e)}"
        ) from e
