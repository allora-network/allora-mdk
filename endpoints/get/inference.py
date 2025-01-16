import json

from fastapi import APIRouter, HTTPException, Request

from data.coingecko_data_fetcher import CoingeckoDataFetcher

# pylint: disable=import-error
from utils.common import load_model_from_config

router = APIRouter()


@router.get("/inference/{coin}")
async def get_inference(
    request: Request,
    coin: str,
):
    # pylint: disable=line-too-long
    """
    This is a dumb endpoint that returns the last prediction value from the model.

    """
    try:

        cgfetcher = CoingeckoDataFetcher(cache_duration=60)
        input_data = cgfetcher.fetch_real_time_data(coin)

        # Load the model
        model = load_model_from_config(request.app.state.active_model)
        model.load()

        # Perform inference
        predictions = model.inference(input_data)

        # return proper/full response from model.inference
        # return {"predictions": predictions.to_dict(orient="records")}

        # Get the last prediction value
        last_prediction = predictions.iloc[-1]

        # Assuming the prediction is under the "prediction" key
        return float(last_prediction["prediction"])

    except json.JSONDecodeError:
        raise HTTPException(
            status_code=400, detail="Invalid JSON payload for GetInference."
        ) from None
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"GetInference error: {str(e)}"
        ) from e
