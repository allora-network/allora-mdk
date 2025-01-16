import json
import os
from datetime import datetime, timedelta

from fastapi import APIRouter, HTTPException, Request

# pylint: disable=import-error
from data.tiingo_data_fetcher import DataFetcher

# pylint: disable=import-error
from utils.common import load_model_from_config

router = APIRouter()


@router.get("/update-model")
async def get_update_model(
    request: Request,
):
    """
    Train the model with the given input data.
    The input data should contain the 'date', 'open', 'high', 'low', 'close', and 'volume' columns.
    Example input data:
    {
        "date": ["2024-09-06", "2024-09-07"],
        "open": [2400, 2700],
        "high": [2500, 2800],
        "low": [1500, 1900],
        "close": [1200, 2300],
        "volume": [1000000, 2000000]
    }
    """

    try:

        end_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        fetcher = DataFetcher()

        # switch token to usd pair from INFER_TOKEN env variable
        if "INFER_TOKEN" in os.environ:
            token = os.environ["INFER_TOKEN"]
        else:
            token = "ethusd"

        # lower INFER_TOKEN check for eth, btc etc. and swap to usd pair
        if token.lower() in ["eth", "btc", "ltc", "xrp", "sol", "ada", "dot", "doge"]:
            token = token.lower() + "usd"

        input_data = fetcher.fetch_tiingo_crypto_data(
            token, "2021-01-01", end_date, "1day"
        )

        # Load the model
        model = load_model_from_config(request.app.state.active_model)

        # train the model
        model.train(input_data)

        return {"training": "Model training and saving complete!"}
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON payload.") from None
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}") from e
