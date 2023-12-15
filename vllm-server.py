import argparse
import json
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
import uvicorn

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams
from vllm.utils import random_uuid

# HINKLE
from lmformatenforcer import CharacterLevelParser
from lmformatenforcer.integrations.vllm import build_vllm_logits_processor
from lmformatenforcer import JsonSchemaParser
from pydantic import BaseModel, Field
import asyncio
from vllm.transformers_utils.tokenizer import get_tokenizer
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
# END HINKLE

TIMEOUT_KEEP_ALIVE = 5  # seconds.
TIMEOUT_TO_PREVENT_DEADLOCK = 1  # seconds.
app = FastAPI()
engine = None

@app.get("/health")
async def health() -> Response:
    """Health check."""
    return Response(status_code=200)


@app.post("/generate")
async def generate(request: Request) -> Response:
    """Generate completion for the request.

    The request should be a JSON object with the following fields:
    - prompt: the prompt to use for the generation.
    - stream: whether to stream the results or not.
    - other fields: the sampling parameters (See `SamplingParams` for details).
    """
    request_dict = await request.json()
    prompt = request_dict.pop("prompt")
    stream = request_dict.pop("stream", False)
    sampling_params = SamplingParams(**request_dict)
    request_id = random_uuid()

    # HINKLE - Activate the logits processor
    sampling_params.logits_processors = [logits_processor]
    sampling_params.stop="}"
    # END HINKLE

    results_generator = engine.generate(prompt, sampling_params, request_id)

    # Streaming case
    async def stream_results() -> AsyncGenerator[bytes, None]:
        async for request_output in results_generator:
            prompt = request_output.prompt
            text_outputs = [
                prompt + output.text for output in request_output.outputs
            ]
            ret = {"text": text_outputs}
            yield (json.dumps(ret) + "\0").encode("utf-8")

    if stream:
        return StreamingResponse(stream_results())

    # Non-streaming case
    final_output = None
    async for request_output in results_generator:
        if await request.is_disconnected():
            # Abort the request if the client disconnects.
            await engine.abort(request_id)
            return Response(status_code=499)
        final_output = request_output

    assert final_output is not None
    prompt = final_output.prompt
    # HINKLE
    #text_outputs = [prompt + output.text for output in final_output.outputs]
    text_outputs = [output.text + "}" for output in final_output.outputs]
    # END HINKLE
    ret = {"text": text_outputs}
    return JSONResponse(ret)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default=None)
    parser.add_argument("--port", type=int, default=8000)
    parser = AsyncEngineArgs.add_cli_args(parser)
    args = parser.parse_args()

    engine_args = AsyncEngineArgs.from_cli_args(args)
    engine = AsyncLLMEngine.from_engine_args(engine_args)

    # HINKLE - Implement our logits processor
    class AnswerFormat(BaseModel):
        sports_and_athletics: str = Field(alias="sports-and-athletics")
        sports_and_athletics_confidence: str = Field(alias="sports-and-athletics-confidence")
        environmentalism_and_sustainability: str = Field(alias="environmentalism-and-sustainability")
        environmentalism_and_sustainability_confidence: str = Field(alias="environmentalism-and-sustainability-confidence")
        gaming_and_e_sports: str = Field(alias="gaming-and-e-sports")
        gaming_and_e_sports_confidence: str = Field(alias="gaming-and-e-sports-confidence")
        college_and_career: str = Field(alias="college-and-career")
        college_and_career_confidence: str = Field(alias="college-and-career-confidence")
        cooking_and_food: str = Field(alias="cooking-and-food")
        cooking_and_food_confidence: str = Field(alias="cooking-and-food-confidence")
        reading_and_literature: str = Field(alias="reading-and-literature")
        reading_and_literature_confidence: str = Field(alias="reading-and-literature-confidence")
        writing_and_creative_writing: str = Field(alias="writing-and-creative-writing")
        writing_and_creative_writing_confidence: str = Field(alias="writing-and-creative-writing-confidence")
        science_and_technology: str = Field(alias="science-and-technology")
        science_and_technology_confidence: str = Field(alias="science-and-technology-confidence")
        mathematics_and_statistics: str = Field(alias="mathematics-and-statistics")
        mathematics_and_statistics_confidence: str = Field(alias="mathematics-and-statistics-confidence")
        history_and_social_studies: str = Field(alias="history-and-social-studies")
        history_and_social_studies_confidence: str = Field(alias="history-and-social-studies-confidence")
        creative_arts: str = Field(alias="creative-arts")
        creative_arts_confidence: str = Field(alias="creative-arts-confidence")
        animals_and_nature: str = Field(alias="animals-and-nature")
        animals_and_nature_confidence: str = Field(alias="animals-and-nature-confidence")
        note: str

    global logits_processor
    
    engine_model_config = asyncio.run(engine.get_model_config())
    tokenizer = get_tokenizer(
        engine_model_config.tokenizer,
        tokenizer_mode=engine_model_config.tokenizer_mode,
        trust_remote_code=engine_model_config.trust_remote_code)

    logits_processor = build_vllm_logits_processor(tokenizer, JsonSchemaParser(AnswerFormat.schema()))    
    # HINKLE - END

    uvicorn.run(app,
                host=args.host,
                port=args.port,
                log_level="debug",
                timeout_keep_alive=TIMEOUT_KEEP_ALIVE)