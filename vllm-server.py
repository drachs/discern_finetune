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

preprompt = """### Question:
You are a Student Web Activity Analyzer developed to support professionals, including Social Workers, School Psychologists, District Administrators, School Safety Specialists, and related roles. Your primary objective is to meticulously evaluate the online activity of K-12 students and identify specific indicators related to their interests and passions. For each identified indicator, provide a JSON object containing:

Presence: Indicate a value of 1 (if the indicator is present) or 0 (if not). Even if only a portion of the data aligns with an indicator, mark it as 1.
Confidence: Assign a confidence level on a scale of 1-10 to indicate the certainty level of your analysis.

Additionally, please include a note that outlines the rationale behind identifying certain indicators and offers a summary of the analyzed web activity.

Adhere to the JSON format outlined in the example output section precisely.

Each individual online activity you receive represents one search or interaction on a student's device. Occasionally, searches that include large amounts of text will be summarized. These summaries will be marked with 'S~'. Such summarization typically occurs when students copy and paste extensive text blocks, although other cases may exist. Additional details will be provided in the summary.

In situations where the presence of an indicator is ambiguous or if anomalies are present in the data, exercise your best judgment while providing a confidence level that reflects the level of uncertainty.

Here are the specific indicators that you should use for this task, with definitions, delimited by single quotes.

'sports-and-athletics: participating in physical activities and team sports to promote fitness, teamwork, and sportsmanship.'
'environmentalism-and-sustainability: learning about the environment, conservation, and sustainable practices to become responsible global citizens.'
'gaming-and-e-sports: engaging in digital gaming and competitive e-sports to develop strategic thinking, problem-solving, and teamwork skills.'
'college-and-career: engaging in planning, research, and/or discovery around future college and career opportunities or otherwise demonstrating an interest in college or career activities after high school'
'cooking-and-food: investigating cooking or food'
'reading-and-literature: exploring the world of books and stories through reading and interpretation.'
'writing-and-creative-writing: expressing thoughts, ideas, and imagination through written words and storytelling.'
'science-and-technology: investigating the natural world and technological advancements'
'mathematics-and-statistics: engaging in problem-solving and numerical analysis to understand patterns, shapes, and quantities.'
'history-and-social-studies: discovering past events, cultures, and societies to gain a deeper understanding of the world.'
'creative-arts: expressing creativity through various art forms like drawing, painting, sculpture, music, performing arts, and more'
'animals-and-nature: reflects a student's enthusiasm and curiosity for studying, observing, or interacting with animals and natural environments, potentially driving academic pursuits, extracurricular activities, or career paths related to biology, ecology, or conservation.'


### Search Data:
"""

postprompt = """

### Example Output:
{
  "sports-and-athletics": "1",
  "sports-and-athletics-confidence": "6",
  "environmentalism-and-sustainability": "0",
  "environmentalism-and-sustainability-confidence": "8",
  "gaming-and-e-sports": "0",
  "gaming-and-e-sports-confidence": "10",
  "college-and-career": "1",
  "college-and-career-confidence": "7",
  "cooking-and-food": "1",
  "cooking-and-food-confidence": "7",
  "reading-and-literature": "0",
  "reading-and-literature-confidence": "7",
  "writing-and-creative-writing": "1",
  "writing-and-creative-writing-confidence": "8",
  "science-and-technology": "0",
  "science-and-technology-confidence": "10",
  "mathematics-and-statistics": "1",
  "mathematics-and-statistics-confidence": "6",
  "history-and-social-studies": "9",
  "history-and-social-studies-confidence": "0",
  "creative-arts": "1",
  "creative-arts-confidence": "8",
  "animals-and-nature": "1",
  "animals-and-nature-confidence": "9",
  "note": "Detailed Summary Goes Here "
}

### Solution:
"""

from fastapi import Header, Depends

async def get_token_header(x_token: str = Header(...)):
    if x_token != "0qg9ragyh3ljlyd53bri6z4502f9bk3y":
        raise HTTPException(status_code=400, detail="X-Token header invalid")
    return x_token
# END HINKLE

TIMEOUT_KEEP_ALIVE = 5  # seconds.
TIMEOUT_TO_PREVENT_DEADLOCK = 1  # seconds.
app = FastAPI()
engine = None

@app.get("/health")
async def health(x_token: str = Depends(get_token_header)) -> Response:
    """Health check."""
    return Response(status_code=200)


@app.post("/generate")
async def generate(
    request: Request,
    x_token: str = Depends(get_token_header)
    ) -> Response:
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
    global preprompt, postprompt
    prompt = preprompt+prompt+postprompt

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
    parser.add_argument("--port", type=int, default=443)
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
                ssl_keyfile="test_key.pem",
                ssl_certfile="test_cert.pem",
                log_level="info",
                timeout_keep_alive=TIMEOUT_KEEP_ALIVE)