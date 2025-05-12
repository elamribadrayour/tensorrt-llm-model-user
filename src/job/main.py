import re
import json
import asyncio
from typing import Any
from argparse import ArgumentParser

import aiohttp
from loguru import logger
from result import Result, Ok, Err
from pydantic import ValidationError

from helpers.network import get_port
from models.model_response import ModelResponse
from models.customer_support_response import CustomerSupportResponse


def get_prompts(data_type: str, data: str) -> list[dict[str, str]]:
    if data_type == "json":
        with open(data, "r") as f:
            prompts = json.load(f)["prompts"]
    else:
        prompts = [data]

    output = [
        {
            "question": prompt,
            "prompt": (
                "<s>[INST] <<SYS>>\n"
                "You are a helpful and professional customer support agent for State Farm insurance. "
                "Your response should be structured as a JSON object in the following format:\n\n"
                "{\n"
                '  "classification": "<the type of issue (e.g., Coverage Inquiry)>",\n'
                '  "answer": "<your answer here>",\n'
                '  "next_action": "<the action you expect the user to take, if any>",\n'
                '  "additional_details": { "<any extra context>" }\n'
                "}\n\n"
                "Please ensure your response is a valid JSON. Do not include any additional text or formatting outside of the JSON object.\n"
                "<</SYS>>\n\n"
                f"{prompt} [/INST]</s>"
            ),
        }
        for prompt in prompts
    ]
    return output


def get_payloads(data_type: str, data: str) -> list[dict[str, Any]]:
    """
    Builds the payload used for the inference request.
    """
    prompts = get_prompts(data_type=data_type, data=data)
    payloads = [
        {
            "payload": {
                "top_p": 0.9,
                "max_tokens": 300,
                "temperature": 0.2,
                "bad_words": list(),
                "text_input": prompt["prompt"],
                "stop_words": ["<<EOF>>"],
            },
            "question": prompt["question"],
        }
        for prompt in prompts
    ]
    return payloads


def get_clean_text(text: str) -> str:
    """
    Cleans model output text by removing unwanted tokens and multiple newlines.
    """
    eof_index = text.find("<<EOF>>")
    if eof_index != -1:
        text = text[:eof_index]
    cleaned = re.sub(r"<s>|</s>|\[INST\]|\[/INST\]|<<SYS>>|<</SYS>>", "", text)
    cleaned = re.sub(r"\n+", "\n", cleaned)
    return cleaned.strip()


def get_clean_json(text: str) -> str:
    """
    Scans the text to extract the first balanced JSON object.
    It returns the substring from the first occurrence of '{'
    to the corresponding matching '}'.
    """
    start = text.find("{")
    if start == -1:
        return ""

    stack = list()
    end = -1
    for i in range(start, len(text)):
        char = text[i]
        if char == "{":
            stack.append("{")
        elif char == "}":
            if stack:
                stack.pop()
                if not stack:
                    end = i + 1  # include the closing brace
                    break

    if end != -1:
        return text[start:end]
    return ""


async def get_response(
    session: aiohttp.ClientSession, url: str, payload: dict[str, Any]
) -> Result[CustomerSupportResponse, str]:
    """
    Sends an inference request, then wraps the output as a CustomerSupportResponse.
    It first extracts a JSON object from the model's response text.
    """
    raw_response = None
    cleaned_text = None
    json_string = None
    try:
        async with session.post(url, json=payload) as response:
            response.raise_for_status()
            json_data = await response.json()

            try:
                raw_response = ModelResponse(**json_data)
                cleaned_text = get_clean_text(raw_response.text_output)
            except ValidationError:
                cleaned_text = get_clean_text(json_data.get("text_output", ""))

            json_string = get_clean_json(text=cleaned_text)

            try:
                parsed_json = json.loads(json_string)
                cs_response = CustomerSupportResponse(**parsed_json)
                return Ok(cs_response)
            except Exception as e:
                return Err(
                    f"Error parsing JSON: {str(e)} - raw_response: {raw_response} -- json_string: {json_string} -- cleaned_text: {cleaned_text}"
                )
    except Exception as e:
        return Err(
            f"Error: {str(e)} - raw_response: {raw_response} -- json_string: {json_string} -- cleaned_text: {cleaned_text}"
        )


async def process_payload(
    session: aiohttp.ClientSession, url: str, payload: dict[str, Any], question: str
) -> dict | None:
    logger.info(f"Processing payload: {question}")
    response = await get_response(session=session, url=url, payload=payload)
    if response.is_err():
        logger.error(
            f"An error occurred while processing payload: {response.err_value}"  # type: ignore
        )
        return None

    return {
        "question": question,
        "response": response.ok_value.model_dump(),  # type: ignore
    }


async def process_payloads(
    url: str, payloads: list[dict[str, Any]], max_concurrent: int
) -> list[dict | None]:
    async with aiohttp.ClientSession() as session:
        tasks = list()
        for payload in payloads:
            task = process_payload(
                session=session,
                url=url,
                payload=payload["payload"],
                question=payload["question"],
            )
            tasks.append(task)

        results = list()
        for i in range(0, len(tasks), max_concurrent):
            chunk = tasks[i : i + max_concurrent]
            chunk_results = await asyncio.gather(*chunk)
            results.extend(chunk_results)

        return results


async def async_main() -> None:
    parser = ArgumentParser()
    parser.add_argument(
        "--data-type",
        type=str,
        required=True,
        default="text",
        choices=["json", "text"],
        help="The type of data to use for the inference request",
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="The path to the data file to use for the inference request or the text to use for the inference request",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=10,
        help="Maximum number of concurrent requests",
    )
    args = parser.parse_args()

    payloads = get_payloads(data_type=args.data_type, data=args.data)

    with (
        get_port(
            namespace="model-serving", service="tensorrt-service", remote_port=8000
        ) as url,
        open("outputs.example.json", "w") as f,
    ):
        results = await process_payloads(
            url=url, payloads=payloads, max_concurrent=args.max_concurrent
        )
        f.write(
            json.dumps([result for result in results if result is not None], indent=4)
        )


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
