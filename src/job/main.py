import re
import json
from typing import Any

import requests
from loguru import logger
from pydantic import ValidationError

from helpers.network import get_port
from models.model_response import ModelResponse
from models.customer_support_response import CustomerSupportResponse


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
    Attempts to extract the first JSON object from a string using a regex.
    """
    # Matches the first occurrence of a JSON object starting with '{' and ending with '}'
    json_match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if json_match:
        return json_match.group(0)
    return ""


def get_prompt() -> str:
    """
    Generates the prompt sent to the model.
    """
    prompt = (
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
        "Hello, I would like to know if my son is covered by my car insurance policy to drive my vehicle? [/INST]</s>"
    )
    return prompt


def get_payload(prompt: str) -> dict[str, Any]:
    """
    Builds the payload used for the inference request.
    """
    payload = {
        "top_p": 0.9,
        "bad_words": [],
        "max_tokens": 150,
        "temperature": 0.2,
        "text_input": prompt,
        "stop_words": ["<<EOF>>"],
    }
    return payload


def get_response(url: str, payload: dict[str, Any]) -> CustomerSupportResponse:
    """
    Sends an inference request, then wraps the output as a CustomerSupportResponse.
    It first extracts a JSON object from the model's response text.
    """
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        json_data = response.json()

        try:
            raw_response = ModelResponse(**json_data)
            cleaned_text = get_clean_text(raw_response.text_output)
        except ValidationError as ve:
            logger.error("Raw response validation error: " + str(ve))
            cleaned_text = get_clean_text(json_data.get("text_output", ""))

        # Extract the first JSON object from the cleaned text.
        json_string = get_clean_json(cleaned_text)
        try:
            parsed_json = json.loads(json_string)
            cs_response = CustomerSupportResponse(**parsed_json)
            return cs_response
        except Exception as e:
            logger.error("Error parsing or validating extracted JSON: " + str(e))
            return CustomerSupportResponse(
                classification="Uncategorized",
                answer=cleaned_text,
                next_action=None,
                additional_details={},
            )
    except Exception as e:
        logger.error(f"Request Error: {str(e)}")
        return CustomerSupportResponse(
            classification="Error", answer="", next_action=None, additional_details={}
        )


def main() -> None:
    with get_port(
        namespace="model-serving", service="tensorrt-service", remote_port=8000
    ) as url:
        prompt = get_prompt()
        payload = get_payload(prompt)
        structured_response = get_response(url, payload)

        logger.info("Structured Output:")
        logger.info("Classification: " + structured_response.classification)
        logger.info("Answer: " + structured_response.answer)
        logger.info("Next Action: " + str(structured_response.next_action))
        logger.info(
            "Additional Details: " + str(structured_response.additional_details)
        )


if __name__ == "__main__":
    main()
