import random
from loguru import logger
from locust import HttpUser, task, between
from models.model_response import ModelResponse
from models.customer_support_response import CustomerSupportResponse
from helpers.runner import get_prompt, get_clean_text, get_clean_json, get_parsed_json


class TensorRTUser(HttpUser):
    """
    Locust user class that simulates users making requests to the TensorRT endpoint.
    """

    host = "http://localhost:8000/v2/models/ensemble/generate"
    wait_time = between(1, 3)  # Wait 1-3 seconds between tasks

    def on_start(self):
        """Initialize the user."""
        logger.info(f"User initialized with host: {self.host}")

    @task
    def make_inference_request(self):
        """Make an inference request to the TensorRT endpoint."""
        # Sample questions for testing
        questions = [
            "How do I file a claim?",
            "What's my policy number?",
            "Can I add my teenager to my policy?",
            "How do I update my payment method?",
            "What's covered under my comprehensive coverage?",
            "How do I cancel my policy?",
            "What's the process for renewing my policy?",
            "Can I get a quote for a new car?",
            "What discounts am I eligible for?",
            "How do I report a change of address?",
        ]

        question = random.choice(questions)
        prompt = get_prompt(question)

        payload = {
            "top_p": 0.9,
            "max_tokens": 300,
            "temperature": 0.2,
            "bad_words": list(),
            "text_input": prompt,
            "stop_words": ["<<EOF>>"],
        }

        logger.info(f"Making request with question: {question}")
        with self.client.post(
            "",  # Empty path since host is already set
            json=payload,
            catch_response=True,
        ) as response:
            if response.status_code == 200:
                try:
                    json_data = response.json()
                    raw_response = ModelResponse(**json_data)
                    cleaned_text = get_clean_text(raw_response.text_output)
                    json_string = get_clean_json(text=cleaned_text)
                    parsed_json = get_parsed_json(text=json_string)
                    CustomerSupportResponse(**parsed_json)
                    response.success()
                    logger.info(f"Successfully processed response for: {question}")
                except Exception as e:
                    error_msg = f"Failed to parse response: {str(e)}"
                    logger.error(error_msg)
                    response.failure(error_msg)
            else:
                error_msg = f"Request failed with status code: {response.status_code}"
                logger.error(error_msg)
                response.failure(error_msg)
