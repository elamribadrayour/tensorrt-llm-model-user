import grpc
from loguru import logger
from helpers.runner import get_prompt, get_clean_text, get_clean_json, get_parsed_json
from models.model_response import ModelResponse
from models.customer_support_response import CustomerSupportResponse

# Import the generated gRPC code
# Note: You'll need to generate this first using:
# python -m grpc_tools.protoc -I./src/protos --python_out=./src/protos --grpc_python_out=./src/protos ./src/protos/tensorrt.proto
from protos import tensorrt_pb2
from protos import tensorrt_pb2_grpc

def make_inference_request(stub: tensorrt_pb2_grpc.TensorRTServiceStub, question: str) -> bool:
    """Make an inference request to the TensorRT gRPC endpoint."""
    prompt = get_prompt(question=question)
    
    request = tensorrt_pb2.GenerateRequest(
        top_p=0.9,
        max_tokens=300,
        temperature=0.2,
        bad_words=list(),
        text_input=prompt,
        stop_words=["<<EOF>>"]
    )

    logger.info(f"Making gRPC request with question: {question}")
    try:
        response = stub.Generate(request)
        
        raw_response = ModelResponse(text_output=response.text_output)
        cleaned_text = get_clean_text(raw_response.text_output)
        json_string = get_clean_json(text=cleaned_text)
        parsed_json = get_parsed_json(text=json_string)
        CustomerSupportResponse(**parsed_json)

        logger.info(f"Successfully processed response for: {question}")
        return True
    except grpc.RpcError as e:
        logger.error(f"gRPC call failed: {e.code()}: {e.details()}")
        return False
    except Exception as e:
        logger.error(f"Failed to parse response: {str(e)}")
        return False

def main() -> None:
    questions = [
        "How do I file a claim?",
        "What's my policy number?",
        "Can I add my teenager to my policy?",
        "How do I update my payment method?",
        "What's covered under my comprehensive coverage?",
    ]
    
    channel = grpc.insecure_channel(target='localhost:8000')
    stub = tensorrt_pb2_grpc.TensorRTServiceStub(channel=channel)
    exit(0)

    for question in questions:
        success = make_inference_request(stub=stub, question=question)
        if not success:
            logger.error(f"Failed to process question: {question}")
            continue

if __name__ == "__main__":
    main() 