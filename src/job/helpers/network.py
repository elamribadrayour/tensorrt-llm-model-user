import time
import shlex
import socket
import subprocess
from typing import Iterator
from contextlib import contextmanager

from loguru import logger


def get_free_port() -> int:
    """
    Finds an available port on the local machine.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        port = s.getsockname()[1]
    return port


@contextmanager
def get_port(namespace: str, service: str, remote_port: int) -> Iterator[str]:
    """
    Establishes a port forward from a Kubernetes service to a local free port.
    Yields: URL string for local endpoint.
    """
    local_port = get_free_port()
    command = (
        f"kubectl port-forward svc/{service} {local_port}:{remote_port} -n {namespace}"
    )
    logger.info(
        f"Starting port forwarding: {service}:{remote_port} -> localhost:{local_port}"
    )

    process = subprocess.Popen(
        shlex.split(command), stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    time.sleep(2)  # Allow port forward to establish
    try:
        yield f"http://localhost:{local_port}/v2/models/ensemble/generate"
    finally:
        process.terminate()
        process.wait()
        logger.info("Port forwarding terminated.")
