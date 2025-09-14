import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import time
import socket
import requests
import subprocess

from utils.enums import EngineType


#############################
# MODEL HOST CLASS WRAPPER #
#############################


class ModelHost:
    def __init__(
        self,
        engine_type: EngineType,
        model_path: str,
        host: str = "127.0.0.1",
        port: int = 8000,
        seed: int = 31232
    ) -> None:
        """
        Unified wrapper for launching and interacting with local model servers.

        Args:
            engine_type (EngineType): The model engine to use (e.g., llama.cpp, vLLM).
            model_path (str): Path to the model.
            host (str): Host to bind the server to.
            port (int): Port to bind the server to.
            seed (int): Seed for generation reproducibility.
        """
        self.type = engine_type
        self.model_path = model_path
        self.host = host
        self.port = port
        self.seed = seed
        self.system_prompt: str | None = None
        self.process: subprocess.Popen | None = None

        self.__run_command = self._build_command()


    ###################
    # COMMAND BUILDER #
    ###################


    def _build_command(self) -> list[str]:
        """
        Internal method to construct the command based on engine type.

        Returns:
            list[str]: CLI command to start the server.
        """
        if self.type == EngineType.llama_cpp:
            return [
                sys.executable, "-m", "llama_cpp.server",
                "--model", self.model_path,
                "--host", self.host,
                "--port", str(self.port),
            ]

        elif self.type == EngineType.vllm:
            return [
                sys.executable, "-m", "vllm.entrypoints.api_server",
                "--model", self.model_path,
                "--host", self.host,
                "--port", str(self.port),
            ]

        else:
            raise ValueError(f"Unknown engine type: {self.type}")


    #####################
    # SERVER MANAGEMENT #
    #####################
 

    def start(self) -> None:
        """
        Start the model server in a subprocess.
        """
        print(f"Starting {self.type.name} server...")
        self.process = subprocess.Popen(self.__run_command)
        self._wait_port()
        print(f"Server started on {self.host}:{self.port} (PID: {self.process.pid})")


    def stop(self) -> None:
        """
        Stop the model server if running.
        """
        if self.process:
            print(f"Stopping {self.type.name} server (PID: {self.process.pid})...")
            self.process.terminate()
            self.process.wait()
            print("Server stopped.") 

        else:
            print("No server process to stop.")


    #####################
    # GENERATION CONFIG #
    #####################
 

    def set_system_prompt(self, prompt: str) -> None:
        """
        Set a system prompt to be prepended to every generation.

        Args:
            prompt (str): System-level prompt content.
        """
        self.system_prompt = prompt


    ###################
    # TEXT GENERATION #
    ###################


    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        top_p: float = 0.95,
        max_tokens: int = 512,
        stop: list[str] = ["\n"],
        stream: bool = False,
        presence_penalty: float = 0,
        frequency_penalty: float = 0
    ) -> tuple[str, int]:
        """
        Perform text generation using the hosted model.

        Args:
            prompt (str): User input prompt.
            temperature (float): Sampling temperature.
            top_p (float): Nucleus sampling probability.
            max_tokens (int): Maximum tokens to generate.
            stop (list[str]): Stop sequences.
            stream (bool): Stream output or not.
            presence_penalty (float): Penalize new tokens for presence.
            frequency_penalty (float): Penalize token repetition.

        Returns:
            tuple[str, int]: Generated text and number of tokens.
        """
        if self.process is None:
            raise AssertionError("To accomplish generation - start server!")

        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})

        messages.append({"role": "user", "content": prompt})

        response = requests.post(
            f"http://{self.host}:{self.port}/v1/chat/completions",
            json={
                "model"            : "local-model",
                "messages"         : messages,
                "temperature"      : temperature,
                "top_p"            : top_p,
                "max_tokens"       : max_tokens,
                "stop"             : stop,
                "stream"           : stream,
                "presence_penalty" : presence_penalty,
                "frequency_penalty": frequency_penalty,
                "seed"             : self.seed,
            }
        )

        if response.status_code != 200:
            raise RuntimeError(f"Generation failed: {response.status_code}, {response.text}")

        output = response.json()
        generated_answer = output["choices"][0]["message"]["content"]
        generated_tokens = output["usage"]["completion_tokens"]

        return generated_answer, generated_tokens
    

    #############
    # UTILITIES #
    #############
    
    
    def _wait_port(self, timeout: float = 60.0, interval: float = 0.5) -> None:
        """Blocks the flow until the port starts accepting connections."""
        t0 = time.time()

        while time.time() - t0 < timeout:
            with socket.socket() as s:
                s.settimeout(interval)

                if s.connect_ex((self.host, self.port)) == 0:
                    return
                
            time.sleep(interval)

        raise TimeoutError(f"Server {self.host}:{self.port} did not start within {timeout} seconds.")

