import os
import yaml
import asyncio
from worker import run_worker

BASE_CONFIG = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yml")


def main():
    with open(BASE_CONFIG, "r", encoding="utf8") as f:
        params = yaml.safe_load(f)

    if params.get("localhost"):
        params["broker"]["host"] = "127.0.0.1"

    asyncio.run(
        run_worker(
            server_host=params["broker"]["host"],
            server_port=params["broker"]["port"],
            token=params["worker"]["token"],
            hf_token=params["worker"]["hf_token"]
        )
    )


if __name__ == "__main__":
    main()
