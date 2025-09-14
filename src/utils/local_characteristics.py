import torch
import psutil
import platform
import multiprocessing

def get_inference_env_info() -> dict:
    info = {}

    # General system information
    info["os"]             = platform.system()
    info["os_version"]     = platform.version()
    info["architecture"]   = platform.machine()
    info["python_version"] = platform.python_version()

    # CPU
    info["cpu_count_logical"]  = multiprocessing.cpu_count()
    info["cpu_count_physical"] = psutil.cpu_count(logical=False)
    info["cpu_freq_mhz"]       = psutil.cpu_freq().max

    # RAM
    ram = psutil.virtual_memory()
    info["ram_total_gb"] = round(ram.total / (1024**3), 2)

    # GPU
    if torch.cuda.is_available():
        gpu_info = {}
        devices_amount = torch.cuda.device_count()

        for i in range(devices_amount):
            props = torch.cuda.get_device_properties(i)

            gpu_info[f"cuda_device_{i}"] = {
                "name": props.name,
                "total_memory_gb": round(props.total_memory / (1024**3), 2),
                "multi_processor_count": props.multi_processor_count,
                "compute_capability": f"{props.major}.{props.minor}"
            }

        info["gpu"] = gpu_info
        info["gpu_amount"] = devices_amount
        info["cuda_available"] = True
        info["cuda_version"] = torch.version.cuda

    else:
        info["cuda_available"] = False
        info["gpu"] = None

    # Torch / Device info
    info["torch_version"] = torch.__version__
    info["torch_device"]  = str(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    return info

