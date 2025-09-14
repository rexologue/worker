from enum import IntEnum

class EngineType(IntEnum):
    llama_cpp = 0
    vllm      = 1
    
class WorkerState(IntEnum):
    idle = 0
    busy = 1