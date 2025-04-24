import torch


def get_device(prefer_gpu: bool = True) -> str:
    """
    Возвращает наилучшее доступное устройство: 'cuda', 'mps' или 'cpu'.
    :param prefer_gpu: если True, приоритет отдаётся CUDA и MPS
    """
    if prefer_gpu:
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
    return "cpu"


def is_gpu_available() -> bool:
    """
    Проверяет доступность GPU (CUDA или MPS).
    """
    return torch.cuda.is_available() or torch.backends.mps.is_available()


def is_cuda_available() -> bool:
    """
    Проверяет доступность CUDA (NVIDIA GPU).
    """
    return torch.cuda.is_available()


def is_mps_available() -> bool:
    """
    Проверяет доступность MPS (GPU от Apple — Mac M1/M2).
    """
    return torch.backends.mps.is_available()


def num_gpus() -> int:
    """
    Возвращает количество доступных CUDA-устройств.
    """
    return torch.cuda.device_count() if torch.cuda.is_available() else 0


def get_device_name() -> str:
    """
    Возвращает имя CUDA-устройства, если оно доступно.
    """
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(torch.cuda.current_device())
    return "CPU"
