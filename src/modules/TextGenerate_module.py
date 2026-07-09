from abc import ABC, abstractmethod
import ollama, asyncio
from ctransformers import AutoModelForCausalLM


#-----------------------------------------------------------------------------------------------
class Frame_TextGenerate(ABC):
    """Базовый класс для +- однотипности других классов TextGenerate"""
    @abstractmethod
    def __init__(self):
        pass
    @abstractmethod
    def InOutPut(text):
        pass
#-----------------------------------------------------------------------------------------------
class Mollama(Frame_TextGenerate):
    def __init__(self):
        pass
    def InOutPut(self, text):
        pass