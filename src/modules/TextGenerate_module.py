from abc import ABC, abstractmethod
import ollama, asyncio
from ctransformers import AutoModelForCausalLM, Config


#-----------------------------------------------------------------------------------------------
class Frame_TextGenerate(ABC):
    """Базовый класс для +- однотипности других классов TextGenerate"""
    @abstractmethod
    def __init__(self):
        pass
    @abstractmethod
    def InOutPut(self, text:str):
        pass
#-----------------------------------------------------------------------------------------------
class Mollama(Frame_TextGenerate):
    def __init__(self, model_name:str, system_promt:str):
        self.model:str = model_name
        self.messages:str = [{"role": "system", "content": system_promt}]

        # Настройки
        self.num_predict:int = 128
        ''' Максимальное количество токенов, которое модель может сгенерировать в одном ответе. По умолчанию равно 128. Установите в -1 для снятия лимита, или в -2 для отмены генерации контекста.'''
        self.num_ctx:int = 2048
        ''' Размер контекстного окна. Определяет, сколько токенов (слов и их частей) модель может «видеть» и помнить одновременно. По умолчанию обычно от 2048 до 8192, в зависимости от модели. Увеличение этого параметра может существенно нагрузить оперативную память или GPU.'''
            
        # Креативность и 'Случайность'
        self.temperature:float = None
        ''' Контролирует «креативность» и непредсказуемость ответов. Значения около 0.0 дают логичные, сфокусированные и предсказуемые ответы. Значения от 0.7 до 1.0 делают текст более разнообразным, творческим и местами хаотичным.'''
        self.top_k:int = None
        ''' Ограничивает выборку слов для токена на каждом шаге K наиболее вероятными вариантами. Предотвращает генерацию полной бессмыслицы. Значение по умолчанию — 40. Уменьшение параметра делает ответы более строгими и предсказуемыми.'''
        self.top_p:float = None 
        ''' Альтернатива top_k (называется "ядерное сэмплирование"). Задает суммарную вероятность слов, из которых модель выбирает ответ. Например, 0.9 означает, что рассматриваются 90% самых вероятных следующих слов. Рекомендуется менять либо top_p, либо temperature'''
        self.min_p:float = None
        ''' Устанавливает минимальный порог вероятности для токена относительно самого вероятного токена. Используется для предотвращения «скачков» при генерации.'''

        # Борьба с Повторениями
        self.repeat_penalty:float = None
        ''' Наказывает модель за повторение одних и тех же слов или фраз. Обычно устанавливается в пределах 1.0 — 1.5 (по умолчанию обычно 1.1). Чем выше, тем меньше повторов.'''
        self.repeat_last_n:int = None
        ''' Сколько последних сгенерированных токенов проверять на предмет повторов (по умолчанию 64). 0 — отключает штраф, -1 — проверяет весь сгенерированный текст.'''
        self.presence_penalty:float = None
        ''' Штрафует модель в целом за затрагивание новых тем, стимулируя ее придерживаться изначального контекста.'''
        self.frequency_penalty:float = None
        ''' Штрафует модель пропорционально тому, как часто она использует определенные слова. Более высокие значения заставляют модель использовать синонимы.'''
            
        # Cтоп слова
        self.stop:list = None
        ''' Список строк-стоп-слов (например, ["/n", "User:"]). Если модель генерирует любую из этих строк, процесс останавливается. Это полезно, чтобы прервать модель до того, как она начнет писать лишний текст.'''
            
        # Продвинутые математические параметры
        self.mirostat:int = None
        ''' Алгоритм динамического контроля температуры в реальном времени. 0 — выключен, 1 — Mirostat, 2 — Mirostat 2.0. Помогает поддерживать качество текста'''
        self.mirostat_tau:float = None
        ''' Целевой уровень «запутанности» (сюрприза) для алгоритма Mirostat. Меняйте, если хотите сделать текст более связным или разнообразным.'''
        self.mirostat_eta:float = None
        ''' Скорость обучения алгоритма Mirostat (как быстро он адаптирует креативность).'''

        # Вычисления и Процессор
        self.num_gpu:int = None
        ''' Количество слоев модели, которые будут загружены в видеопамять (VRAM) графического процессора. Чем больше слоев на GPU, тем быстрее работает генерация. Настройка зависит от вашей видеокарты.'''
        self.seed:int = None
        ''' Задает начальное число для генератора случайных чисел. Полезно для обеспечения воспроизводимости ответа: если задать одинаковый seed, модель при одинаковых параметрах выдаст абсолютно идентичный текст.'''
        

    def settings(self):
        return {key: val for key, val in {
            'num_predict': self.num_predict, 'num_ctx': self.num_ctx,
            'temperature': self.temperature, 'top_k': self.top_k, 'top_p': self.top_p, 'min_p': self.min_p,
            'repeat_penalty': self.repeat_penalty, 'repeat_last_n': self.repeat_last_n, 'presence_penalty': self.presence_penalty, 'frequency_penalty': self.frequency_penalty,
            'stop': self.stop,
            'mirostat': self.mirostat, 'mirostat_tau': self.mirostat_tau , 'mirostat_eta': self.mirostat_eta,
            'num_gpu': self.num_gpu, 'seed': self.seed
        }.items() if val is not None}

    def InOutPut(self, text:str):
        self.messages.append({'role': 'user', 'content': text})
        response = ollama.chat(model=self.model, messages=self.messages, stream=False, options=self.settings())
        print(response["message"]['content'])
        return response["message"]['content']
    
    async def asynhron_InOutPut(self, text:str):
        self.messages.append({'role': 'user', 'content': text})
        asynhron_client = ollama.AsyncClient()
        response = await asynhron_client.chat(model=self.model, messages=self.messages, stream=False, options=self.settings())
        print(response["message"]['content'])
        return response["message"]['content']
    # asyncio.run(asynhron_InOutPut(TEXT)) <-- если вам нужно запустить это\
#-----------------------------------------------------------------------------------------------
class Mctransformers(Frame_TextGenerate):
    def __init__(self, model_path:str, model_type:str, system_promt:str, lib:str="basic"):
        self.chat_history:list = []
        self.system_promt:str = system_promt + "\n"
        config=Config(top_k=40, top_p=0.95, temperature=0.8,repetition_penalty=1.1, last_n_tokens=64, seed=-1,batch_size=8,threads=-1, max_new_tokens=256,stream=False,reset=True,context_length=-1,gpu_layers=0,mmap=True,mlock=False)
        self.llm = AutoModelForCausalLM.from_pretrained(model_path, model_type=model_type, lib=lib, config=config)

    def reinit_llm(self, model_path:str, model_type:str, system_promt:str, lib:str="basic",
                   top_k=40, top_p=0.95, temperature=0.8,repetition_penalty=1.1, last_n_tokens=64, 
                   seed=-1,batch_size=8,threads=-1, max_new_tokens=256,
                   stream=False,reset=True,
                   context_length=-1,gpu_layers=0,
                   mmap=True,mlock=False):
        config=Config(top_k=top_k, top_p=top_p, temperature=temperature, repetition_penalty=repetition_penalty, last_n_tokens=last_n_tokens, seed=seed, batch_size=batch_size, threads=threads, max_new_tokens=max_new_tokens, stream=stream, reset=reset, context_length=context_length, gpu_layers=gpu_layers, mmap=mmap, mlock=mlock)
        self.llm = AutoModelForCausalLM.from_pretrained(model_path, model_type=model_type, lib=lib, config=config)
    
    def get_promt(self):
        for message in self.chat_history[-10:]:
            full_prompt += f"{message['role']}: {message['content']}\n"
        full_prompt += "assistant:"
        return full_prompt
    
    def InOutPut(self, text:str, max_new_tokens:int=128):
        self.chat_history.append({"role": "user", "content": text})
        respone = self.llm(self.get_promt(), max_new_tokens=max_new_tokens)
        self.chat_history.append({"role": "assistant", "content": respone})
        return respone
#-----------------------------------------------------------------------------------------------
if __name__ == "__main__":
    pass