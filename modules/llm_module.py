import torch
import logging
from typing import Optional, Dict, Any, Generator, List
import json
import asyncio
from threading import Thread
from queue import Queue
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LocalLLM:
    def __init__(
        self,
        model_path: str = "models/llama-2-7b-chat.gguf",
        device: str = "cuda",
        context_length: int = 4096,
        n_gpu_layers: int = -1,
        temperature: float = 0.7,
        max_tokens: int = 512,
        stream: bool = True
    ):
        self.model_path = model_path
        self.device = device if torch.cuda.is_available() else "cpu"
        self.context_length = context_length
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.stream = stream
        
        try:
            from llama_cpp import Llama
            
            logger.info(f"Loading LLM model from {model_path}")
            self.model = Llama(
                model_path=model_path,
                n_ctx=context_length,
                n_gpu_layers=n_gpu_layers if self.device == "cuda" else 0,
                n_threads=8,
                use_mlock=True,
                verbose=False
            )
            logger.info("LLM model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load LLM model: {e}")
            self.model = None
            
        self.conversation_history = []
        self.system_prompt = "You are a helpful AI assistant engaged in a voice conversation. Keep responses concise and conversational."
        
    def set_system_prompt(self, prompt: str):
        self.system_prompt = prompt
        
    def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: Optional[bool] = None
    ) -> Generator[str, None, None]:
        
        if not self.model:
            yield "Model not loaded"
            return
            
        temperature = temperature or self.temperature
        max_tokens = max_tokens or self.max_tokens
        stream = stream if stream is not None else self.stream
        
        full_prompt = self._build_prompt(prompt)
        
        try:
            if stream:
                response = self.model(
                    full_prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stream=True,
                    stop=["User:", "\n\n"]
                )
                
                for chunk in response:
                    if chunk and 'choices' in chunk:
                        token = chunk['choices'][0].get('text', '')
                        if token:
                            yield token
            else:
                response = self.model(
                    full_prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stop=["User:", "\n\n"]
                )
                
                if response and 'choices' in response:
                    yield response['choices'][0].get('text', '').strip()
                    
        except Exception as e:
            logger.error(f"Generation error: {e}")
            yield f"Error: {str(e)}"
            
    def _build_prompt(self, user_input: str) -> str:
        prompt_parts = [f"System: {self.system_prompt}"]
        
        for msg in self.conversation_history[-4:]:
            prompt_parts.append(f"{msg['role']}: {msg['content']}")
            
        prompt_parts.append(f"User: {user_input}")
        prompt_parts.append("Assistant:")
        
        return "\n".join(prompt_parts)
        
    def add_to_history(self, role: str, content: str):
        self.conversation_history.append({
            "role": role.capitalize(),
            "content": content
        })
        
    def clear_history(self):
        self.conversation_history = []
        
    async def generate_async(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            self._generate_sync,
            prompt,
            temperature,
            max_tokens
        )
        return result
        
    def _generate_sync(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        
        tokens = []
        for token in self.generate(prompt, temperature, max_tokens, stream=True):
            tokens.append(token)
        return "".join(tokens)

class TransformersLLM:
    def __init__(
        self,
        model_name: str = "microsoft/phi-2",
        device: str = "cuda",
        max_length: int = 2048,
        temperature: float = 0.7,
        use_8bit: bool = True
    ):
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        
        self.device = device if torch.cuda.is_available() else "cpu"
        self.max_length = max_length
        self.temperature = temperature
        self.conversation_history = []
        self.system_prompt = "You are a helpful AI assistant engaged in a voice conversation. Keep responses concise and conversational."
        
        logger.info(f"Loading model: {model_name}")
        
        quantization_config = None
        if use_8bit and self.device == "cuda":
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_compute_dtype=torch.float16
            )
            
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        logger.info("Model loaded successfully")
    
    def set_system_prompt(self, prompt: str):
        self.system_prompt = prompt
        
    def add_to_history(self, role: str, content: str):
        self.conversation_history.append({
            "role": role.capitalize(),
            "content": content
        })
        
    def clear_history(self):
        self.conversation_history = []
        
    def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_new_tokens: int = 256,
        stream: bool = True
    ) -> Generator[str, None, None]:
        
        temperature = temperature or self.temperature
        
        full_prompt = f"{self.system_prompt}\n\nUser: {prompt}\nAssistant:"
        
        inputs = self.tokenizer(full_prompt, return_tensors="pt", padding=True)
        if self.device == "cuda":
            inputs = {k: v.cuda() for k, v in inputs.items()}
            
        with torch.no_grad():
            if stream:
                from transformers import TextIteratorStreamer
                
                streamer = TextIteratorStreamer(
                    self.tokenizer,
                    skip_prompt=True,
                    skip_special_tokens=True
                )
                
                generation_kwargs = dict(
                    inputs,
                    streamer=streamer,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=True,
                    top_p=0.95
                )
                
                thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
                thread.start()
                
                for text in streamer:
                    yield text
                    
                thread.join()
                
            else:
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=True,
                    top_p=0.95
                )
                
                response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                response = response[len(full_prompt):].strip()
                yield response