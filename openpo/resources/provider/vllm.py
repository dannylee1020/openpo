from typing import Any, Dict, List, Optional, Union

from vllm.outputs import RequestOutput

from openpo.internal.error import ProviderError

from .base import LLMProvider


class VLLM:
    """VLLM provider class for high-performance inference using the vLLM engine.

    This class provides an interface to the vLLM engine for running various LLMs locally.

    Attributes:
        model: An instance of vLLM's LLM class that handles the model operations.

    Args:
        model (str): The name or path of the model to load (e.g., 'meta-llama/Llama-2-7b-chat-hf').
        **kwargs: Additional keyword arguments passed to vLLM's LLM initialization.
            These can include parameters like tensor_parallel_size, gpu_memory_utilization, etc.

    Raises:
        ImportError: If the vLLM package is not installed. The error message includes
            installation instructions.
    """

    def __init__(self, model: str, **kwargs) -> None:
        try:
            from vllm import LLM, SamplingParams
        except ImportError:
            raise ImportError(
                "vLLM requires additional dependencies. Install with: pip install openpo[eval]"
            )

        self.model = LLM(model=model, **kwargs)
        self.SamplingParam = SamplingParams

    def generate(
        self,
        messages: List,
        use_tqdm: bool = True,
        lora_request=None,
        chat_template=None,
        chat_template_content_format="auto",
        add_generation_prompt: bool = True,
        continue_final_message: bool = False,
        tools: Optional[List[Dict[str, Any]]] = None,
        mm_processor_kwargs: Optional[Dict[str, Any]] = None,
        sampling_params: Optional[Dict] = {},
    ) -> List[RequestOutput]:
        """Generate responses using the vLLM model.

        This method processes input messages and generates responses using the loaded model.
        It supports various generation parameters and features like chat templates, LoRA
        adapters, and tool-based interactions.

        Args:
            messages (List): List of input messages to process.
            use_tqdm (bool, optional): Whether to show a progress bar. Defaults to True.
            lora_request: Optional LoRA adapter configuration for on-the-fly model adaptation.
            chat_template: Optional template for formatting chat messages.
            chat_template_content_format (str, optional): Format for the chat template content.
                Defaults to "auto".
            add_generation_prompt (bool, optional): Whether to add generation prompt.
                Defaults to True.
            continue_final_message (bool, optional): Whether to continue from the final message.
                Defaults to False.
            tools (Optional[List[Dict[str, Any]]], optional): List of tools available for the model.
            mm_processor_kwargs (Optional[Dict[str, Any]], optional): Additional keyword arguments
                for multimodal processing.
            sampling_params (Optional[dict]): Model specific parameters passed to vLLM's SamplingParams.

        Returns:
            The generated vLLM output from the model.

        Raises:
            ProviderError: If generation fails, with details about the error.
        """
        try:
            params = self.SamplingParams(**sampling_params)
            res = self.model.chat(
                messages=messages,
                use_tqdm=use_tqdm,
                lora_request=lora_request,
                chat_template=chat_template,
                chat_template_content_format=chat_template_content_format,
                add_generation_prompt=add_generation_prompt,
                continue_final_message=continue_final_message,
                tools=tools,
                mm_processor_kwargs=mm_processor_kwargs,
                sampling_params=params,
            )

            return res
        except Exception as e:
            raise ProviderError(
                provider="vllm", message=f"model inference failed: {str(e)}"
            )
