from typing import Dict, Iterable, List, Optional, Union

import httpx
import openai
from typing_extensions import Literal


class Completions:
    def __init__(self):
        self.client = openai.OpenAI()

    def create(
        self,
        *,
        model: str,
        messages: str,
        frequency_penalty: Optional[float] | None = None,
        function_call=None,
        logit_bias: Optional[Dict[str, int]] | None = None,
        logprobs: Optional[bool] | None = None,
        max_completion_tokens: Optional[int] | None = None,
        max_tokens: Optional[int] | None = None,
        metadata: Optional[Dict[str, str]] | None = None,
        n: Optional[int] | None = None,
        parallel_tool_calls: bool = False,
        presence_penalty: Optional[float] | None = None,
        response_format=None,
        seed: Optional[int] | None = None,
        service_tier: Optional[Literal["auto", "default"]] | None = None,
        stop: Union[Optional[str], List[str]] | None = None,
        store: Optional[bool] = False,
        stream: Optional[Literal[False]] | None = None,
        stream_options=None,
        temperature: Optional[float] | None = None,
        tool_choice=None,
        tools=None,
        top_logprobs: Optional[int] | None = None,
        top_p: Optional[float] | None = None,
        user: str | None = None,
        extra_headers=None,
        extra_query=None,
        extra_body=None,
        timeout: float | httpx.Timeout | None = None,
    ):
        tools_kwargs = {
            "tools": tools,
            "parallel_tool_calls": parallel_tool_calls,
            "tool_choice": tool_choice,
        }

        if tools:
            completions = self.client.chat.completions.create(
                model=model,
                messages=messages,
                frequency_penalty=frequency_penalty,
                function_call=function_call,
                logit_bias=logit_bias,
                logprobs=logprobs,
                max_completion_tokens=max_completion_tokens,
                max_tokens=max_tokens,
                metadata=metadata,
                n=n,
                presence_penalty=presence_penalty,
                response_format=response_format,
                seed=seed,
                service_tier=service_tier,
                stop=stop,
                store=store,
                stream=stream,
                stream_options=stream_options,
                temperature=temperature,
                tool_choice=tool_choice,
                top_logprobs=top_logprobs,
                top_p=top_p,
                user=user,
                extra_headers=extra_headers,
                extra_query=extra_query,
                extra_body=extra_body,
                timeout=timeout,
                **tool_kwarg,
            )

            return completions

        completions = self.client.chat.completions.create(
            model=model,
            messages=messages,
            frequency_penalty=frequency_penalty,
            function_call=function_call,
            logit_bias=logit_bias,
            logprobs=logprobs,
            max_completion_tokens=max_completion_tokens,
            max_tokens=max_tokens,
            metadata=metadata,
            n=n,
            presence_penalty=presence_penalty,
            response_format=response_format,
            seed=seed,
            service_tier=service_tier,
            stop=stop,
            store=store,
            stream=stream,
            stream_options=stream_options,
            temperature=temperature,
            top_logprobs=top_logprobs,
            top_p=top_p,
            user=user,
            extra_headers=extra_headers,
            extra_query=extra_query,
            extra_body=extra_body,
            timeout=timeout,
        )

        return completions
