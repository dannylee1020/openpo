import json
import random
from typing import Dict, Iterable, List, Optional, Union

import httpx
import openai
from pydantic import BaseModel
from typing_extensions import Literal


class ExtractModel(BaseModel):
    main_answer: str
    key_points: str


class PreferenceModel(BaseModel):
    first_response: str
    second_response: str | None


class ResponseModel(BaseModel):
    first_response: str


class Completions:
    def __init__(self):
        self.client = openai.OpenAI()

    def parse_preference(
        self,
        *,
        model: str,
        messages: List[Dict[str, str]],
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
        response_format: Optional[BaseModel] | None = None,
        seed: Optional[int] | None = None,
        service_tier: Optional[Literal["auto", "default"]] | None = None,
        stop: Union[Optional[str], List[str]] | None = None,
        store: Optional[bool] = False,
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
        diff_frequency: Optional[float] = 0.0,
    ):
        if random.random() <= diff_frequency:
            print("preference")

            core_response = self.client.beta.chat.completions.parse(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "extract key facts and answers from the resposne of this query from user",
                    },
                    *messages[1:],
                ],
                response_format=ExtractModel,
                temperature=0.1,
            )

            core_content = json.loads(core_response.choices[0].message.content)

            pref_response = self.client.beta.chat.completions.parse(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": f"Provide exactly two different responses. Answer in your own style. vary your language and tone, but do not contradict or add to these core facts. Main answer: {core_content['main_answer']}, Key points: {core_content['key_points']}",
                    },
                    *messages[1:],
                ],
                response_format=PreferenceModel,
                temperature=1.2,
                presence_penalty=0.6,
                frequency_penalty=0.6,
            )

            return pref_response

        print("no preference")

        res = self.client.beta.chat.completions.parse(
            model=model,
            messages=messages,
            frequency_penalty=frequency_penalty,
            function_call=function_call,
            logit_bias=logit_bias,
            logprobs=logprobs,
            max_completion_tokens=max_completion_tokens,
            max_tokens=max_tokens,
            metadata=metadata,
            n=1,
            presence_penalty=presence_penalty,
            response_format=response_format or ResponseModel,
            seed=seed,
            service_tier=service_tier,
            stop=stop,
            store=store,
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

        return res

    def stream_preference(
        self,
        *,
        model: str,
        messages: List[Dict[str, str]],
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
        response_format: Optional[BaseModel] | None = None,
        seed: Optional[int] | None = None,
        service_tier: Optional[Literal["auto", "default"]] | None = None,
        stop: Union[Optional[str], List[str]] | None = None,
        store: Optional[bool] = False,
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
        diff_frequency: Optional[float] = 0.0,
    ):
        if random.random() <= diff_frequency:
            core_response = self.client.beta.chat.completions.parse(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "extract key facts and answers from the resposne of this query from user",
                    },
                    *messages[1:],
                ],
                response_format=ExtractModel,
                temperature=0.1,
            )

            core_content = json.loads(core_response.choices[0].message.content)

            pref_response = self.client.beta.chat.completions.stream(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": f"answer in your own style. vary your language and tone, but do not contradict or add to these core facts. Main answer: {core_content['main_answer']}, Key points: {core_content['key_points']}",
                    },
                    *messages[1:],
                ],
                response_format=PreferenceModel,
                n=2,
                temperature=1.2,
                presence_penalty=0.8,
            )

            return pref_response

        res = self.client.beta.chat.completions.stream(
            model=model,
            messages=messages,
            frequency_penalty=frequency_penalty,
            function_call=function_call,
            logit_bias=logit_bias,
            logprobs=logprobs,
            max_completion_tokens=max_completion_tokens,
            max_tokens=max_tokens,
            metadata=metadata,
            n=1,
            presence_penalty=presence_penalty,
            response_format=response_format or ResponseModel,
            seed=seed,
            service_tier=service_tier,
            stop=stop,
            store=store,
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

        return res

    def parse(
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
            completions = self.client.beta.chat.completions.parse(
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

        completions = self.client.beta.chat.completions.parse(
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
