import json
import random
from typing import Dict, List, Optional, Union

import litellm
from pydantic import BaseModel

from peony.internal import helper, prompt


class ExtractModel(BaseModel):
    main_answer: str
    key_points: str


class Completions:
    def create_preference(
        self,
        *,
        model: str,
        messages: List = [],
        # Optional OpenAI params
        timeout: Optional[Union[float, int]] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        n: Optional[int] = None,
        stream: Optional[bool] = None,
        stream_options: Optional[dict] = None,
        stop=None,
        max_completion_tokens: Optional[int] = None,
        max_tokens: Optional[int] = None,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        logit_bias: Optional[dict] = None,
        user: Optional[str] = None,
        # openai v1.0+ new params
        response_format: Optional[dict] = None,
        seed: Optional[int] = None,
        tools: Optional[List] = None,
        tool_choice: Optional[str] = None,
        parallel_tool_calls: Optional[bool] = None,
        logprobs: Optional[bool] = None,
        top_logprobs: Optional[int] = None,
        deployment_id=None,
        # soon to be deprecated params by OpenAI
        functions: Optional[List] = None,
        function_call: Optional[str] = None,
        # set api_base, api_version, api_key
        base_url: Optional[str] = None,
        api_version: Optional[str] = None,
        api_key: Optional[str] = None,
        model_list: Optional[list] = None,
        # diff frequency threshold
        diff_frequency: Optional[float] = 0.0,
        # Optional liteLLM function params
        **kwargs,
    ):
        if helper.should_run(diff_frequency):
            core_response = litellm.completion(
                model=model,
                messages=[
                    {"role": "system", "content": prompt.EXTRACT_PROMPT},
                    *messages[1:],
                ],
                response_format=ExtractModel,
                temperature=0.0,
            )

            core_content = json.loads(core_response.choices[0].message.content)
            original_prompt = messages[0]["content"]

            pref_response = litellm.completion(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": prompt.PREF_PROMPT.format(
                            original_prompt,
                            core_content["main_answer"],
                            core_content["key_points"],
                        ),
                    },
                    *messages[1:],
                ],
                drop_params=True,
                response_format=response_format,
                temperature=1.0,
                presence_penalty=0.6,
                frequency_penalty=0.6,
                timeout=timeout,
                n=n,
                stream=stream,
                stream_options=stream_options,
                stop=stop,
                max_completion_tokens=max_completion_tokens,
                max_tokens=max_tokens,
                logit_bias=logit_bias,
                user=user,
                seed=seed,
                tools=tools,
                tool_choice=tool_choice,
                parallel_tool_calls=parallel_tool_calls,
                logprobs=logprobs,
                top_logprobs=top_logprobs,
                deployment_id=deployment_id,
                functions=functions,
                function_call=function_call,
                base_url=base_url,
                api_version=api_version,
                api_key=api_key,
                model_list=model_list,
                **kwargs,
            )

            return pref_response

        completion = litellm.completion(
            model=model,
            messages=messages,
            drop_params=True,
            timeout=timeout,
            temperature=temperature,
            top_p=top_p,
            n=n,
            stream=stream,
            stream_options=stream_options,
            stop=stop,
            max_completion_tokens=max_completion_tokens,
            max_tokens=max_tokens,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            logit_bias=logit_bias,
            user=user,
            response_format=response_format,
            seed=seed,
            tools=tools,
            tool_choice=tool_choice,
            parallel_tool_calls=parallel_tool_calls,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            deployment_id=deployment_id,
            functions=functions,
            function_call=function_call,
            base_url=base_url,
            api_version=api_version,
            api_key=api_key,
            model_list=model_list,
            **kwargs,
        )

        return completion

    def create(
        self,
        *,
        model: str,
        messages: List = [],
        # Optional OpenAI params
        timeout: Optional[Union[float, int]] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        n: Optional[int] = None,
        stream: Optional[bool] = None,
        stream_options: Optional[dict] = None,
        stop=None,
        max_completion_tokens: Optional[int] = None,
        max_tokens: Optional[int] = None,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        logit_bias: Optional[dict] = None,
        user: Optional[str] = None,
        # openai v1.0+ new params
        response_format: Optional[dict] = None,
        seed: Optional[int] = None,
        tools: Optional[List] = None,
        tool_choice: Optional[str] = None,
        parallel_tool_calls: Optional[bool] = None,
        logprobs: Optional[bool] = None,
        top_logprobs: Optional[int] = None,
        deployment_id=None,
        # soon to be deprecated params by OpenAI
        functions: Optional[List] = None,
        function_call: Optional[str] = None,
        # set api_base, api_version, api_key
        base_url: Optional[str] = None,
        api_version: Optional[str] = None,
        api_key: Optional[str] = None,
        model_list: Optional[list] = None,  # pass in a list of api_base,keys, etc.
        # Optional liteLLM function params
        **kwargs,
    ):
        completions = litellm.completion(
            model=model,
            messages=messages,
            drop_params=True,
            timeout=timeout,
            temperature=temperature,
            top_p=top_p,
            n=n,
            stream=stream,
            stream_options=stream_options,
            stop=stop,
            max_completion_tokens=max_completion_tokens,
            max_tokens=max_tokens,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            logit_bias=logit_bias,
            user=user,
            response_format=response_format,
            seed=seed,
            tools=tools,
            tool_choice=tool_choice,
            parallel_tool_calls=parallel_tool_calls,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            deployment_id=deployment_id,
            functions=functions,
            function_call=function_call,
            base_url=base_url,
            api_version=api_version,
            api_key=api_key,
            model_list=model_list,
            **kwargs,
        )

        return completions
