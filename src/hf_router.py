from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from openai import OpenAI


class HFRouterError(RuntimeError):
    pass


class HFRouterClient:
    """
    Hugging Face Inference Providers Router용 OpenAI 호환 클라이언트.
    base_url: https://router.huggingface.co/v1
    model example: google/gemma-4-26B-A4B-it:deepinfra
    """

    def __init__(
        self,
        hf_token: str,
        candidates: List[str],
        base_url: str = "https://router.huggingface.co/v1",
        timeout: int = 60,
    ):
        self.hf_token = hf_token
        self.candidates = [m for m in candidates if m]
        self.base_url = base_url
        self.timeout = timeout
        if not hf_token:
            raise HFRouterError("HF_TOKEN이 비어 있습니다. .env 또는 Streamlit Secrets에 HF_TOKEN을 설정하세요.")
        if not self.candidates:
            raise HFRouterError("사용 가능한 HF Router 모델 후보가 없습니다.")

        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.hf_token,
            timeout=self.timeout,
        )

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 1800,
        response_format: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        errors = []
        for model in self.candidates:
            try:
                kwargs = {
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }
                if response_format:
                    kwargs["response_format"] = response_format

                completion = self.client.chat.completions.create(**kwargs)
                content = completion.choices[0].message.content or ""
                return {
                    "ok": True,
                    "model": model,
                    "content": content,
                    "raw": completion.model_dump() if hasattr(completion, "model_dump") else str(completion),
                }
            except Exception as e:
                errors.append(f"{model}: {type(e).__name__}: {e}")
                continue

        raise HFRouterError("모든 HF Router 모델 호출 실패:\n" + "\n".join(errors))

    def json_chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.1,
        max_tokens: int = 2200,
    ) -> Dict[str, Any]:
        result = self.chat(messages, temperature=temperature, max_tokens=max_tokens)
        content = result["content"]
        parsed = extract_json(content)
        result["json"] = parsed
        return result


def extract_json(text: str) -> Any:
    """
    LLM 출력에서 JSON을 최대한 안전하게 추출.
    JSON mode를 지원하지 않는 provider/model 대응용.
    """
    text = (text or "").strip()
    if not text:
        raise ValueError("빈 응답입니다.")

    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fenced:
        text = fenced.group(1).strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    start_obj = text.find("{")
    start_arr = text.find("[")
    if start_obj == -1 and start_arr == -1:
        raise ValueError("JSON 시작 문자를 찾지 못했습니다.")

    if start_arr != -1 and (start_obj == -1 or start_arr < start_obj):
        start = start_arr
        end = text.rfind("]")
    else:
        start = start_obj
        end = text.rfind("}")

    if end <= start:
        raise ValueError("JSON 끝 문자를 찾지 못했습니다.")

    return json.loads(text[start : end + 1])
