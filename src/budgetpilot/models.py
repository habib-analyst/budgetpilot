"""Thin wrapper around local HF models: greedy generation with logprobs.

Cost accounting counts prompt + completion tokens (what an API would
bill). The budget controller budgets *completion* tokens (the part the
controller can actually throttle); both are reported in the results.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.set_num_threads(2)


@dataclass
class GenResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    confidence: float  # exp(mean logprob of completion tokens), in (0, 1]
    hit_cap: bool  # True if generation stopped at max_new_tokens (no EOS)


class LocalModel:
    def __init__(self, path: str, name: str):
        self.name = name
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=torch.bfloat16
        )
        self.model.eval()

    def generate(self, user_prompt: str, max_new_tokens: int) -> GenResult:
        messages = [{"role": "user", "content": user_prompt}]
        text_in = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        enc = self.tokenizer(text_in, return_tensors="pt")
        prompt_len = enc.input_ids.shape[1]
        with torch.no_grad():
            out = self.model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                output_scores=True,
                return_dict_in_generate=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        gen_ids = out.sequences[0][prompt_len:]
        n_gen = int(gen_ids.shape[0])
        # mean logprob over generated tokens
        logps = []
        for i, score in enumerate(out.scores):
            lp = torch.log_softmax(score[0].float(), dim=-1)
            logps.append(float(lp[gen_ids[i]]))
        mean_lp = sum(logps) / len(logps) if logps else -10.0
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
        eos_id = self.tokenizer.eos_token_id
        hit_cap = n_gen >= max_new_tokens and (
            n_gen == 0 or int(gen_ids[-1]) != eos_id
        )
        return GenResult(
            text=text.strip(),
            prompt_tokens=int(prompt_len),
            completion_tokens=n_gen,
            confidence=math.exp(mean_lp),
            hit_cap=hit_cap,
        )
