import math
import os
import statistics
from typing import Dict, List, Union

import numpy as np
import sacrebleu
import torch
import transformers
from tqdm import tqdm

from utils.logging_utils import log_for_0


def _mean_std_sem(values):
    n = len(values)
    mean = sum(values) / n
    std = statistics.pstdev(values) if n > 1 else 0.0
    sem = std / math.sqrt(n) if n > 1 else 0.0
    return mean, std, sem


def compute_bleu(hypotheses, references):
    return sacrebleu.corpus_bleu(hypotheses, [references], lowercase=True, use_effective_order=True).score


def compute_rouge(hypotheses, references, return_std=False):
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    r1, r2, rL = [], [], []
    for hyp, ref in zip(hypotheses, references):
        s = scorer.score(ref, hyp)
        r1.append(s["rouge1"].fmeasure * 100)
        r2.append(s["rouge2"].fmeasure * 100)
        rL.append(s["rougeL"].fmeasure * 100)
    m1, s1, e1 = _mean_std_sem(r1)
    m2, s2, e2 = _mean_std_sem(r2)
    mL, sL, eL = _mean_std_sem(rL)
    means = {"rouge1": m1, "rouge2": m2, "rougeL": mL}
    if not return_std:
        return means
    stds = {
        "rouge1_std": s1,
        "rouge2_std": s2,
        "rougeL_std": sL,
        "rouge1_sem": e1,
        "rouge2_sem": e2,
        "rougeL_sem": eL,
    }
    return means, stds


class NLL:
    def __init__(self):
        self.reset()

    def reset(self):
        self.mean_value = torch.tensor(0.0, dtype=torch.float32)
        self.weight = torch.tensor(0.0, dtype=torch.float32)

    def update(self, value: Union[float, torch.Tensor, np.ndarray], weight: Union[float, torch.Tensor, np.ndarray] = 1.0):
        if not isinstance(value, torch.Tensor):
            value = torch.as_tensor(value, dtype=torch.float32)
        if weight is not None and not isinstance(weight, torch.Tensor):
            weight = torch.as_tensor(weight, dtype=torch.float32)
        weight = torch.broadcast_to(weight, value.shape)
        if value.numel() == 0:
            return
        self.mean_value = self.mean_value + value.float().sum().cpu()
        self.weight = self.weight + weight.float().sum().cpu()


class Perplexity(NLL):
    def compute(self) -> torch.Tensor:
        return torch.exp(self.mean_value / self.weight)


class MeanMetric:
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum_value = torch.tensor(0.0, dtype=torch.float32)
        self.count = torch.tensor(0.0, dtype=torch.float32)

    def update(self, value: Union[float, torch.Tensor, np.ndarray]):
        if not isinstance(value, torch.Tensor):
            value = torch.as_tensor(value, dtype=torch.float32)
        self.sum_value = self.sum_value + value.float().sum().cpu()
        self.count = self.count + value.numel()

    def compute(self) -> torch.Tensor:
        return self.sum_value / self.count


class Metrics:
    def __init__(
        self,
        gen_ppl_eval_model_name_or_path=None,
        eval_ppl_batch_size=None,
        eval_context_size=1024,
        device=None,
    ) -> None:
        self.gen_ppl = Perplexity()
        self.sample_entropy = MeanMetric()
        self.eval_ppl_batch_size = eval_ppl_batch_size
        self.gen_ppl_eval_model_name_or_path = gen_ppl_eval_model_name_or_path
        self.eval_context_size = eval_context_size
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self._ppl_model = None

        use_fast = "mt5" not in gen_ppl_eval_model_name_or_path.lower()
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(gen_ppl_eval_model_name_or_path, use_fast=use_fast)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def reset(self):
        self.gen_ppl.reset()
        self.sample_entropy.reset()

    def _eval_retokenize(self, text_samples, max_length):
        out = self.tokenizer(
            text_samples,
            return_tensors="pt",
            return_token_type_ids=False,
            return_attention_mask=True,
            truncation=True,
            padding=True,
            max_length=max_length,
        )
        return out["input_ids"], out["attention_mask"], self.eval_context_size

    def _load_ppl_model(self):
        if self._ppl_model is None:
            from transformers import AutoModelForCausalLM

            log_for_0(f"Loading Torch model: {self.gen_ppl_eval_model_name_or_path}")
            self._ppl_model = AutoModelForCausalLM.from_pretrained(self.gen_ppl_eval_model_name_or_path).to(self.device)
            self._ppl_model.eval()
            log_for_0("PPL model cached for reuse")
        return self._ppl_model

    @torch.no_grad()
    def record_generative_perplexity(
        self,
        text_samples: List[str],
        max_length: int,
        retokenize: bool = True,
    ) -> Dict:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        eval_model = self._load_ppl_model()

        if retokenize:
            samples, attn_mask, eval_context_size = self._eval_retokenize(text_samples, max_length=max_length)
        else:
            samples = torch.as_tensor(text_samples)
            attn_mask = torch.ones_like(samples)
            eval_context_size = samples.shape[-1]

        batch_size = self.eval_ppl_batch_size or samples.shape[0]
        batch_size = min(batch_size, samples.shape[0])
        num_batches = (samples.shape[0] + batch_size - 1) // batch_size
        log_for_0(f"PPL: batch_size={batch_size}, {num_batches} batches")

        per_sample_nll_sum = np.zeros(samples.shape[0], dtype=np.float64)
        per_sample_token_count = np.zeros(samples.shape[0], dtype=np.float64)

        eos_token_id = self.tokenizer.eos_token_id
        for i in tqdm(range(num_batches), desc="Evaluating perplexity"):
            batch_start = i * batch_size
            batch_end = min((i + 1) * batch_size, samples.shape[0])
            batch_samples = samples[batch_start:batch_end].to(self.device)
            batch_attn_mask = attn_mask[batch_start:batch_end].to(self.device)

            for chunk_start in range(0, batch_samples.shape[1], eval_context_size):
                chunk_end = min(chunk_start + eval_context_size, batch_samples.shape[1])
                sample_chunk = batch_samples[:, chunk_start:chunk_end]
                attn_mask_chunk = batch_attn_mask[:, chunk_start:chunk_end]
                logits = eval_model(sample_chunk, attention_mask=attn_mask_chunk).logits
                targets = sample_chunk[:, 1:]
                logits_pred = logits[:, :-1, :]
                log_probs = torch.log_softmax(logits_pred.float(), dim=-1)
                nlls = -torch.gather(log_probs, dim=-1, index=targets[..., None]).squeeze(-1)
                is_eos = sample_chunk == eos_token_id
                first_eos = torch.cumsum(is_eos.to(torch.int32), dim=-1) == 1
                token_mask = sample_chunk != eos_token_id
                valid_tokens = (first_eos[:, 1:] | token_mask[:, 1:]).float()
                weighted_nlls = (nlls * valid_tokens).cpu().numpy()
                valid_tokens_np = valid_tokens.cpu().numpy()
                self.gen_ppl.update(weighted_nlls, valid_tokens_np)
                per_sample_nll_sum[batch_start:batch_end] += weighted_nlls.sum(axis=-1)
                per_sample_token_count[batch_start:batch_end] += valid_tokens_np.sum(axis=-1)

        with np.errstate(divide="ignore", invalid="ignore"):
            per_sample_ppl = np.exp(per_sample_nll_sum / per_sample_token_count)
        per_sample_ppl = np.where(per_sample_token_count > 0, per_sample_ppl, np.nan).tolist()

        per_sample_entropy = []
        samples_np = samples.cpu().numpy()
        attn_np = attn_mask.cpu().numpy()
        for i in range(samples.shape[0]):
            valid_len = int(attn_np[i].sum())
            valid_tokens = samples_np[i, :valid_len]
            _, counts = np.unique(valid_tokens, return_counts=True)
            probs = counts.astype(np.float32) / counts.sum()
            entropy = float(-np.sum(probs * np.log(probs + 1e-10)))
            per_sample_entropy.append(entropy)
            self.sample_entropy.update(entropy)

        return {
            "ppl": float(self.gen_ppl.compute()),
            "per_sample_ppl": per_sample_ppl,
            "mean_entropy": sum(per_sample_entropy) / len(per_sample_entropy),
        }
