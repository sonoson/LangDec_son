
import os
import re
import random
import logging
import math
import numpy as np
from collections import defaultdict
from typing import Any, Dict, List, Literal, Optional, Tuple

import torch
import torch.nn.functional as F

from interfaces import BaseGenerator, BasePRM

level = logging.INFO
if os.getenv('DEBUG', False):
    level = logging.DEBUG

# 로깅 설정
logging.basicConfig(
    level=level,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("bootstrap_search.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def aggregate(vals, agg_method):
    if agg_method == "min":
        aggregate_scores, _ = torch.min(vals, dim=-1)
    elif agg_method == "mean":
        aggregate_scores = torch.mean(vals, dim=-1)
    elif agg_method == "sum":
        aggregate_scores = torch.sum(vals, dim=-1)
    elif agg_method == "last":
        aggregate_scores = vals[:, -1]
    elif agg_method == "prod":
        aggregate_scores = torch.cumprod(vals, dim=1)[:, -1]
    else:
        raise NotImplementedError(
            f"{agg_method} aggregation is not implemented."
        )
    return aggregate_scores


def select_case_index(
    complete_beams: dict,
    strategy: str = "random",   # "best", "random", 또는 정수 인덱스
) -> int:
    """
    (1) complete_beams에서 사용할 케이스 하나 선택하고, 그 인덱스를 리턴하는 함수.
    complete_beams["aggregate_scores"]를 기준으로 선택.
    """
    scores = complete_beams.get("aggregate_scores", None)
    if scores is None or len(scores) == 0:
        raise ValueError("complete_beams['aggregate_scores']가 비어 있습니다.")

    if strategy == "best":
        return int(np.argmax(scores))
    elif strategy == "random":
        return random.randint(0, len(scores)-1)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def build_cur_input_ids_from_maxima(
    question: str,
    generator,
    complete_beams: dict,
    case_idx: int,
    maxima_mode: str = "local_max",   # "local_max" 또는 "global_max"
    step_pattern: str = r"## Step",
) -> torch.Tensor:
    """
    (2) 선택된 case_idx에 대해 step_scores에서 maxima를 찾고,
        그 지점까지의 answer를 prefix로 사용해서 cur_input_ids를 만들어 반환.

    반환: cur_input_ids (1, L)  — question + truncated_answer 전체를 encode한 결과
    """
    answers = complete_beams.get("answer", None)
    step_scores_list = complete_beams.get("step_scores", None)

    if answers is None or step_scores_list is None:
        raise ValueError("complete_beams에 'answer' 또는 'step_scores'가 없습니다.")

    answer = answers[case_idx]
    step_scores = torch.tensor(step_scores_list[case_idx], dtype=torch.float32)  # (T,)

    # maxima 위치 찾기
    if maxima_mode == "local_max":
        local_max_idxs = find_local_maxima(step_scores)
        if len(local_max_idxs) > 0:
            anchor_step = local_max_idxs[0]  # 가장 높은 local maxima
        else:
            # local maxima가 없으면 global max로 fallback
            anchor_step = int(torch.argmax(step_scores).item())
    elif maxima_mode == "global_max":
        anchor_step = int(torch.argmax(step_scores).item())
    else:
        raise ValueError(f"Unknown maxima_mode: {maxima_mode}")

    # anchor_step까지의 답변만 남기기
    truncated_answer = _truncate_answer_at_step(
        answer,
        step_index=anchor_step,
        step_pattern=step_pattern,
    )

    # question + truncated_answer를 하나의 프롬프트로 묶어서 encode
    # (원래 프롬프트 포맷에 맞게 "\n\n" 등은 원하는 대로 바꿔도 됨)
    full_prompt = question + "\n" + truncated_answer
    cur_input_ids = generator.encode(full_prompt)  # (1, L)

    return cur_input_ids

def compute_metric(
    metric: str,
    logits: torch.Tensor,           # (T, V)
) -> torch.Tensor:
    probs = F.softmax(logits, dim=-1)  # (T,V)
    # 표준: entropy = -sum p log p; perplexity = exp(entropy)
    if metric in ("top1"):
        metric = probs.topk(1, dim=-1).values.sum(-1)
    else:
        raise KeyError(metric)

    return metric

def find_local_maxima(nll: torch.Tensor):
    T = nll.numel()
    idxs, vals = [], []
    for i in range(1, T-1):
        if (nll[i-1] < nll[i]) and (nll[i] >= nll[i+1]):
            idxs.append(i)
            vals.append(float(nll[i]))
    # 가장 높은 maxima의 index부터 전달하게 됩니다.
    return [i for i, _ in sorted(zip(idxs, vals), key=lambda x: -x[1])]

class GeneticSearch:
    def __init__(
        self,
        method,
        generator: BaseGenerator,
        prm: BasePRM,

        temp_update_rule=None,
        max_trials: int = None,
        score_aggregation: Literal["min", "mean", "last", 'prod'] = "min",
        metric:str = 'top1',
        select_strategy:str = 'random'
    ):
        self.method = method
        self.generator = generator
        self.prm = prm

        self.temp_update_rule = temp_update_rule
        self.max_trials = max_trials
        self.trials = 0

        self.score_aggregation = score_aggregation

        self.return_all_steps = True

        self.init_number_of_beams = 1

        self.trial_to_ids = {}
        self.trial_to_logits = {}
        self.metric = metric
        self.select_strategy = select_strategy

    def compute_step_scores(self, responses: list, prm_state):
        score_tok = getattr(self.prm, "score_token", None)
        responses = [r.replace("\n\n## Step", f"{score_tok}## Step") for r in responses]
        return self.prm(responses, prm_state, return_all_steps=self.return_all_steps)

    def _update_temperature(self):
        if not hasattr(self.generator, "temperature"):
            return None
        # __call__에서 한 번 저장해둔 기준 온도 (없으면 현재 온도 사용)
        base_T = float(getattr(self, "_base_temperature", float(self.generator.temperature)))

        # max_trials 정보가 없거나 1 이하이면 그냥 base_T 유지
        if self.max_trials is None or self.max_trials <= 1:
            self.generator.temperature = base_T
            return base_T

        # self.trials 는 for 루프 안에서:
        #   - _update_temperature() 호출 시점: 0, 1, 2, ... (0-based 인덱스 느낌)
        #   - 최대값은 (max_trials - 2) 정도
        step_idx = max(0, min(self.trials, self.max_trials - 1))
        progress = step_idx / float(self.max_trials - 1)  # 0.0 ~ 1.0

        # ── 더욱 공격적인 쿨링 파라미터 ──
        target_min_ratio = 0.35   # 마지막에는 base_T의 35% 근처까지 떨어뜨리기
        alpha = 2.0               # alpha > 1 이면 초반부터 빠르게 식음 (front-loaded cooling)

        # ratio(progress):
        #   progress = 0   → ratio = 1.0 (base_T)
        #   progress ~ 0.5 → ratio ≈ 0.51 (이미 꽤 낮음)
        #   progress → 1   → ratio = target_min_ratio (0.35)
        ratio = target_min_ratio + (1.0 - target_min_ratio) * ((1.0 - progress) ** alpha)

        T = base_T * ratio

        # 안전 범위 클램프 (너무 극단적이지 않도록)
        # 정확도에 올인하는 세팅이므로 상한은 1.0 정도로 두고,
        # 하한은 0.15까지 허용
        T = max(0.15, min(1.0, T))

        # 실제 generator에 반영
        self.generator.temperature = float(T)

        logger.info(
            f"[TempUpdate-Math100-AggressiveCooling] "
            f"trials={self.trials}/{self.max_trials}, "
            f"base_T={base_T:.3f}, progress={progress:.2f}, "
            f"ratio={ratio:.4f}, T={T:.3f}"
        )

        return T

    def __call__(self, question: str):
        # ─── 온도 초기화 (질문마다 독립적인 초기 T 사용) ───
        if hasattr(self.generator, "temperature"):
            # 첫 호출에서 기준 온도를 저장해두고, 이후엔 그걸로 리셋
            if not hasattr(self, "_base_temperature"):
                self._base_temperature = float(self.generator.temperature)
            self.generator.temperature = float(self._base_temperature)

        # 온도 업데이트에 쓸 히스토리 버퍼 초기화
        self._answers = []
        self._agg_scores = []
        
        self.trial_to_ids = {}
        self.trial_to_logits = {}
        self.trials = 0
        input_ids_question = self.generator.encode(question)
        gen_state_question = self.generator.init_state(input_ids_question)
        prm_state_question = self.prm.init_state(question)

        input_ids_question = input_ids_question.repeat(self.init_number_of_beams, 1)
        gen_state_question = self.generator.inflate_state(gen_state_question, self.init_number_of_beams)
        prm_state_question = self.prm.inflate_state(prm_state_question, self.init_number_of_beams)

        input_len = input_ids_question.shape[1]
        complete_beams = defaultdict(list)
        
        proposal_ids, proposal_logits, gen_state = self.generator(input_ids_question, gen_state_question)
        # print(f'proposal_logits:{proposal_logits.shape}')
        # proposal_metric_seq = compute_metric(self.metric, proposal_logits)
        # peak_ids = find_local_maxima(proposal_metric_seq)
        self.trial_to_ids[self.trials] = proposal_ids
        self.trial_to_logits[self.trials] = proposal_logits
        self.trials += 1

        proposal_response_ids = proposal_ids[:, input_len :]
        proposal_response_text = self.generator.tokenizer.batch_decode(proposal_response_ids)
      
        proposal_scores, proposal_score_logits, prm_state = self.compute_step_scores(proposal_response_text, prm_state_question)

        proposal_agg_scores = aggregate(proposal_scores, self.score_aggregation).item()

        self._answers.append(proposal_response_text[0])
        self._agg_scores.append(proposal_agg_scores)
        
        is_complete = self.generator.is_complete(proposal_ids)
        # if not is_complete[0]:
        #     complete_beams['CaseType'].append('Candidates')
        #     complete_beams['answer'] = []
        #     complete_beams['aggregate_scores'] = []
        #     complete_beams['step_scores'] = []
        #     complete_beams['temp'] = [self.generator.temperature]
        # else:
        complete_beams['CaseType'].append('Candidates')
        complete_beams['answer'].append(proposal_response_text[0])
        complete_beams['aggregate_scores'].append(proposal_agg_scores)
        complete_beams['step_scores'].append(proposal_scores.tolist())
        complete_beams['temp'].append(self.generator.temperature)

        # last_proposal_ids = proposal_ids.clone()
        logger.info(f'[Genetic] Intial {self.trials}/{self.max_trials} : {proposal_agg_scores:.4f}')

        for trial_idx in range(self.max_trials-1):
            selected_idx = select_case_index(complete_beams, strategy=self.select_strategy)
            selected_ids = self.trial_to_ids[selected_idx]
            selected_logits = self.trial_to_logits[selected_idx][0]
            print(f'selected_logits:{selected_logits.shape}')
            proposal_metric_seq = compute_metric(self.metric, selected_logits)
            peak_ids = find_local_maxima(proposal_metric_seq)
            self._update_temperature()

            # 새 후보도 히스토리에 기록 (온도 업데이트에 사용)
            self._answers.append(new_proposal_response_text[0])
            self._agg_scores.append(new_proposal_agg_scores)
            if len(peak_ids) == 0:
                # peak이 없는 상황의 경우 그냥 처음부터 새로 생성
                input_ids_for_proposal = input_ids_question
                new_proposal_ids, new_proposal_logits, new_gen_state = self.generator(input_ids_for_proposal, gen_state_question)
                self.trial_to_ids[self.trials] = new_proposal_ids
                self.trial_to_logits[self.trials] = new_proposal_logits
                self.trials += 1

                new_proposal_respose_ids = new_proposal_ids[:, input_len :]

                new_proposal_response_text = self.generator.tokenizer.batch_decode(new_proposal_respose_ids)
                new_proposal_scores, new_proposal_score_logits, prm_state = self.compute_step_scores(new_proposal_response_text, prm_state_question)
                
                new_proposal_agg_scores = aggregate(new_proposal_scores, self.score_aggregation).item()
                logger.info(f'[Genetic] Generating from question {self.trials}/{self.max_trials} : {new_proposal_agg_scores:.4f}')

                complete_beams['CaseType'].append('Candidates')
                complete_beams['answer'].append(new_proposal_response_text[0])
                complete_beams['aggregate_scores'].append(new_proposal_agg_scores)
                complete_beams['step_scores'].append(new_proposal_scores.tolist())
                complete_beams['temp'].append(self.generator.temperature)                
            else:
                peak_idx = peak_ids[0] # 가장 높은 local maxima를 선택함
                input_ids_for_proposal = selected_ids[:, :input_len+peak_idx]
                new_proposal_ids, new_proposal_logits, new_gen_state = self.generator(input_ids_for_proposal, gen_state_question)
                self.trial_to_ids[self.trials] = new_proposal_ids
                self.trial_to_logits[self.trials] = new_proposal_logits
                self.trials += 1

                new_proposal_respose_ids = new_proposal_ids[:, input_len :]

                new_proposal_response_text = self.generator.tokenizer.batch_decode(new_proposal_respose_ids)
                new_proposal_scores, new_proposal_score_logits, prm_state = self.compute_step_scores(new_proposal_response_text, prm_state_question)
                
                new_proposal_agg_scores = aggregate(new_proposal_scores, self.score_aggregation).item()
                logger.info(f'[Genetic] Generating from {selected_idx}-th sample {self.trials}/{self.max_trials} : {new_proposal_agg_scores:.4f}')

                complete_beams['CaseType'].append('Candidates')
                complete_beams['answer'].append(new_proposal_response_text[0])
                complete_beams['aggregate_scores'].append(new_proposal_agg_scores)
                complete_beams['step_scores'].append(new_proposal_scores.tolist())
                complete_beams['temp'].append(self.generator.temperature)

        return complete_beams
