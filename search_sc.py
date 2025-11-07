from typing import Any, Literal, List

import torch

from interfaces import BaseGenerator, BasePRM

import random
import numpy as np
import os
import re
import logging
import torch.nn.functional as F

from collections import defaultdict

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


class SelfConsistencySearch:
    def __init__(
        self,
        method,
        generator: BaseGenerator,
        prm: BasePRM,

        temp_update_rule=None,
        max_trials: int = None,
        score_aggregation: Literal["min", "mean", "last", 'prod'] = "min",
    ):
        self.method = method
        self.generator = generator
        self.prm = prm

        self.temp_update_rule = temp_update_rule
        self.max_trials = max_trials

        self.score_aggregation = score_aggregation

        self.return_all_steps = False

        self.init_number_of_beams = 1

    def compute_step_scores(self, responses: list, prm_state):
        responses = list(map(lambda x: x.replace('\n\n## Step', f'{self.prm.score_token}## Step'), responses))
        return self.prm(responses, prm_state)

    def _update_temperature(self):
        if self.temp_update_rule is None:
            return None
        else:
            # TODO
            # self.generator.temperature = ...
            raise NotImplementedError()

    def __call__(self, question: str):
        input_ids = self.generator.encode(question)
        gen_state = self.generator.init_state(input_ids)
        prm_state = self.prm.init_state(question)

        input_ids = input_ids.repeat(self.init_number_of_beams, 1)
        gen_state = self.generator.inflate_state(gen_state, self.init_number_of_beams)
        prm_state = self.prm.inflate_state(prm_state, self.init_number_of_beams)

        input_len = input_ids.shape[1]
        complete_beams = defaultdict(list)
        
        proposal_ids, proposal_logits, gen_state = self.generator(input_ids, gen_state)

        proposal_response_ids = proposal_ids[:, input_len :]
        proposal_response_text = self.generator.tokenizer.batch_decode(proposal_response_ids)
      
        proposal_scores, proposal_score_logits, prm_state = self.compute_step_scores(proposal_response_text, prm_state)

        proposal_agg_scores = aggregate(proposal_scores, self.score_aggregation).item()

        is_complete = self.generator.is_complete(proposal_ids)
        if not is_complete[0]:
            complete_beams['CaseType'].append('InitialIncompletion')
            complete_beams['answer'] = []
            complete_beams['aggregate_scores'] = []
            complete_beams['step_scores'] = []
            complete_beams['temp'] = [self.generator.temperature]
        else:
            complete_beams['CaseType'].append('Candidates')
            complete_beams['answer'].append(proposal_response_text[0])
            complete_beams['aggregate_scores'].append(proposal_agg_scores)
            complete_beams['step_scores'].append(proposal_scores.tolist())
            complete_beams['temp'].append(self.generator.temperature)

        trials = 0
        last_proposal_ids = proposal_ids.clone()
        best_score = proposal_agg_scores
        logger.info(f'[SelfConsistency] Intial : {best_score:.4f}')

        for trial_idx in range(self.max_trials):
            self._update_temperature()
            new_proposal_ids, new_proposal_logits, new_gen_state = self.generator(input_ids, gen_state)

            new_proposal_respose_ids = new_proposal_ids[:, input_len :]

            new_proposal_response_text = self.generator.tokenizer.batch_decode(new_proposal_respose_ids)
            new_proposal_scores, new_proposal_score_logits, prm_state = self.compute_step_scores(new_proposal_response_text, prm_state)
            
            new_proposal_agg_scores = aggregate(new_proposal_scores, self.score_aggregation).item()
            logger.info(f'[SelfConsistency] New score {trial_idx+1}/{self.max_trials} : {new_proposal_agg_scores:.4f}')

            complete_beams['CaseType'].append('Candidates')
            complete_beams['answer'].append(new_proposal_response_text[0])
            complete_beams['aggregate_scores'].append(new_proposal_agg_scores)
            complete_beams['step_scores'].append(new_proposal_scores.tolist())
            complete_beams['temp'].append(self.generator.temperature)

        return complete_beams