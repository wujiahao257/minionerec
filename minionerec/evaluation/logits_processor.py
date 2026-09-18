from transformers.generation import LogitsProcessor
from transformers import AutoTokenizer
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union
import math
import numpy as np
import torch
import warnings

from transformers.utils import add_start_docstrings

LOGITS_PROCESSOR_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)
        scores (`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`):
            Prediction scores of a language modeling head. These can be logits for each vocabulary when not using beam
            search or log softmax for each vocabulary token when using beam search

    Return:
        `torch.FloatTensor` of shape `(batch_size, config.vocab_size)`: The processed prediction scores.

"""

class ConstrainedLogitsProcessor(LogitsProcessor):

    """按目录 SID 的合法前缀屏蔽生成分数；内部 count 状态只适用于一次生成。

    Args:
        prefix_allowed_tokens_fn (Callable[[int, list[int]], list[int]]): 接收 batch 编号和前缀 token ID，返回合法后继 token ID 列表的回调。
        num_beams (int): beam 搜索宽度；相应生成配置通常返回同样数量的候选序列。
        base_model (str | None): 模型 checkpoint 目录或模型标识；需要配套的模型配置、权重和 tokenizer。
        eos_token_id (int | None): 结束 token 的词表编号；无合法后继时用于强制结束序列。
    """
    def __init__(
        self,
        prefix_allowed_tokens_fn: Callable[[int, torch.Tensor], List[int]],
        num_beams: int,
        base_model: str = None,
        eos_token_id: int = None
    ):
        """初始化 ConstrainedLogitsProcessor：按目录 SID 的合法前缀屏蔽生成分数；内部 count 状态只适用于一次生成。

        Args:
            self (ConstrainedLogitsProcessor): 当前实例，由 Python 在调用实例方法时自动传入。
            prefix_allowed_tokens_fn (Callable[[int, list[int]], list[int]]): 接收 batch 编号和前缀 token ID，返回合法后继 token ID 列表的回调。
            num_beams (int): beam 搜索宽度；相应生成配置通常返回同样数量的候选序列。
            base_model (str | None): 模型 checkpoint 目录或模型标识；需要配套的模型配置、权重和 tokenizer。
            eos_token_id (int | None): 结束 token 的词表编号；无合法后继时用于强制结束序列。

        Returns:
            None: 完成实例初始化。
        """
        self._prefix_allowed_tokens_fn = prefix_allowed_tokens_fn
        self._num_beams = num_beams
        self.count=0
        self.base_model = base_model
        self.eos_token_id = eos_token_id
        if self.base_model.lower().find("gpt2") > -1:
            self.prefix_index = 4
        else:
            self.prefix_index = 3

    
    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """将不符合目录前缀的 token 分数设为负无穷，并推进当前生成步计数。

        Args:
            self (ConstrainedLogitsProcessor): 当前实例，由 Python 在调用实例方法时自动传入。
            input_ids (torch.LongTensor): 已生成的完整序列，shape [B*num_beams,L]；从尾部提取当前前缀。
            scores (torch.Tensor): 当前生成步的分数，shape [B*num_beams,V]，非法后继会被设为负无穷。

        Returns:
            torch.Tensor: shape [B*num_beams,V] 的受约束 log 分数。
        """
        scores = torch.nn.functional.log_softmax(scores, dim=-1)
        mask = torch.full_like(scores, float('-inf'))
            
        for batch_id, beam_sent in enumerate(input_ids.view(-1, self._num_beams, input_ids.shape[-1])):
            for beam_id, sent in enumerate(beam_sent):
                if self.count == 0:
                    hash_key = sent[-self.prefix_index:]
                else:
                    hash_key=sent[-self.count:]
                hash_key = hash_key.tolist()
                prefix_allowed_tokens = self._prefix_allowed_tokens_fn(batch_id, hash_key)

                if len(prefix_allowed_tokens) == 0:
                    warnings.warn(
                        f"No valid tokens found for hash_key {hash_key} at step {self.count}. "
                        f"This indicates the model generated an unexpected token. "
                    )
                    # Force EOS token to end invalid sequence
                    if self.eos_token_id is not None:
                        mask[batch_id * self._num_beams + beam_id, self.eos_token_id] = 0
                    continue 
                
                mask[batch_id * self._num_beams + beam_id, prefix_allowed_tokens] = 0

        self.count += 1

        scores = scores + mask
        return scores