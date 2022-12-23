# Copyright (C) Xuechen Li
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""Isolated test for inference of reward model."""

import fire
import torch


def oai_test(fp16_activations=True):
    # pipenv run python iso_test.py --task "oai_test"
    # install packages ftfy, blobfile

    from summarize_from_feedback.reward_model import RewardModel
    from summarize_from_feedback.query_response_model import ModelSpec, RunParams
    from summarize_from_feedback.tasks import TaskHParams, TaskQueryHParams, TaskResponseHParams
    from summarize_from_feedback.model_layout import ModelLayout

    # This downloads and caches to /tmp/bf-dir-cache/. Not great on nlp-cluster.
    # info.json by default stored at
    # /tmp/bf-dir-cache/az/openaipublic/summarize-from-feedback/models/rm4/info.json
    reward_model_spec = ModelSpec(
        device='cuda',
        load_path='https://openaipublic.blob.core.windows.net/summarize-from-feedback/models/rm4',
        use_cache=True,
        short_name='rm4',
        init_heads=None,
        map_heads={},
        run_params=RunParams(
            fp16_embedding_weights=False,
            fp16_conv_weights=False,
            attn_dropout=0.0,
            resid_dropout=0.0,
            emb_dropout=0.0,
            n_shards=1
        )
    )
    print('model_spec okay')

    layout = ModelLayout.standard(
        n_shards=1,
        total_gpus=1,
        my_rank=0,
    )
    print('layout okay')

    task_hparams = TaskHParams(
        query=TaskQueryHParams(
            length=512,
            dataset='tldr_3_filtered',
            format_str='SUBREDDIT: r/{subreddit}\n\nTITLE: {title}\n\nPOST: {post}\n\nTL;DR:',
            truncate_field='post',
            truncate_text='\n',
            padding=None,
            pad_side='left'
        ),
        response=TaskResponseHParams(
            ref_format_str=' {reference}', length=48, truncate_token=50256
        ),
    )
    print('task_hparams okay')

    reward_model = RewardModel(task_hparams=task_hparams, spec=reward_model_spec, layout=layout)
    print('reward_model okay')

    query_tokens = torch.ones(512, dtype=torch.long)  # query length.
    response_tokens = torch.ones(1, 48, dtype=torch.long)  # response length. (num_responses, seq_len).
    act_dtype = torch.float16 if fp16_activations else torch.float32
    results = reward_model.reward(
        query_tokens=query_tokens.unsqueeze(0),
        response_tokens=response_tokens.unsqueeze(0),
        act_dtype=act_dtype,
    )
    print(results)


def tok_test():
    import transformers
    tok = transformers.AutoTokenizer.from_pretrained('gpt2')
    l1 = tok("this is good\n\nTL;DR:this is how you should")
    l2 = tok("this is good\n\nTL;DR: this is how you should")
    l3 = tok("this is good\n\nTL;DR:")
    l4 = tok("this is how you should")
    print(l1)
    print(l2)
    print(l3, l4)
    print(tok("S"))


def main(task="oai_test", **kwargs):
    globals()[task](**kwargs)


if __name__ == "__main__":
    fire.Fire(main)
