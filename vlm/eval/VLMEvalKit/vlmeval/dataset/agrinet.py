from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from vlmeval.smp import dump, load

from .image_base import ImageBaseDataset


class AgriNetVLM(ImageBaseDataset):
    TYPE = 'VQA'
    MODALITY = 'IMAGE'
    DATASET_URL = {'AgriNetVLM': ''}

    def __init__(self, dataset='AgriNetVLM', data_file=None, repo_root='.', **kwargs):
        self.data_file = str(data_file or '')
        self.repo_root = Path(repo_root).resolve()
        if not self.data_file:
            raise ValueError('AgriNetVLM requires data_file pointing to a local TSV.')
        super().__init__(dataset=dataset, skip_noimg=False)
        self.force_use_dataset_prompt = True

    @classmethod
    def supported_datasets(cls):
        return ['AgriNetVLM']

    def load_data(self, dataset):
        path = Path(self.data_file)
        if not path.exists():
            raise FileNotFoundError(f'AgriNet TSV not found: {path}')
        data = load(str(path))
        if 'index' not in data:
            data['index'] = [str(i) for i in range(len(data))]
        if 'image_path' not in data:
            raise ValueError(f'{path} must contain image_path')
        if 'question' not in data:
            raise ValueError(f'{path} must contain question')
        return data.fillna('')

    def dump_image(self, line):
        image_path = Path(str(line['image_path']))
        if not image_path.is_absolute():
            image_path = self.repo_root / image_path
        if not image_path.exists():
            raise FileNotFoundError(f'image not found: {image_path}')
        return [str(image_path)]

    def evaluate(self, eval_file, **judge_kwargs):
        from vlm.eval.tools.normalize_answers import score_row, summarize

        data = load(str(eval_file))
        rows = data.fillna('').to_dict('records') if isinstance(data, pd.DataFrame) else data
        scored = [score_row(row) for row in rows]
        metrics = summarize(scored)

        eval_path = Path(eval_file)
        scored_jsonl = eval_path.with_name(eval_path.stem + '_scored.jsonl')
        metrics_json = eval_path.with_name(eval_path.stem + '_metrics.json')
        scored_csv = eval_path.with_name(eval_path.stem + '_scored.csv')

        with scored_jsonl.open('w', encoding='utf-8') as f:
            for row in scored:
                f.write(json.dumps(row, ensure_ascii=False) + '\n')
        metrics_json.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        dump(pd.DataFrame(scored), str(scored_csv))

        return metrics
