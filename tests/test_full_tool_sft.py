import json
from pathlib import Path

import pytest

from agrinet.data.full_tool_sft import freeze
from agrinet.vlm.full_tool_trial import checkpoint_at


def test_freeze_refuses_existing_destination(tmp_path):
    with pytest.raises(ValueError, match='existing freeze'):
        freeze({'inputs': {'source': 'absent'}, 'outputs': {'artifact': str(tmp_path)}})


def test_checkpoint_uses_epoch_not_assumed_step(tmp_path):
    p = tmp_path / 'v0' / 'checkpoint-87'
    p.mkdir(parents=True)
    (p / 'trainer_state.json').write_text(json.dumps({'epoch': 3.0}))
    assert checkpoint_at(tmp_path, 3) == p
    with pytest.raises(ValueError):
        checkpoint_at(tmp_path, 6)
    duplicate = tmp_path / 'v1' / 'checkpoint-90'
    duplicate.mkdir(parents=True)
    (duplicate / 'trainer_state.json').write_text(json.dumps({'epoch': 3.0}))
    with pytest.raises(ValueError):
        checkpoint_at(tmp_path, 3)
