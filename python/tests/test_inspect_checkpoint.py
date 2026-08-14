import torch
import pytest

import evaluation.inspect_checkpoint as ic


def _stage3_ckpt(tmp_path, **overrides):
    ckpt = {
        "epoch": 342, "val_loss": 0.4, "val_loss_ema": 0.437,
        "ae_checkpoint": "checkpoints/stage2/128x128-stage2.pt",
        "f_theta_state_dict": {"w": torch.zeros(3)},
        "config": {"latent_channels": 8, "hidden_dim": 256, "alpha": 0.5,
                    "max_substeps": 512, "z1_resync": True},
        "data_config": {"max_dt": 2500, "z0_noise_scale": 0.15},
        "test_dirs": ["d1", "d2"],
    }
    ckpt.update(overrides)
    p = tmp_path / "ck.pt"
    torch.save(ckpt, p)
    return p


def test_prints_the_saved_training_config(tmp_path, capsys):
    ic.inspect_checkpoint(_stage3_ckpt(tmp_path))
    out = capsys.readouterr().out
    assert "z0_noise_scale" in out and "0.15" in out
    assert "alpha" in out and "0.5" in out
    assert "epoch" in out and "342" in out


def test_flags_meaningful_but_unsaved_fields(tmp_path, capsys):
    """
    The trap that lost a run's provenance: lr and use_dt_decade_weights are
    used at training time but never written to the checkpoint. Their absence
    must be REPORTED -- otherwise the tool silently omits exactly the fields
    someone is checking for and the reader assumes they were at default.
    """
    ic.inspect_checkpoint(_stage3_ckpt(tmp_path))
    out = capsys.readouterr().out
    assert "NOT in this checkpoint" in out
    assert "use_dt_decade_weights" in out
    assert "lr" in out
    assert ".log" in out, "does not say where to recover the unsaved fields"


def test_does_not_flag_an_unsaved_field_that_IS_present(tmp_path, capsys):
    """If a future checkpoint does start saving lr, it must not appear in the
    'not saved' list -- the flag is about actual absence, not a static list."""
    ck = _stage3_ckpt(tmp_path)
    d = torch.load(ck, weights_only=True)
    d["lds_config"] = {"lr": 1e-6}
    torch.save(d, ck)
    ic.inspect_checkpoint(ck)
    out = capsys.readouterr().out
    not_saved_section = out.split("NOT in this checkpoint")[-1] if "NOT in" in out else ""
    assert "lr " not in not_saved_section.replace("lr:", ""), (
        "lr was present in lds_config but still listed as unsaved"
    )


def test_key_mode_prints_only_the_value(tmp_path, capsys):
    import sys
    ck = _stage3_ckpt(tmp_path)
    sys.argv = ["x", str(ck), "--key", "z0_noise_scale"]
    ic.main()
    assert capsys.readouterr().out.strip() == "0.15"


def test_key_mode_searches_nested_config_blocks(tmp_path, capsys):
    """--key must find a field wherever it lives -- alpha is in `config`,
    z0_noise_scale in `data_config`; the caller shouldn't need to know which."""
    import sys
    ck = _stage3_ckpt(tmp_path)
    sys.argv = ["x", str(ck), "--key", "alpha"]
    ic.main()
    assert capsys.readouterr().out.strip() == "0.5"


def test_key_mode_on_a_known_unsaved_field_explains_where_to_look(tmp_path):
    import sys
    ck = _stage3_ckpt(tmp_path)
    sys.argv = ["x", str(ck), "--key", "use_dt_decade_weights"]
    with pytest.raises(SystemExit) as e:
        ic.main()
    assert ".log" in str(e.value), (
        "a known-unsaved field should point at the .log, not just say 'not found'"
    )


def test_key_mode_missing_field_exits_nonzero(tmp_path):
    import sys
    ck = _stage3_ckpt(tmp_path)
    sys.argv = ["x", str(ck), "--key", "no_such_field"]
    with pytest.raises(SystemExit):
        ic.main()


def test_never_loads_weight_tensors(tmp_path, monkeypatch):
    """The tool must be safe on any .pt: weights_only=True, and it must not
    build a model. Guard by making torch.load reject a non-weights_only call."""
    ck = _stage3_ckpt(tmp_path)
    real_load = torch.load

    def guarded(*args, **kwargs):
        assert kwargs.get("weights_only") is True, (
            "inspect_checkpoint must load with weights_only=True"
        )
        return real_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", guarded)
    ic.inspect_checkpoint(ck)
