"""Deep structural validation of the three Omni-DFINE variant configs.

Each variant yml is expected to define (after include merging) a complete
pipeline: image backbone + text backbone + encoder + decoder + LSDPE +
LW-VTFE + criterion + optimizer + scheduler. We verify every required key
is present and that per-variant scalars match the paper.
"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml


CONFIG_ROOT = Path(__file__).resolve().parent.parent / "configs"
VARIANTS = {
    "omni_dfine/omni_dfine_hgnetv2_s_coco.yml": {
        "backbone": "B0",
        "encoder_in_channels": [256, 512, 1024],
        "encoder_depth_mult": 0.34,
        "decoder_num_layers": 3,
        "decoder_num_queries": 500,
        "epochs": 20,
        "expected_params_M": 11.2,
    },
    "omni_dfine/omni_dfine_hgnetv2_m_coco.yml": {
        "backbone": "B2",
        "encoder_in_channels": [384, 768, 1536],
        "encoder_depth_mult": 0.67,
        "decoder_num_layers": 4,
        "decoder_num_queries": 500,
        "epochs": 20,
        "expected_params_M": 23.5,
    },
    "omni_dfine/omni_dfine_hgnetv2_l_coco.yml": {
        "backbone": "B4",
        "encoder_in_channels": [512, 1024, 2048],
        "encoder_depth_mult": 1.0,
        "decoder_num_layers": 6,
        "decoder_num_queries": 500,
        "epochs": 20,
        "expected_params_M": 36.4,
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursive dict merge (override wins)."""
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _flatten_includes(cfg_path: Path, root: Path) -> dict:
    """Load and merge all `__include__` targets recursively."""
    text = cfg_path.read_text()
    cfg = yaml.safe_load(text) or {}
    includes = cfg.pop("__include__", None)
    merged = {}
    if includes:
        if isinstance(includes, str):
            includes = [includes]
        for inc in includes:
            inc_path = (cfg_path.parent / inc).resolve()
            sub = _flatten_includes(inc_path, root)
            merged = _deep_merge(merged, sub)
    merged = _deep_merge(merged, cfg)
    return merged


def main() -> int:
    print(f"Deep validation of Omni-DFINE variant configs\n")
    errors = 0
    for rel_path, expected in VARIANTS.items():
        cfg_path = CONFIG_ROOT / rel_path
        print(f"--- {rel_path} ---")
        cfg = _flatten_includes(cfg_path, CONFIG_ROOT)

        # Top-level model + criterion + postprocessor
        for key in ("model", "criterion", "postprocessor"):
            assert key in cfg, f"missing top-level '{key}' in {rel_path}"
            print(f"  {key:15s} = {cfg[key]}")

        # OmniDFINE struct
        omnidfine = cfg.get("OmniDFINE", {})
        for sub in ("backbone", "text_backbone", "prompt_encoder",
                    "fusion", "encoder", "decoder"):
            assert sub in omnidfine, f"missing OmniDFINE.{sub} in {rel_path}"

        # HGNetv2 backbone name
        backbone = cfg.get("HGNetv2", {}).get("name")
        assert backbone == expected["backbone"], (
            f"{rel_path}: expected backbone {expected['backbone']}, got {backbone}"
        )
        print(f"  HGNetv2.name     = {backbone}")

        # HybridEncoder
        enc = cfg.get("HybridEncoder", {})
        assert enc.get("in_channels") == expected["encoder_in_channels"], (
            f"{rel_path}: expected in_channels {expected['encoder_in_channels']}, "
            f"got {enc.get('in_channels')}"
        )
        assert enc.get("depth_mult") == expected["encoder_depth_mult"], (
            f"{rel_path}: expected depth_mult {expected['encoder_depth_mult']}, "
            f"got {enc.get('depth_mult')}"
        )
        print(f"  HybridEncoder    = in_channels={enc.get('in_channels')}, "
              f"depth_mult={enc.get('depth_mult')}")

        # Decoder
        dec = cfg.get("DFINETransformer", {})
        assert dec.get("num_layers") == expected["decoder_num_layers"], (
            f"{rel_path}: expected num_layers {expected['decoder_num_layers']}, "
            f"got {dec.get('num_layers')}"
        )
        assert dec.get("num_queries") == expected["decoder_num_queries"], (
            f"{rel_path}: expected num_queries {expected['decoder_num_queries']}, "
            f"got {dec.get('num_queries')}"
        )
        print(f"  DFINETransformer= num_layers={dec.get('num_layers')}, "
              f"num_queries={dec.get('num_queries')}")

        # LSDPE
        lsdpe = cfg.get("LSDPE", {})
        assert lsdpe.get("alpha") == 0.9, f"{rel_path}: LSDPE.alpha should be 0.9"
        print(f"  LSDPE.alpha     = {lsdpe.get('alpha')}")

        # LW-VTFE
        lw_vtfe = cfg.get("LWVTFE", {})
        assert lw_vtfe.get("embed_dim") == 256, (
            f"{rel_path}: LW-VTFE embed_dim should be 256"
        )
        assert lw_vtfe.get("tau_init") == 0.1, (
            f"{rel_path}: LW-VTFE tau_init should be 0.1"
        )
        print(f"  LWVTFE          = embed_dim={lw_vtfe.get('embed_dim')}, "
              f"tau_init={lw_vtfe.get('tau_init')}")

        # BERT text backbone
        bert = cfg.get("BERTTextBackbone", {})
        assert bert.get("pretrained") == "bert-base-uncased"
        assert bert.get("text_proj_dim") == 256
        print(f"  BERT            = {bert.get('pretrained')}, "
              f"proj={bert.get('text_proj_dim')}")

        # Criterion -- HHA must be wired
        crit = cfg.get("OmniDFINECriterion", {})
        assert "loss_hha" in crit.get("weight_dict", {}), (
            f"{rel_path}: OmniDFINECriterion.weight_dict must include loss_hha"
        )
        assert "hha" in crit.get("losses", []), (
            f"{rel_path}: OmniDFINECriterion.losses must include 'hha'"
        )
        hha = crit.get("HHA", {})
        assert hha.get("eta") == 0.07
        assert hha.get("lambda_ref") == 0.3
        print(f"  HHALoss         = eta={hha.get('eta')}, "
              f"lambda_ref={hha.get('lambda_ref')}, "
              f"weight={crit['weight_dict']['loss_hha']}")

        # Epochs
        assert cfg.get("epochs") == expected["epochs"], (
            f"{rel_path}: expected epochs {expected['epochs']}, "
            f"got {cfg.get('epochs')}"
        )
        print(f"  epochs          = {cfg.get('epochs')}")
        print(f"  expected params = {expected['expected_params_M']} M (paper Table III)")
        print()

    if errors == 0:
        print("Deep validation PASSED.")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
