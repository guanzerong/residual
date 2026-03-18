# Third-Party Backbones

This directory is reserved for local source checkouts that should not be
committed into the main repository.

Expected layout:

```text
third_party/
  dinov2/
  vggt/
```

Use the helper scripts from the repository root:

```bash
conda activate residual1
bash scripts/setup_consisvla_backbones.sh
scripts/run_with_consisvla_backbones.sh python -c "from dinov2.hub.backbones import dinov2_vitl14_reg; from vggt.models.vggt import VGGT; print('ok')"
```

Design choices:

- The repos are kept as source trees under `third_party/` instead of editable
  installs so the main environment does not inherit DINOv2's pinned `torch`
  and `xformers` requirements.
- The setup script only installs a small set of missing utility packages needed
  by the source trees. It does not downgrade `torch` or `torchvision`.
- Use `scripts/run_with_consisvla_backbones.sh ...` for commands that need both
  `dinov2` and `vggt` on `PYTHONPATH`.
