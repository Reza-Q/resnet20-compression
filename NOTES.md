# Notes on the refactor

This file documents every behavior-affecting difference between this repository and the
original notebook (`ResNet20.ipynb`), split into **bugs that were fixed** and **design
changes that don't change numerical behavior**. Both bugs below were confirmed
empirically (not just by reading the code) before being fixed; the scripts used are
described inline so you can reproduce the diagnosis yourself if you want to double-check
the original notebook's behavior.

## Bug 1: the notebook's "INT4" QAT model was mostly still INT8

**Where:** the QAT cells (Scenario C / Scenario D), specifically the notebook's
`prepare_qat_model(model, qconfig_type)` helper.

**What happened:** the notebook prepared INT8 and INT4 QAT by calling
`prepare_qat_model` twice **on the same in-memory model object** — once with
`qconfig_type='fbgemm'`, then again with `qconfig_type='int4'`. That helper builds a
thin wrapper by assigning the model's existing submodules directly
(`self.conv1 = base_model.conv1`, etc. — a reference, not a copy) and then calls
`torch.quantization.prepare_qat(qat_model, inplace=True)`, which walks the module tree
and gives every submodule a `.qconfig` attribute plus fake-quantize hooks.

PyTorch's internal qconfig propagation (`propagate_qconfig_`) keeps a module's
**existing** `.qconfig` if it already has one, rather than overwriting it with its
parent's. Because the second (`int4`) call reused the exact same submodule objects the
first (`fbgemm`) call had already mutated in place, every internal conv and activation
kept its INT8 range from the first call. Only the freshly constructed top-level input
`QuantStub` — a brand-new object with no prior `.qconfig` — actually picked up the INT4
range.

**In other words:** the notebook's "INT4" model was, internally, still an INT8 model
with only its input stub quantized to 4 bits. The reported INT4 accuracy/compression
numbers in the original notebook do not reflect genuine INT4 quantization of the
network's convolutions.

**How this was confirmed:** the notebook's own `prepare_qat_model` and `ResNetQAT` code
was executed as-is (via `exec`, on an isolated model instance) with instrumentation that
printed `quant_min`/`quant_max` of `layer2[0].conv1.weight_fake_quant`,
`layer2[0].conv1.activation_post_process` and `quant.activation_post_process` right
after the INT8 call and again right after the INT4 call on the same model. The internal
conv's range stayed `[-128, 127]` / `[0, 127]` (INT8) after the "INT4" call, while only
the input stub changed to `[0, 15]`.

**The fix:** [`quantization.py`](resnet_compression/quantization.py)'s
`prepare_qat_model` deep-copies the floating-point model **before** wrapping it, so
preparing one bit-width can never leave fake-quantizer state behind for another call.
`tests/test_quantization.py::test_int8_and_int4_preparations_are_independent` is a
regression test for this: it asserts the two wrappers share no submodules and that
INT4's internal conv actually has `quant_min=-8, quant_max=7` (not `-128, 127`).

**Practical implication if you're comparing against the original notebook's numbers:**
expect this codebase's INT4 accuracy to be *lower* and its compression ratio to be
*higher* than the original notebook reported, because a genuinely 4-bit network is a
more aggressive approximation than an 8-bit network with a 4-bit input stub.

## Bug 2: Scenario B wrote its checkpoints into Scenario A's directory

**Where:** the Scenario B cell's intermediate-checkpoint save path.

**What happened:** the checkpoint interval logic inside Scenario B's loop built its
path with `os.path.join(CONFIG['RESULTS_A_DIR'], f'checkpoint_{...}_pruned.pth')` — a
copy-pasted constant from Scenario A's cell that was never updated to
`RESULTS_B_DIR`. Scenario B's final model and results JSON correctly used
`RESULTS_B_DIR`; only the periodic intermediate checkpoints were affected. Since
Scenario A runs first and uses the same file-naming pattern for the same set of
pruning rates and checkpoint percentages, Scenario B's intermediate checkpoints could
silently land on top of Scenario A's own intermediate checkpoints of the same nominal
rate/percentage.

**Impact:** this does not affect the notebook's final headline numbers (those come from
each scenario's own `final_model_*.pth` and `results` dict, which used the correct
directories), but it does mean Scenario A's saved *intermediate* checkpoints could not
be trusted after Scenario B ran — inspecting, say, "Scenario A pruned to 4%" from disk
after a full run could actually load a Scenario B (no-KD) model.

**How this was confirmed:** `grep -n "RESULTS_A_DIR\|RESULTS_B_DIR" cell_09.py`
(the Scenario B cell) showed `RESULTS_B_DIR` used for the results-JSON path and the
final-model path, but `RESULTS_A_DIR` used for the mid-loop checkpoint path — the only
one of the three that was wrong.

**The fix:** [`pruning_pipeline.py`](resnet_compression/pruning_pipeline.py) implements
Scenarios A and B as a **single** parameterized function (`run_pruning_scenario`, with a
`use_kd` flag) that always saves to `cfg.scenario_dir(scenario)`. Because there is only
one code path for both scenarios, a scenario-specific path constant can no longer be
copy-pasted incorrectly — the smoke test in this repo's CI confirms both scenarios'
checkpoints land in their own, separate directories.

## Design changes that do not affect numerical results

These are refactors made for maintainability; each was checked to produce identical
tensors/outputs to the original where the original could be run as a fair comparison.

- **One `BasicBlock`/`ResNet` class instead of three near-duplicates.** The notebook
  defined its residual block three times: once in the baseline/teacher cell (plain
  `out += self.shortcut(x)`), once in the pruning cell (same, under the name
  `PrunedBasicBlock`), and once again in the QAT cell (a third copy, this time using
  `FloatFunctional().add(...)` instead of `+=`, which is what makes a block
  quantization-ready). [`models.py`](resnet_compression/models.py) defines the block
  once, using `FloatFunctional` from the start, and reuses it for the baseline,
  teachers, pruned models and QAT alike. `FloatFunctional.add(a, b)` is numerically
  identical to `a + b` in floating point (it only changes behavior once a model is
  actually converted to a quantized `nn.Module`), which was verified directly: pruning
  10 rates x 6 iterations each with this unified class and comparing every resulting
  tensor and forward-pass output against the original notebook's plain-`+=` pruning
  code produced bit-identical results in every case
  (see `tests/test_pruning.py` for the invariants this left behind, and the module
  docstring of `pruning.py` for the algorithm itself).
- **Reconstructing a pruned model from its checkpoint no longer needs a hand-written
  architecture-inference cell.** The QAT cells re-derived a pruned model's per-stage
  width by reading `state_dict['layer1.0.conv1.weight'].shape[0]` (block 0 only) and
  assuming every block in that stage shared that width. That assumption is actually
  true for this pruning algorithm (proportional per-conv pruning started from a
  uniform-width network keeps every block in a stage at equal width, inductively, at
  every iteration), but it's an assumption the original code never checked.
  [`models.py`](resnet_compression/models.py)'s `infer_block_channels` instead reads
  each block's own `conv2` output width independently, so it would still build the
  correct architecture even if that invariant were ever violated (e.g. by a future,
  non-uniform pruning strategy), and there is no longer a ~60-line block of
  reconstruction code duplicated across the two QAT scenario cells.
- **`get_quantized_model_size_mb` writes to a temp file instead of a fixed relative
  path.** The notebook's version wrote to a hardcoded `temp_quantized_model.pth` in the
  working directory; harmless for a single sequential run, but not safe if this were
  ever called concurrently. `utils.py`'s version uses `tempfile.TemporaryDirectory()`.
- **Pruning a block can change 1x1 shortcut convolutions in ways not captured by "number
  of filters pruned".** Not a bug, but a subtlety worth knowing if you instrument the
  pipeline yourself: `prune_filters`'s returned pruned-filter count only tallies the two
  3x3 convolutions of each residual block. Keeping the residual add shape-consistent can
  also *shrink* an existing 1x1 shortcut/downsample conv (when a stage's width changes)
  or *create a new one* (when a stage's first block, previously requiring no
  projection, ends up narrower than its input). Both change the network's total filter
  and parameter count independently of the "filters pruned" figure. See the docstring
  and tests in `pruning.py` / `test_pruning.py` for the exact behavior.

## Non-issues, for completeness

- The original notebook's introductory markdown cell says "8 pruning rates" while
  listing (and `CONFIG['PRUNE_RATES']` using) ten values, 1% through 10%. This is a
  documentation miscount in a markdown cell, not a code bug — `default_pruning_rates()`
  in this repo uses all ten, matching what the original code actually ran.
- No Persian (or other non-ASCII-beyond-symbols) text was found anywhere in the original
  notebook's code or markdown cells; the only non-ASCII characters were decorative
  symbols in `print()` calls (checkmarks, warning signs, arrows, etc.), which this
  refactor replaces with the standard `logging` module throughout.
