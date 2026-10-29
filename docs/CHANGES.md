# Portfolio revision

- Added a clean training package and command-line entry point independent of notebook state.
- Corrected preprocessing-before-split leakage; restored five global-mean-filled payload values to missing.
- Introduced chronological development/validation/test partitions, training-only CV and validation-only threshold choice.
- Removed direct outcome fields and selected retrospective metadata from predictors.
- Recomputed all supported results, including a baseline, uncertainty and balanced accuracy.
- Rebuilt the Dash interface with clear historical scope, outcome colors, site filters and a slider covering 15,600 kg launches; omitted unknown payloads rather than plotting imputed masses as real observations.
- Moved course notebooks and old scores/figures to a labeled archive; organized datasets in data/.
- Added pinned direct dependencies, tests, CI configuration, provenance, methodological caveats and licensing scope.
