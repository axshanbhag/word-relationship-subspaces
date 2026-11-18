# Beyond Shared Directions

AICompS poster on whether word-relationship subspaces transfer to held-out pairs. The MATLAB experiment compares observed word-pair offsets with shuffled controls and evaluates subspaces on held-out word pairs.

## Run the experiment

Use the original `glove_100k.txt` input from the study. Its precise release provenance is unverified. The repository includes a lossless gzip copy. From the repository root, unpack it with `gzip -dk glove_100k.txt.gz`. Then, from this directory in MATLAB, run:

```matlab
resultsFolder = run_poster_controls('original-experiment-settings.mat', 'glove_100k.txt', 'Original study embedding file');
```

The original settings contain the word-pair lists and vocabulary cap. Check the new run's observed summaries and coverage against the archived CSVs before interpreting any differences.

## Archived outputs

`results/` contains observed metrics, permutation draws and summaries, held-out comparisons, fold assignments and run settings. The MATLAB source was copied from the saved run directory. The final presentation is in `poster.pdf`.
