# SpaceX Falcon 9 Landing Prediction Capstone

This repository contains the completed notebooks, datasets, visual evidence, and dashboard application for the IBM Data Science Professional Certificate capstone.

## Project question

Can public launch records help explain and predict whether the Falcon 9 first stage will land successfully?

## Main results

- The analysis covers 90 Falcon 9 launches from 2010 through 2020.
- 60 launches landed successfully, giving an overall success rate of 66.7%.
- Landing performance improved substantially after 2013.
- KSC LC 39A achieved a 77.3% success rate in this sample, while CCSFS SLC 40 handled the most launches.
- SVM and K-nearest neighbors both reached 83.3% test accuracy. SVM had the stronger cross-validation result and was selected as the final model.

## Repository structure

```text
assets/       Charts and presentation evidence
data/         Cleaned, encoded, geographic, and model-result files
notebooks/    Completed capstone notebooks
spacex_dash_app.py
```

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python spacex_dash_app.py
```

Then open the local Dash URL shown in the terminal.

## Notes on reproducibility

The data-collection notebook includes the IBM course snapshot because the historical SpaceX community API endpoint used in the original lab is no longer consistently available. The supplied snapshot preserves the course dataset and makes the downstream analysis reproducible.
