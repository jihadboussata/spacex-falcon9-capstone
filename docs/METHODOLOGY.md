# Evaluation protocol and caveats

The canonical source is data/dataset_part_2.csv. Sort by Date and FlightNumber. The 60/20/20 partition is fixed by order, not stratified or reshuffled to improve the score. Split boundaries must lie between dates. Training-only three-fold TimeSeriesSplit uses a one-launch gap. All numeric imputation/scaling and categorical imputation/encoding are inside a Pipeline.

Features are PayloadMass, Flights, Block, GridFins, Reused, Legs, Orbit and LaunchSite. Outcome and Class are prohibited predictors. LandingPad, Serial and ReusedCount are excluded to avoid outcome proxies, identifier memorization and snapshot reuse metadata. Even retained metadata is retrospective, so this is a disciplined benchmark rather than a fully audited point-in-time launch forecast.

The original payload collection filled missing entries with the same global mean, 6123.547647058824 kg. Five entries matching that documented value (absolute tolerance 1e-7) are restored to missing for this fixed snapshot. This repair is snapshot-specific; it must not be applied blindly to another dataset. Training-fold median imputation then replaces missing payloads.

Candidates: logistic regression C {0.1,1,10}; SVM C {0.1,1,10} with linear/RBF kernels; decision trees depth {2,3}, leaf size {2,4}; KNN neighbors {3,5,7}. CV chooses parameters by average precision. Validation average precision selects the model, and validation F1 determines the threshold. The estimator remains training-fitted to keep the threshold tied to its validation probabilities.

The test is only 18 launches with three negatives. Accuracy, balanced accuracy, ranking measures, the confusion matrix and exact binomial accuracy interval are all reported. The interval assumes independent Bernoulli outcomes; shared hardware, changing mission mix and chronological dependence limit its meaning. No claim of statistical superiority is made.

Historical no-attempt outcomes are grouped with failures by the course label. A launch configured without recovery hardware is easier to classify, but this does not answer the narrower question of recovery success given an attempt. That would require a newly defined target and a larger, point-in-time dataset.

Reference: [scikit-learn guidance on leakage and Pipelines](https://scikit-learn.org/stable/common_pitfalls.html).
