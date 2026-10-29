import unittest
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from spacex.train import FEATURES, LEGACY_PAYLOAD_FILL, pipeline, prepare


class TrainingTests(unittest.TestCase):
    def test_outcomes_and_snapshot_reuse_are_not_predictors(self):
        for forbidden in ["Class", "Outcome", "ReusedCount", "Serial", "LandingPad", "Date"]:
            self.assertNotIn(forbidden, FEATURES)

    def test_legacy_global_mean_is_restored_to_missing(self):
        frame = pd.DataFrame({"Date": ["2020-01-01", "2019-01-01"], "FlightNumber": [2, 1],
            "PayloadMass": [LEGACY_PAYLOAD_FILL, 525.]})
        clean = prepare(frame)
        self.assertEqual(clean.FlightNumber.tolist(), [1, 2])
        self.assertTrue(np.isnan(clean.PayloadMass.iloc[1]))

    def test_training_imputation_and_unseen_site(self):
        frame = pd.DataFrame({"PayloadMass": [100., 300., np.nan, 200.], "Flights": [1, 1, 2, 3],
            "Block": [1., 2., 3., 4.], "GridFins": [False, True, True, True], "Reused": [False, False, True, True],
            "Legs": [False, True, True, True], "Orbit": ["LEO", "GTO", "LEO", "GTO"], "LaunchSite": ["A", "B", "A", "B"]})
        model = pipeline(LogisticRegression()).fit(frame, [0, 1, 0, 1])
        future = frame.iloc[[0]].copy()
        future["PayloadMass"] = np.nan
        future["LaunchSite"] = "new site"
        self.assertTrue(np.isfinite(model.predict_proba(future)).all())
        self.assertEqual(model.named_steps["preprocess"].named_transformers_["numeric"].named_steps["impute"].statistics_[0], 200.)


if __name__ == "__main__": unittest.main()
