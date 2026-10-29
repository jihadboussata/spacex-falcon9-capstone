# Fixed course data snapshot

`dataset_part_2.csv` is the canonical 90-row launch table used by the current workflow. Its SHA-256 is `29852b2f0c716758ee8bc108004dfa166e0296d34fe2b417c61a4161566fe364`. It covers 2010–2020, with 60 successful labels and 30 unsuccessful/no-successful-landing labels.

`dataset_part_1.csv`, `dataset_part_3.csv` and `spacex_launch_geo.csv` are retained course-stage artifacts. The globally encoded dataset_part_3 is not used by the new model pipeline.

The source work follows IBM's Applied Data Science Capstone, using SpaceX API and Wikipedia collection exercises. Preserve original source notices. The current code repairs five known global-mean payload fills and documents the repair in docs/METHODOLOGY.md; it does not claim a fresh API extraction or current launch coverage.
