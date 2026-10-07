# Customer Satisfaction: PCA, Regularization and Boosting

Built a reproducible customer-satisfaction benchmark on 76,020 records, comparing regularization, PCA and gradient boosting. Used duplicate-aware partitions and training-only preprocessing to prevent leakage. The selected gradient-boosting model achieved held-out ROC-AUC 0.836 versus 0.779 for the same-split logistic baseline, with average precision improving from 0.142 to 0.194. Documented the limits of accuracy on a highly imbalanced target.

**Skills:** Python, scikit-learn, PCA, imbalanced classification, experiment design.

**Supporting context:** The target is rare: a majority-only classifier already achieves about 96% accuracy. Report ROC-AUC and average precision, not accuracy as evidence of strong detection. At the default 0.5 threshold, minority recall remains very low; operational threshold/cost calibration is future work. The historical cross-validation score is not directly comparable to this new test split.
