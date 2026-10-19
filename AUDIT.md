AUDIT.md — Wafer Anomaly Detector
1. Silent Bugs and Fixes I could make
1. Data Leakage
Bug: The original pipeline used preprocessing and evaluation procedures that could allow information from future wafers or the evaluation population to influence training, making the reported F1 score unreliable.
Fix: I sorted the data chronologically and split it into training, validation, and test sets before fitting preprocessing transformations. The imputer and scaler are fitted on training data only. Validation data is used for model selection and threshold selection, while the test set is reserved for final evaluation.
2. Incorrect Loss and Gradient Computation
Bug: Incorrect tensor dimensions can cause PyTorch to broadcast predictions and targets during loss calculation instead of comparing corresponding samples. This can produce an incorrect loss and gradients without an obvious runtime error.
Fix: I aligned the prediction and target shapes, separated continuous-feature reconstruction from classification, and added shape assertions and gradient checks. The checks verify that the reconstruction output matches its target, classification logits match the label batch, and all intended model branches receive non-zero gradients.
3. Invariant Blindness — Legacy Probe Station PS-L
Bug: The original preprocessing treated telemetry values as ordinary continuous measurements without preserving the legacy probe station's different data semantics. For PS-L wafers, sensor_0 = 0.0 is a sentinel, and sensor_1 contains categorical firmware codes 100, 200, and 300.
Fix: I handled the PS-L sentinel separately and represented firmware codes categorically rather than scaling them as continuous measurements. The repaired model reconstructs continuous features separately and includes a firmware-classification head, allowing it to learn the categorical firmware information without treating the codes as continuous values.
2. Evaluation at Natural Defect Prevalence
The dataset contains 1,567 wafers, including 104 defective wafers, giving an overall defect prevalence of approximately 6.64%.
The chronological test set contains 314 wafers, including 17 defective wafers. Its defect prevalence is 5.41%. The test set was not artificially balanced.
Final Test Results
Metric
Result
Precision
0.1222
Recall
0.6471
F1 score
0.2056
F2 score
0.3481
Average Precision
0.1258
True positives
11
False positives
79
False negatives
6
3. Conclusion and Recommendation
The repaired model detects 11 of the 17 defective wafers in the held-out test set. However, 79 passing wafers are incorrectly flagged, resulting in low precision and an F1 score of 0.2056.
The original reported F1 score of at least 0.90 should not be accepted without a leakage-safe evaluation. The repaired results provide a more realistic assessment of performance on naturally imbalanced data.
Recommendation: No-go for production deployment at this stage. The pipeline addresses the identified preprocessing, tensor-shape, gradient, and PS-L handling issues, but the false-positive rate remains high. Further model tuning and validation are required before production release.
