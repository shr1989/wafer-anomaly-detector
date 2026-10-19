# Wafer Guard — Technical Audit

## 1. Data leakage invalidated the original evaluation

**Location:** Original preprocessing and resampling pipeline.

The original pipeline fitted `SimpleImputer` and `StandardScaler` on the full dataset, applied SMOTE before splitting, and then randomly split the resampled data. Consequently, information from the evaluation population influenced preprocessing, and synthetic samples derived from the original dataset could enter evaluation. This made the reported release-gate score unreliable.

**Repair:** Sort wafers chronologically, reserve the latest 20% for final testing, and fit preprocessing exclusively on training data. Keep the test set untouched during model selection and evaluate at natural prevalence.

## 2. Incorrect tensor broadcasting corrupted reconstruction loss

**Location:** Original `make_loader` and reconstruction-loss calculation.

The original loader used `unsqueeze(1)`, creating input batches shaped `[B, 1, F]`. The model flattened its input and returned reconstructions shaped `[B, F]`. Subtracting these tensors directly caused PyTorch broadcasting to produce `[B, B, F]`, comparing reconstructions against inputs from other wafers and corrupting the reconstruction-loss gradients.

**Repair:** Remove the unnecessary channel dimension, enforce `[B, F]` inputs, and assert that reconstruction and input shapes match before computing the loss. The repaired notebook's gradient sanity check reported nonzero gradients for the encoder, decoder, and classification head.

## 3.Legacy probe-station telemetry was treated as ordinary numeric data

**Location:** Original feature preprocessing and reconstruction objective.

PS-L legacy wafers use special telemetry conventions. In particular, `sensor_0 = 0` is a sentinel, and `sensor_1` contains firmware codes `100`, `200`, and `300`. Treating these values as ordinary continuous measurements distorts imputation, scaling, and learned representations.

**Repair:** Identify legacy wafers, handle sentinel values as missing where appropriate, encode firmware codes as categorical states, and fit imputation and scaling parameters on training data only.

## Evaluation and residual risk

The repaired notebook's recorded chronological test run evaluated **314 wafers**, including **17 defective wafers (5.414% prevalence)**. At a threshold selected using validation data only:

- Precision: **0.1236**
- Recall: **0.6471**
- F1: **0.2075**
- F2: **0.3503**
- Average precision: **0.1236**
- Confusion matrix `[ [TN, FP], [FN, TP] ]`: `[[219, 78], [6, 11]]`

These results are substantially below the claimed F1 release gate of 0.90. The model detects 11 of 17 defects but raises 78 false alarms among passing wafers. Average precision is important because it measures ranking quality under class imbalance; the threshold metrics show the operational trade-off at the selected operating point.

**Residual risks:** The test set contains only 17 defects, so estimates are uncertain. Temporal drift, firmware-specific failure patterns, and performance on future production lots require further validation.

