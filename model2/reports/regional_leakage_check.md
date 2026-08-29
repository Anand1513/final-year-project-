# Regional Leakage Check

We have conducted a thorough audit to prevent target leakage.

### Target Variable
The target to predict (or rank against) is `Yield`.

### Features Used
- `lag1_yield`: Yield from Year - 1.
- `lag2_yield`: Yield from Year - 2.
- `historical_mean_yield`: Expanding mean of yield strictly from years < current Year.
- `NITROGEN`, `PHOSPHOROUS`, `POTASSIUM`, `PH`: District-level soil aggregates.
- `avgtmp_jan`, `avgtmp_jul`, `avgann_rf`: Static regional climate data.

### Verification
The `Yield` and `Production` of the current row (Target Year) are completely excluded from the input feature vector during training.
