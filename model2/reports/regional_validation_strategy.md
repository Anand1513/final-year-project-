# Regional Validation Strategy

To prevent target leakage and ensure temporal generalization, we use a strict temporal split.

- **Train Set**: Years 1997 to 2013
- **Validation Set**: Year 2014
- **Test Set**: Year 2015

This guarantees that the model only learns from historical trends and is tested on unseen future periods.
