# Soil Feature Validation

## Source Data
The `soil-nutrient-analysis.csv` dataset contains block- and village-level soil nutrient samples. 
Relevant columns: `state_name`, `district_name`, `nutrient_name`, `value`.

## Handling of the `value` Field
The `value` field represents the observed concentration of a nutrient for a specific sample. 

### Methodology for Regional Features
Since the goal of the Regional Model is to predict crop performance at the **District Level**, we cannot use individual soil samples directly. 
Instead, we construct District-level Soil Aggregates:
1. **Filtering**: Extract rows where `nutrient_name` equals N, P, K, or pH (if available).
2. **Aggregation**: Calculate the median `value` grouped by `state_name` and `district_name` for each nutrient. 
3. **Imputation**: If a district is missing a specific nutrient aggregate, we fallback to the State-level median to prevent data loss, followed by the global median.

This method provides a scientifically defensible district-level soil profile without fabricating individual values.
