# Baseline vs Regional Model Comparison

## Baseline Model (7-Parameter)
- **Features**: N, P, K, Temperature, Humidity, pH, Rainfall.
- **Goal**: Predict physiological suitability of a crop strictly based on soil and weather.
- **Status**: Preserved and functioning perfectly. Does not use District/State historical trends.

## Regional Model
- **Features**: State, District, Season, Nitrogen, Phosphorus, Potassium, pH, Rainfall, Historical Yields.
- **Goal**: Predict crop performance historically based on the specific location's agricultural track record.
- **Status**: Trained and saved as a separate pipeline.

## Conclusion
The two models serve complementary purposes. The baseline recommends crops that *could* grow based on chemistry, while the regional model recommends crops that *traditionally perform well* in that specific district. The baseline model was strictly preserved.
