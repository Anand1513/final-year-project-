import shap
import matplotlib
# Use Agg backend for thread-safe server rendering without a display
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
import numpy as np
import warnings

# Suppress sklearn warnings about unpickling
warnings.filterwarnings("ignore", category=UserWarning)

# Global cache for the explainer to speed up repeated inference
_explainer = None

def get_shap_explainer(model):
    global _explainer
    if _explainer is None:
        _explainer = shap.TreeExplainer(model)
    return _explainer

def generate_shap_plot_base64(model, scaler, feature_values, feature_names):
    """
    Generates a SHAP plot for the predicted class and returns it as a base64 encoded PNG.
    """
    explainer = get_shap_explainer(model)
    
    input_vector = np.array([feature_values])
    scaled_vector = scaler.transform(input_vector)
    
    # Calculate SHAP values
    shap_values = explainer.shap_values(scaled_vector)
    
    # Get predicted class index
    predicted_class_idx = int(model.predict(scaled_vector)[0])
    
    # For RandomForest, shap_values is typically a list of arrays (one array per class)
    if isinstance(shap_values, list):
        class_shap_values = shap_values[predicted_class_idx][0]
    else:
        class_shap_values = shap_values[0, :, predicted_class_idx]
        
    # Create the plot matching terminal UI dark mode
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(6, 3.5))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#0d1117')
    
    y_pos = np.arange(len(feature_names))
    
    # Green for positive impact, Red for negative impact
    colors = ['#4ade80' if val > 0 else '#f85149' for val in class_shap_values]
    
    ax.barh(y_pos, class_shap_values, align='center', color=colors)
    ax.set_yticks(y_pos, labels=feature_names)
    ax.invert_yaxis()  # labels read top-to-bottom
    
    ax.set_xlabel('SHAP Value (Impact on Prediction)', color='#8b949e')
    ax.set_title('Feature Importance (SHAP)', color='#c9d1d9', pad=15)
    
    ax.tick_params(colors='#8b949e')
    for spine in ax.spines.values():
        spine.set_color('#30363d')
        
    plt.tight_layout()
    
    # Save to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=120, bbox_inches='tight', facecolor='#0d1117')
    plt.close('all')
    
    # Encode as base64
    img_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    return img_b64
