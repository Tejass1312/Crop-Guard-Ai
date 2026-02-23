def analyze_risk(disease, confidence):
    if disease == "healthy":
        return {
            "risk_level": "Low",
            "severity": "None",
            "recommendation": "Crop is healthy. Continue regular monitoring."
        }

    if confidence > 0.85:
        level = "High"
    elif confidence > 0.65:
        level = "Medium"
    else:
        level = "Low"

    return {
        "risk_level": level,
        "severity": "Moderate" if level != "Low" else "Mild",
        "recommendation": "Apply recommended fungicide and monitor crop weekly."
    }
