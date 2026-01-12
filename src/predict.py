# Step 9: Prediction pipeline
def predict(features):
    model = joblib.load("models/buy_sell_classifier.pkl")
    return model.predict(features)
