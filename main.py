from preprocessing.data_cleaning import clean_data, normalize
from models.cnn_model import create_cnn
from models.lstm_model import create_lstm
from models.dnn_model import create_dnn
from models.ensemble_model import ensemble_predict
from utils.metrics import evaluate
import pandas as pd

# Load data
data = pd.read_csv('data/yield_data.csv')
data = normalize(clean_data(data))

# Split features and labels
X, y = data.drop('yield', axis=1), data['yield']

# Initialize models (example shapes)
cnn = create_cnn((64,64,3))
lstm = create_lstm((10,5))
dnn = create_dnn(X.shape[1])

# Example predictions
cnn_preds = cnn.predict(X_cnn)
lstm_preds = lstm.predict(X_lstm)
dnn_preds = dnn.predict(X_dnn)

final_preds = ensemble_predict(cnn_preds, lstm_preds, dnn_preds)
metrics = evaluate(y, final_preds)
print(metrics)
