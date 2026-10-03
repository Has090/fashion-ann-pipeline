import json
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

x_test = np.load('data/processed/x_test.npy')
y_test = np.load('data/processed/y_test.npy')
model = load_model('models/model.h5')

loss, accuracy = model.evaluate(x_test, y_test)

predictions = np.argmax(model.predict(x_test), axis=1)
cm = confusion_matrix(y_test, predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.savefig('confusion_matrix.png')

with open('metrics.json', 'w') as f:
    json.dump({'test_loss': loss, 'test_accuracy': accuracy}, f)