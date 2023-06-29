import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

confusion_matrix = np.array([[49, 1, 0], [0, 47, 3], [0, 2, 48]])
display_labels = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
cm_display = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=display_labels)
cm_display.plot(cmap='Blues', values_format='d')
plt.show()
