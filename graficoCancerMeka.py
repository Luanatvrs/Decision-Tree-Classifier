import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

confusion_matrix = np.array([[197, 15], [23, 334]])

display_labels = ['Malignant', 'Benign']

cm_display = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=display_labels)

cm_display.plot(cmap='Blues', values_format='d')

plt.show()
