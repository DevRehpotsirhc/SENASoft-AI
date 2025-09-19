import numpy as np
import matplotlib.pyplot as plt

def frontera(x_combined, y_combined, model):
    """
    Se encarga de gráficar las fronteras de dicisión
    de los modelos predictivos

    Args:
        x_combined (dataset): combinación de las características (de train y test) estandarizadas StandarScaler
        y_combined (dataset): combinación de las etiquetas
        model (mp): modelo predictivo ej: LogisticRegression
    """
    x_min, x_max = x_combined[:,0].min() -1, x_combined[:,0].max() +1
    y_min, y_max = x_combined[:,1].min() -1, x_combined[:,1].max() +1

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 500),
        np.linspace(y_min, y_max, 500)
    )

    z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    z = z.reshape(xx.shape)

    plt.contourf(xx, yy, z, alpha=.3, cmap="coolwarm")
    plt.scatter(x_combined[:,0], x_combined[:,1], c=y_combined, edgecolors="k", cmap="coolwarm")
    plt.title("Frontera de desición - SVM")
    plt.show()