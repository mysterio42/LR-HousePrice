import matplotlib.pyplot as plt


def model_plot(model, features, labels):
    """
    :param model: LinearRegression model
    :param features: sqft_living features
    :param labels: House price labels
    :return: plot predicted linear line with the data
    """
    plt.scatter(features, labels, color='green')
    plt.plot(features, model.predict(features), color='black')
    plt.title('Linear Regression')
    plt.xlabel('sqft_living')
    plt.ylabel('price')
    plt.show()
