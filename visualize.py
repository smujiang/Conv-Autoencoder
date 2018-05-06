# dependency: pip install pydot & brew install graphviz
from autoencoder import autoencoder
from keras.utils import plot_model


if __name__ == '__main__':
    img_rows, img_cols = 224, 224
    channel = 3
    model = autoencoder(img_rows, img_cols, channel)
    plot_model(model, to_file='model.png')