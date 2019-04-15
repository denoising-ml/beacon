from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
from study_sae_prices import sae


def display(images1, images2):
    n = 10  # how many digits we will display
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # display original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(images1[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(images2[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()


if __name__ == "__main__":

    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    training_xs, training_ys = mnist.train.next_batch(1000)
    validation_xs, validation_ys = mnist.train.next_batch(100)

    print(training_xs.shape)
    print(training_ys.shape)

    autoencoder, encoder = sae(training_xs, validation_xs, [32], loss="binary_crossentropy")

    test_xs, test_ys = mnist.train.next_batch(10)
    decoded_imgs = autoencoder.predict(test_xs)

    display(test_xs, decoded_imgs)


