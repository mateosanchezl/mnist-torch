import mnist_model
from importlib import reload
reload(mnist_model)

mnist_model.run(32, 0.01, 1, "./datasets", "Adam")