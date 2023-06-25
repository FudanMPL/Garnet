Machine Learning
----------------

MP-SPDZ supports a limited subset of the Keras interface for machine
learning. This includes the SGD and Adam optimizers and the following
layer types: dense, 2D convolution, 2D max-pooling, and dropout.

The machine learning code only works in with arithmetic machines, that
is, you cannot compile it with ``-B``.

In the following we will walk through the example code in
``keras_mnist_dense.mpc``, which trains a dense neural network for
MNIST. It starts by defining tensors to hold data::

  training_samples = sfix.Tensor([60000, 28, 28])
  training_labels = sint.Tensor([60000, 10])

  test_samples = sfix.Tensor([10000, 28, 28])
  test_labels = sint.Tensor([10000, 10])

The tensors are then filled with inputs from party 0 in the order that
is used by ``convert.sh`` in `the preparation code
<https://github.com/csiro-mlai/deep-mpc>`_::

  training_labels.input_from(0)
  training_samples.input_from(0)

  test_labels.input_from(0)
  test_samples.input_from(0)

This is followed by Keras-like code setting up the model and training
it::

  from Compiler import ml
  tf = ml

  layers = [
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10,  activation='softmax')
  ]

  model = tf.keras.models.Sequential(layers)

  optim = tf.keras.optimizers.SGD(momentum=0.9, learning_rate=0.01)

  model.compile(optimizer=optim)

  opt = model.fit(
    training_samples,
    training_labels,
    epochs=1,
    batch_size=128,
    validation_data=(test_samples, test_labels)
  )

Lastly, the model is stored on disk in secret-shared form::

  for var in model.trainable_variables:
    var.write_to_file()


Prediction
~~~~~~~~~~

The example code in ``keras_mnist_dense_predict.mpc`` uses the model
stored above for prediction. Much of the setup is the same, but
instead of training it reads the model from disk::

  model.build(test_samples.sizes)

  start = 0
  for var in model.trainable_variables:
    start = var.read_from_file(start)

Then it runs the prediction::

  guesses = model.predict(test_samples)

Using ``var.input_from(player)`` instead the model would be input
privately by a party.
