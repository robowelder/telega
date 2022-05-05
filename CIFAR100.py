import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


def draw_img(x):
    plt.figure()
    plt.imshow(x)
    plt.show()

(cx_train, cy_train), (cx_test, cy_test) = tf.keras.datasets.cifar100.load_data()
cx_train, cx_test = cx_train / 255.0, cx_test / 255.0

classes = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
    'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
    'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
    'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
    'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
    'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
    'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm tree', 'pear',
    'pickup_truck', 'pine tree', 'plain', 'plate', 'poppy', 'porcupine',
    'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
    'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
    'spider', 'squirrel', ' streetcar', 'sunflower', 'sweet_pepper', 'table',
    'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
    'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm'
]

idx = np.random.randint(0,9999)
draw_img(cx_test[idx])
print(classes[cy_test[idx][0]])

model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Conv2D(32, kernel_size = (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Conv2D(32, kernel_size = (3, 3), kernel_initializer = 'he_uniform', activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.MaxPooling2D(pool_size = (2, 2)))
model.add(tf.keras.layers.Dropout(0.25))

model.add(tf.keras.layers.Conv2D(64, kernel_size = (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Conv2D(64, kernel_size = (3, 3), kernel_initializer = 'he_uniform', activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.MaxPooling2D(pool_size = (2, 2)))
model.add(tf.keras.layers.Dropout(0.25))

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(512, activation = 'relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(100, activation = 'softmax'))
opt = tf.keras.optimizers.SGD(lr = 0.01, momentum = 0.9, decay = 1e-6, nesterov = True)
model.compile(optimizer = opt, loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
model.summary()
tf.keras.utils.plot_model(model, show_shapes = True)


# %load_ext tensorboard
import datetime

# ! rm - rf./logs/

log_dir = 'logs/fit/' +datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
tensorboard_callback =tf.keras.callback.TensorBoard(log_dir=log_dir, histogram=1)


history = model.fit(cx_train, cy_train, epochs = 5)
# for tensorboard
history = model.fit(cx_train, cy_train, epochs = 100, validation_data = (cx_test, cy_test), callbacks = [tensorboard_callback])

model.save('cifar100.h5'
)
