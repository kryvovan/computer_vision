import tensorflow as tf

from tensorflow.keras import layers, models
import numpy as np

from tensorflow.keras.preprocessing import image

train_ds = tf.keras.preprocessing.image_dataset_from_directory('Data/train', image_size=(128, 128), batch_size=30, label_mode='categorical')
test_ds = tf.keras.preprocessing.image_dataset_from_directory('Data/test', image_size=(128, 128), batch_size=30, label_mode='categorical')

#нормалізація зображень

normalization_layer = layers.Rescaling(1./255)

train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))

#побудова моделі

model = models.Sequential()
#перший фільтр
model.add(layers.Conv2D(filters = 32, kernel_size = (3, 3), activation = 'relu', input_shape = (128, 128, 3)))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(filters = 64, kernel_size = (3, 3), activation = 'relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(filters = 128, kernel_size = (3, 3), activation = 'relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Flatten())

#внутрішній шар

model.add(layers.Dense(64, activation = 'relu'))

model.add(layers.Dense(3, activation = 'softmax'))


model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

history = model.fit(
    train_ds,
    epochs = 15,
    validation_data = test_ds,
)

test_lost, test_accuracy = model.evaluate(test_ds)
print(f'Якість: {test_accuracy}')

class_name = ['bananas', 'cows', 'people']

img = image.load_img('Images/Banana.jpg', target_size = (128, 128))
image_array = image.img_to_array(img)

image_array = image_array / 255.0

image_array = tf.expand_dims(image_array, 0)

predictions = model.predict(image_array)

predicted_index = np.argmax(predictions[0])
print(f'Імовірність: {predictions[0]}')
print(f'Модель визначила: {class_name[predicted_index]}')