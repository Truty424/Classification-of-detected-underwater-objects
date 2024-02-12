import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

print(tf.config.list_physical_devices())

data_dir = '/Users/adam/PycharmProjects/DataAugmentation/Images/'
class_names = os.listdir(data_dir)
image_data = []
labels = []


def load_img(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # cv2.COLOR_BGR2RGB
    img = cv2.resize(img, (450, 300))
    return img


def load_train_set(data_dir, class_names, image_data, labels):
    for class_name in class_names:
        class_path = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_path):
            continue  # Ignoruj pliki, które nie są katalogami
        for image_name in os.listdir(class_path):
            image_path = os.path.join(class_path, image_name)
            if image_name.startswith('.'):
                continue  # Ignoruj pliki ukryte, takie jak .DS_Store
            img = load_img(image_path)
            image_data.append(img)
            labels.append(class_name)
    x_train = np.array(image_data)
    y_train = np.array(labels)
    return x_train, y_train


def prepare_data():
    # Podział danych na zbiór treningowy, walidacyjny i testowy (np. 80% treningowy, 10% walidacyjny, 10% testowy)
    img_data, label = load_train_set(data_dir, class_names, image_data, labels)
    x_train, x_temp, y_train, y_temp = train_test_split(img_data, label, test_size=0.2, random_state=42)
    x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42)

    x_train = np.array(x_train) / 255.0
    x_val = np.array(x_val) / 255.0
    x_test = np.array(x_test) / 255.0

    # Zakodowanie etykiet klas
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_val_encoded = label_encoder.transform(y_val)
    y_test_encoded = label_encoder.transform(y_test)

    y_train_encoded = tf.one_hot(y_train_encoded, depth=3)  # 3 to liczba klas
    y_val_encoded = tf.one_hot(y_val_encoded, depth=3)
    y_test_encoded = tf.one_hot(y_test_encoded, depth=3)

    return x_train, y_train_encoded, x_val, y_val_encoded, x_test, y_test_encoded


def train_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(300, 450, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),

        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),

        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),

        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),

        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.5),

        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),

        tf.keras.layers.Dense(3, activation='softmax')

    ])
    model.summary()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

    x_train, y_train_encoded, x_val, y_val_encoded, x_test, y_test_encoded = prepare_data()

    history = model.fit(x_train, y_train_encoded, epochs=5, validation_data=(x_val, y_val_encoded))
    model.save('final_model.h5')
    #
    test_loss, test_accuracy = model.evaluate(x_test, y_test_encoded)
    print(f"Test accuracy: {test_accuracy}")


if __name__ == '__main__':
    train_model()
