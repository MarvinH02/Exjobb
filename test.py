import tensorflow as tf
import time
import pandas as pd
import matplotlib.pyplot as plt

# 1. Load and prepare the MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize and reshape
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]

# 2. Build the CNN model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# 3. Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 4. Train and time the model
start_time = time.time()
history = model.fit(x_train, y_train, epochs=5, validation_split=0.1, verbose=2)
end_time = time.time()

# 5. Evaluate on test data
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"\nTest accuracy: {test_acc:.4f}")
print(f"Training time: {end_time - start_time:.2f} seconds")

# 6. Save results to CSV
results = {
    'Framework': ['TensorFlow'],
    'Training Time (s)': [end_time - start_time],
    'Test Accuracy': [test_acc]
}
df = pd.DataFrame(results)
df.to_csv('benchmark_results_python.csv', index=False)
print("\nResults saved to benchmark_results_python.csv")

# 7. Optional: Plot accuracy and loss
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Training Accuracy per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
