import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import os

# 1. Create a list of 93 Excel files
excel_files = [rf'C:\Users\A\Desktop\A002\dataframes/0{str(i).zfill(2)}.xlsx' for i in range(1, 93)]

dfs = []
for excel_file in excel_files:
    if os.path.exists(excel_file):  # Check if the file exists
        df = pd.read_excel(excel_file)
        df['collision'] = df['collision'].astype(str)  # Convert 'collision' column to string type
        image_folder = f'C:/Users/A/Desktop/A002/{str(excel_files.index(excel_file) + 1).zfill(3)}'  # Folder path (001 to 093)
        df['image_path'] = df['file'].apply(lambda x: os.path.join(image_folder, str(x) + '.png'))
        df['image_path'] = df['image_path'].apply(lambda x: x.replace("\\", "/"))  # Ensure forward slashes
        dfs.append(df)
    else:
        print(f"Warning: {excel_file} not found, skipping.")  # Skip file if it doesn't exist

# Combine all datasets
df_combined = pd.concat(dfs, ignore_index=True)

# Randomly select 83 datasets for training, and the remaining 10 for validation
train_df = df_combined.sample(frac=83/93, random_state=42)  # Randomly select 83 samples for training
val_df = df_combined.drop(train_df.index)  # Remaining 10 samples used for validation

# Check if all image paths are valid
invalid_paths_train = [path for path in train_df['image_path'] if not os.path.exists(path)]
invalid_paths_val = [path for path in val_df['image_path'] if not os.path.exists(path)]

if invalid_paths_train or invalid_paths_val:
    print(f"Invalid image paths in training set: {invalid_paths_train}")
    print(f"Invalid image paths in validation set: {invalid_paths_val}")
else:
    print("All image paths are valid.")

# 3. Create data augmentation generator for training
train_datagen = ImageDataGenerator(
    rescale=1./255,  # Normalize pixel values to [0,1]
    rotation_range=20,  # Random rotation
    width_shift_range=0.2,  # Horizontal shift
    height_shift_range=0.2,  # Vertical shift
    shear_range=0.2,  # Shear transformation
    zoom_range=0.2,  # Zoom in/out
    horizontal_flip=True,  # Horizontal flip
    fill_mode='nearest'  # Fill strategy
)

# 4. Load training data using flow_from_dataframe
train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,           # Training DataFrame
    directory=None,               # Image path provided in image_path column
    x_col='image_path',           # Image path column
    y_col='collision',            # Label column
    target_size=(224, 224),       # Resize images to 224x224
    batch_size=32,                # Batch size
    class_mode='binary'           # Binary classification (collision or not)
)

# 5. Create validation data generator
val_datagen = ImageDataGenerator(rescale=1./255)  # Only normalize for validation

# Load validation data using flow_from_dataframe
val_generator = val_datagen.flow_from_dataframe(
    dataframe=val_df,
    directory=None,
    x_col='image_path',
    y_col='collision',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

# 6. Define the model
base_model = models.Sequential([
    layers.InputLayer(input_shape=(224, 224, 3)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # Binary output (0 or 1)
])

# Compile the model
base_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 7. Train the model
history = base_model.fit(
    train_generator,
    epochs=100,  # Train for 100 epochs
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=val_generator,
    validation_steps=val_generator.samples // val_generator.batch_size
)

# 8. Evaluate the model
test_loss, test_accuracy = base_model.evaluate(
    val_generator,
    steps=val_generator.samples // val_generator.batch_size
)

print(f"Test Loss (on validation data): {test_loss}")
print(f"Test Accuracy (on validation data): {test_accuracy}")

# 9. Visualize training loss and accuracy
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Training and Validation Loss and Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss / Accuracy')
plt.legend()
plt.show()
