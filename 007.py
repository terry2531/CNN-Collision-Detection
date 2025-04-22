import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import os
import numpy as np
import cv2
from tensorflow.keras.utils import Sequence  # 导入 Sequence

# 1. 创建包含93个 Excel 文件的列表
excel_files = [f'C:/Users/A/Desktop/video/dataframes/video-000{str(i).zfill(2)}.xlsx' for i in range(1, 94)]

dfs = []
for excel_file in excel_files:
    if os.path.exists(excel_file):  # 检查文件是否存在
        df = pd.read_excel(excel_file)
        df['collision'] = df['collision'].astype(int)  # 将 'collision' 列转换为整数类型
        image_folder_1 = f'C:/Users/A/Desktop/video/{str(excel_files.index(excel_file) + 1).zfill(3)}'  # 文件夹路径1（001 到 093）
        image_folder_2 = f'C:/Users/A/Desktop/video001/{str(excel_files.index(excel_file) + 1).zfill(3)}'  # 文件夹路径2
        df['image_path_1'] = df['file'].apply(lambda x: os.path.join(image_folder_1, x + '.png'))
        df['image_path_2'] = df['file'].apply(lambda x: os.path.join(image_folder_2, x + '.png'))
        df['image_path_1'] = df['image_path_1'].apply(lambda x: x.replace("\\", "/"))  # 确保使用正斜杠
        df['image_path_2'] = df['image_path_2'].apply(lambda x: x.replace("\\", "/"))
        dfs.append(df)
    else:
        print(f"Warning: {excel_file} not found, skipping.")  # 如果文件不存在，跳过该文件

# 合并所有数据集
df_combined = pd.concat(dfs, ignore_index=True)

# 随机选择83个数据集作为训练集，剩余10个数据集作为验证集
train_df = df_combined.sample(frac=83 / 93, random_state=42)  # 随机选择83个样本
val_df = df_combined.drop(train_df.index)  # 剩余的10个样本作为验证集

# 检查所有图像路径是否有效，并跳过无效路径
train_df = train_df[train_df['image_path_1'].apply(os.path.exists) & train_df['image_path_2'].apply(os.path.exists)]
val_df = val_df[val_df['image_path_1'].apply(os.path.exists) & val_df['image_path_2'].apply(os.path.exists)]

# 2. 定义自定义生成器以加载两张图像
class DualImageGenerator(Sequence):
    def __init__(self, dataframe, batch_size, target_size, shuffle=True):
        self.dataframe = dataframe
        self.batch_size = batch_size
        self.target_size = target_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.dataframe))

    def __len__(self):
        return int(np.floor(len(self.dataframe) / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        batch_indexes = self.indexes[index * self.batch_size: (index + 1) * self.batch_size]
        batch_df = self.dataframe.iloc[batch_indexes]

        # 加载图像和标签
        image_1 = []
        image_2 = []
        labels = []

        for img_path_1, img_path_2, label in zip(batch_df['image_path_1'], batch_df['image_path_2'],
                                                 batch_df['collision']):
            # 检查标签是否有效
            if pd.isna(label):  # 如果标签是NaN，跳过该样本
                print(f"Skipping image pair due to invalid label: {img_path_1}, {img_path_2}")
                continue

            try:
                img_1 = cv2.imread(img_path_1)
                img_2 = cv2.imread(img_path_2)

                if img_1 is None or img_2 is None:
                    print(f"Warning: {img_path_1} or {img_path_2} could not be loaded.")
                else:
                    img_1 = cv2.resize(img_1, self.target_size)
                    img_2 = cv2.resize(img_2, self.target_size)
                    image_1.append(img_1 / 255.0)
                    image_2.append(img_2 / 255.0)
                    labels.append(label)
            except Exception as e:
                print(f"Error loading {img_path_1} and {img_path_2}: {e}")
                continue

        # 打印当前批次的数据
        print(f"Batch {index}: Loaded {len(image_1)} pairs of images.")

        # 如果没有有效图像
        if len(image_1) == 0 or len(image_2) == 0:
            print(f"Warning: No valid images in batch {index}")

        # 转换为numpy数组
        image_1 = np.array(image_1)
        image_2 = np.array(image_2)
        labels = np.array(labels)

        # 检查图像和标签的维度
        print(f"Image 1 shape: {image_1.shape}, Image 2 shape: {image_2.shape}, Labels shape: {labels.shape}")

        # 合并两个图像
        x = np.concatenate([image_1, image_2], axis=-1)

        # 打印输出 x 和 y 形状
        print(f"x shape: {x.shape}, y shape: {labels.shape}")

        # 确保返回的数据不为空
        if x.size == 0 or labels.size == 0:
            print(f"Warning: Empty data in batch {index}")

        return x, labels


# 3. 定义数据增强生成器
train_datagen = ImageDataGenerator(
    rescale=1. / 255,  # 将图像像素值缩放到0-1之间
    rotation_range=20,  # 随机旋转图像
    width_shift_range=0.2,  # 水平平移
    height_shift_range=0.2,  # 垂直平移
    shear_range=0.2,  # 随机错切变换
    zoom_range=0.2,  # 随机缩放
    horizontal_flip=True,  # 随机水平翻转
    fill_mode='nearest'  # 填充方式
)

# 4. 使用 DualImageGenerator 创建训练和验证生成器
train_generator = DualImageGenerator(train_df, batch_size=32, target_size=(224, 224))
val_generator = DualImageGenerator(val_df, batch_size=32, target_size=(224, 224), shuffle=False)

# 5. 定义模型（与之前相同）
base_model = models.Sequential([
    layers.InputLayer(input_shape=(224, 224, 6)),  # 这里的输入形状是(224, 224, 6)，因为有两个图像作为输入
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # 输出一个0或1
])

# 编译模型
base_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 6. 训练模型
history = base_model.fit(
    train_generator,  # 训练数据生成器
    epochs=10,  # 训练20轮
    steps_per_epoch=train_generator.__len__(),  # 每个epoch的步数
    validation_data=val_generator,  # 验证数据生成器
    validation_steps=val_generator.__len__()  # 每个验证步骤
)

# 7. 测试模型
test_loss, test_accuracy = base_model.evaluate(
    val_generator,  # 使用验证集数据进行测试
    steps=val_generator.__len__()  # 每个验证步骤
)

print(f"Test Loss (on validation data): {test_loss}")
print(f"Test Accuracy (on validation data): {test_accuracy}")

# 8. 可视化训练过程中的损失和准确度
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Training and Validation Loss and Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss / Accuracy')
plt.legend()
plt.show()
