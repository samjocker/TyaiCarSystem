import tensorflow as tf
from keras import layers, models
from keras.preprocessing.image import ImageDataGenerator

# 指定資料夾路徑
train_data_dir = "train\image"

# 設定圖片大小和通道數
img_width, img_height = 30, 30
channels = 3  # 彩色圖片為3，灰度圖片為1

# 創建模型
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, channels)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(8, activation='softmax'))

# 編譯模型
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 使用ImageDataGenerator進行資料增強
train_datagen = ImageDataGenerator(rescale=1./255)

# 設定批次大小和目標大小
batch_size = 32
target_size = (img_width, img_height)

# 使用flow_from_directory生成訓練資料
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=target_size,
    batch_size=batch_size,
    class_mode='sparse'
)

# 訓練模型
model.fit(train_generator, epochs=10)

# 儲存模型
model.save("trained_model.h5")

print("模型訓練完成並保存為 trained_model.h5")
