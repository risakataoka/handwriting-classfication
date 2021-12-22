import numpy as np
from PIL import Image
import japanize_matplotlib
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras import models
from keras import layers
import os

# baseディレクトリ
base_dir = 'mnist/data'
# 訓練ディレクトリ
train_dir = os.path.join(base_dir, 'train')
# テストディレクトリ
validation_dir = os.path.join(base_dir, 'test')

# 訓練ディレクトリ配下の数字１～9の画像のディレクトリ
train_0_dir = os.path.join(train_dir, '0')
train_1_dir = os.path.join(train_dir, '1')
train_2_dir = os.path.join(train_dir, '2')
train_3_dir = os.path.join(train_dir, '3')
train_4_dir = os.path.join(train_dir, '4')
train_5_dir = os.path.join(train_dir, '5')
train_6_dir = os.path.join(train_dir, '6')
train_7_dir = os.path.join(train_dir, '7')
train_8_dir = os.path.join(train_dir, '8')
train_9_dir = os.path.join(train_dir, '9')

# 訓練データの合計ファイル数
total_train_images = 0
total_train_images += len(os.listdir(train_0_dir))
total_train_images += len(os.listdir(train_1_dir))
total_train_images += len(os.listdir(train_2_dir))
total_train_images += len(os.listdir(train_3_dir))
total_train_images += len(os.listdir(train_4_dir))
total_train_images += len(os.listdir(train_5_dir))
total_train_images += len(os.listdir(train_6_dir))
total_train_images += len(os.listdir(train_7_dir))
total_train_images += len(os.listdir(train_8_dir))
total_train_images += len(os.listdir(train_9_dir))
print("====================================================")
print("total training  images:", total_train_images)
num_train_images = total_train_images
print("====================================================")

# 各数字毎のファイル数
print("total training 0 images:", len(os.listdir(train_0_dir)))
print("total training 1 images:", len(os.listdir(train_1_dir)))
print("total training 2 images:", len(os.listdir(train_2_dir)))
print("total training 3 images:", len(os.listdir(train_3_dir)))
print("total training 4 images:", len(os.listdir(train_4_dir)))
print("total training 5 images:", len(os.listdir(train_5_dir)))
print("total training 6 images:", len(os.listdir(train_6_dir)))
print("total training 7 images:", len(os.listdir(train_7_dir)))
print("total training 8 images:", len(os.listdir(train_8_dir)))
print("total training 9 images:", len(os.listdir(train_9_dir)))


# 検証ディレクトリ配下の数字１～9の画像のディレクトリ
validation_0_dir = os.path.join(validation_dir, '0')
validation_1_dir = os.path.join(validation_dir, '1')
validation_2_dir = os.path.join(validation_dir, '2')
validation_3_dir = os.path.join(validation_dir, '3')
validation_4_dir = os.path.join(validation_dir, '4')
validation_5_dir = os.path.join(validation_dir, '5')
validation_6_dir = os.path.join(validation_dir, '6')
validation_7_dir = os.path.join(validation_dir, '7')
validation_8_dir = os.path.join(validation_dir, '8')
validation_9_dir = os.path.join(validation_dir, '9')


# 検証データの合計ファイル数
total_validation_images = 0
total_validation_images += len(os.listdir(validation_0_dir))
total_validation_images += len(os.listdir(validation_1_dir))
total_validation_images += len(os.listdir(validation_2_dir))
total_validation_images += len(os.listdir(validation_3_dir))
total_validation_images += len(os.listdir(validation_4_dir))
total_validation_images += len(os.listdir(validation_5_dir))
total_validation_images += len(os.listdir(validation_6_dir))
total_validation_images += len(os.listdir(validation_7_dir))
total_validation_images += len(os.listdir(validation_8_dir))
total_validation_images += len(os.listdir(validation_9_dir))
print("====================================================")
print("total validation　images:", total_validation_images)
num_test_images = total_validation_images
print("====================================================")
print("total validation 0 images:", len(os.listdir(validation_0_dir)))
print("total validation 1 images:", len(os.listdir(validation_1_dir)))
print("total validation 2 images:", len(os.listdir(validation_2_dir)))
print("total validation 3 images:", len(os.listdir(validation_3_dir)))
print("total validation 4 images:", len(os.listdir(validation_4_dir)))
print("total validation 5 images:", len(os.listdir(validation_5_dir)))
print("total validation 6 images:", len(os.listdir(validation_6_dir)))
print("total validation 7 images:", len(os.listdir(validation_7_dir)))
print("total validation 8 images:", len(os.listdir(validation_8_dir)))
print("total validation 9 images:", len(os.listdir(validation_9_dir)))

# 学習モデルの定義
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])

batch_size = 10
# 以下のジェネレータは28x28のグレースケール画像からならるバッチ（(20,28,28,1)と10クラスのラベル(形状は(20,10)を生成する。
# バッチごとに1０のサンプルが存在する。

# 全イメージデータを1/255(0～1)にrescale
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    # ターゲットのディレクトリを指定（今回だとc\django\ML\data\trainフォルダ）
    train_dir,
    # すべての画像を28x28にリサイズする（アップロードするファイルサイズによらず28x28にするため）
    target_size=(28, 28),
    # グレースケール指定（今回はグレースケール画像なので。カラーなら'rgb'を指定する）
    color_mode='grayscale',
    # バッチサイズ
    batch_size=batch_size,
    # categorical_crossentropyをつかうので多クラス分類が必要。
    class_mode='categorical')

# 以下は上記 train_datagenと同じことをしているだけ。
validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(28, 28),
    color_mode='grayscale',
    batch_size=batch_size,
    class_mode='categorical')

# Generatorの戻り値としては、以下のようにデータとラベルに関するデータが生成される。
for data_batch, labels_batch in train_generator:
    print('data batch shape:', data_batch.shape)
    print('labels batch shape:', labels_batch.shape)
    print('indices:', train_generator.class_indices)
    break

# 学習
history = model.fit_generator(
    train_generator,
    steps_per_epoch=num_train_images // batch_size,
    epochs=100,
    validation_data=validation_generator,
    validation_steps=num_test_images // batch_size)


# モデルの保存
model.save('mnist.h5')

# モデルのロード
model = load_model('mnist.h5')


acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training精度')
plt.plot(epochs, val_acc, 'b', color="red", label='Validation精度')
plt.title('Training＆validation精度')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training損失')
plt.plot(epochs, val_loss, 'b', color="red", label='Validation損失')
plt.title('Training＆validation損失')
plt.xlabel('エポック数')
plt.legend()
plt.show()

# 新しいデータで予測
img = Image.open(os.path.join(base_dir, 'sample/sample.jpg'))
gray_img = img.convert('L')
img = gray_img.resize((28, 28))
img = np.array(img).reshape(1, 28, 28, 1)
result = (model.predict(img, batch_size=None, verbose=0, steps=None)).argmax()
print(result)
