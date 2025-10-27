組員：11121003 卓逸峰

-----

# CIFAR-10 圖像分類器 (Keras CNN)

這是一個使用 Keras 建立的簡單卷積神經網路 (CNN) 專案，用於對 CIFAR-10 數據集中的圖片進行分類。CIFAR-10 包含 10 個類別的 32x32 彩色圖片。

此 README 檔案將逐步解釋程式碼的結構、模型的建立、訓練過程以及如何解讀其效能結果。

## 專案目標

  * 載入並預處理 CIFAR-10 數據集。
  * 建立一個序貫 (Sequential) CNN 模型。
  * 編譯並訓練模型。
  * 使用 `matplotlib` 將訓練過程 (準確率與損失) 視覺化。
  * 評估模型的最終準確率。
  * 使用 `pandas` 建立並分析混淆矩陣 (Confusion Matrix)，以了解模型的具體預測情況。

## 程式碼說明

### 1\. 設置與資料預處理

此階段的目的是載入函式庫、設定環境並準備用於訓練的資料。

```python
import numpy
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD
from tensorflow.python.keras.utils import np_utils

# 設定隨機種子以確保結果的可重現性
seed = 7
numpy.random.seed(seed)

# 載入 CIFAR-10 數據集
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# 標準化 (Normalization)
# 將像素值從 0-255 (整數) 轉換為 0.0-1.0 (浮點數)
X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0

# 獨熱編碼 (One-Hot Encoding)
# 將類別標籤 (例如 3) 轉換為二進制向量 (例如 [0,0,0,1,0,0,0,0,0,0])
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]
```

### 2\. 建立 CNN 模型架構

我們使用 `Sequential` 模型，它允許我們按順序堆疊神經網路層。

  * **`Conv2D`**: 卷積層，使用 3x3 濾鏡提取圖像特徵。
  * **`Dropout`**: 丟棄層，隨機關閉部分神經元以防止過度擬合 (Overfitting)。
  * **`MaxPooling2D`**: 最大池化層，將特徵圖縮小 (降維)，同時保留最重要的特徵。
  * **`Flatten`**: 扁平層，將 3D 的特徵圖轉換為 1D 向量，以輸入到全連接層。
  * **`Dense`**: 全連接層，負責整合特徵並進行分類。
  * **`softmax`**: 輸出層的激活函數，將輸出轉換為 10 個類別的機率分佈。



```python
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(3, 3), input_shape=(32, 32, 3), padding="same", activation="relu"))
model.add(Dropout(0.2))
model.add(Conv2D(filters=32, kernel_size=(3, 3), activation="relu", padding="same"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(512, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation="softmax"))

# 印出模型摘要
print(model.summary())
```

### 3\. 編譯與訓練模型

在訓練之前，我們需要「編譯」模型，設定優化器、損失函數和評估指標。

  * **優化器 (Optimizer)**: `SGD` (隨機梯度下降)，並加入動量 (Momentum) 來加速收斂。
  * **損失函數 (Loss)**: `categorical_crossentropy`，適用於多類別分類問題 (搭配 One-Hot 編碼)。
  * **評估指標 (Metrics)**: `accuracy` (準確率)。

`model.fit()` 函數會啟動訓練過程，`train_history` 物件會儲存每個 epoch 的訓練日誌。

```python
# 編譯模型
epochs = 25
lrate = 0.01
sgd = SGD(learning_rate=lrate, momentum=0.9, nesterov=False)
model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])

# 訓練模型
train_history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs,
          batch_size=32, verbose=2)
```

### 4\. 將訓練過程視覺化

我們定義一個輔助函數 `show_train_history` 來繪製訓練過程中的準確率和損失值變化。這對於診斷模型是否「過度擬合」或「擬合不足」至關重要。

```python
import matplotlib.pyplot as plt

def show_train_history(train_history,train,validation):
  plt.plot(train_history.history[train])
  plt.plot(train_history.history[validation])
  plt.title('Train history')
  plt.ylabel(train)
  plt.xlabel('epoch')
  plt.legend(['train','validation'],loc='upper left')
  plt.show()

# 繪製準確率 (Accuracy) 變化圖
show_train_history(train_history,'accuracy','val_accuracy')
# 繪製損失值 (Loss) 變化圖
show_train_history(train_history,'loss','val_loss')
```

### 5\. 最終效能評估

`model.evaluate()` 用於在「測試集」(模型從未見過的資料) 上評估模型的最終效能。

```python
# 評估模型在測試集上的表現
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1] * 100))
```

範例輸出：

```
313/313 ━━━━━━━━━━━━━━━━━━━━ 1s 2ms/step - accuracy: 0.6761 - loss: 1.3896
Accuracy: 67.42%
```

### 6\. 建立混淆矩陣 (Confusion Matrix)

混淆矩陣能讓我們深入了解模型在「哪些類別上做得好」以及「最常把 A 類別誤判成 B 類別」。

1.  `model.predict(X_test)`: 取得模型對測試集的原始預測 (10 個類別的機率)。
2.  `np.argmax(..., axis=1)`: 找出機率最高的索引 (0\~9)，將其轉換為最終的預測標籤。同時也將 One-Hot 編碼的真實標籤還原。
3.  `pd.crosstab`: 建立交叉列表 (即混淆矩陣)，比較「真實標籤」與「預測標籤」。



```python
import pandas as pd

# 取得預測標籤
prediction = model.predict(X_test)
prediction = np.argmax(prediction, axis=1)

# 還原真實標籤
y_test_label = np.argmax(y_test, axis=1)

# 建立並顯示混淆矩陣
pd.crosstab(y_test_label, prediction, rownames=['label'], colnames=['prediction'])
```

## 結果分析

### 最終準確率

模型在 10,000 筆測試資料上的最終準確率約為 **67.42%**。

### 混淆矩陣分析

以下是模型預測結果的混淆矩陣：

| prediction | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
| :--- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **label** | | | | | | | | | | |
| **0** | **769** | 22 | 38 | 18 | 9 | 5 | 9 | 6 | 72 | 52 |
| **1** | 13 | **825** | 5 | 16 | 5 | 4 | 2 | 5 | 39 | 86 |
| **2** | 98 | 12 | **509** | 101 | 95 | 63 | 53 | 33 | 24 | 12 |
| **3** | 46 | 27 | 67 | **535** | 51 | 141 | 54 | 30 | 30 | 19 |
| **4** | 33 | 3 | 66 | 94 | **594** | 53 | 45 | 88 | 16 | 8 |
| **5** | 23 | 5 | 64 | 239 | 37 | **544** | 23 | 35 | 14 | 16 |
| **6** | 10 | 13 | 43 | 101 | 52 | 32 | **720** | 14 | 10 | 5 |
| **7** | 21 | 6 | 46 | 60 | 50 | 62 | 6 | **709** | 7 | 33 |
| **8** | 91 | 41 | 13 | 14 | 6 | 3 | 4 | 2 | **793** | 33 |
| **9** | 38 | 119 | 9 | 29 | 5 | 9 | 2 | 9 | 36 | **744** |

**如何解讀：**

  * **對角線 (粗體)**: 代表**預測正確**的數量。例如，825 張「真實為 1」的圖片被「正確預測為 1」。
  * **非對角線**: 代表**預測錯誤**的數量。

**主要發現：**

  * **表現較好**：類別 1 (汽車) 和 8 (船) 的識別率最高 (825 和 793)。
  * **主要混淆點**：
      * **類別 3 (貓) vs 5 (狗)**: 239 張「真實為 5 (狗)」的圖片被誤判為 3 (貓)；141 張「真實為 3 (貓)」的圖片被誤判為 5 (狗)。
      * **類別 1 (汽車) vs 9 (卡車)**: 119 張「真實為 9 (卡車)」的圖片被誤判為 1 (汽車)。
      * **類別 2 (鳥)**: 該類別的正確率最低 (509)，且經常被誤判為 3 (貓) 和 4 (鹿)。

*(CIFAR-10 標籤: 0:飛機, 1:汽車, 2:鳥, 3:貓, 4:鹿, 5:狗, 6:青蛙, 7:馬, 8:船, 9:卡車)*
