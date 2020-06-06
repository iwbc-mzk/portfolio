import numpy as np
import pickle
import datetime
from tqdm import tqdm
from PIL import Image
from tensorflow.python.keras.datasets import mnist


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))


class NumberClassifier:
    def __init__(self, eta=0.1, epochs=10, batch_size=100, random_state=1):
        self.eta = eta
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state

        self._rgen = np.random.RandomState(self.random_state)

        self._hidden_node = 128

        self._w1 = None
        self._w2 = None
        self.is_weight_initialized = False

        self.error_norm_list = []

    def _shuffle(self, training_data, target_data):
        r = self._rgen.permutation(len(target_data))
        return training_data[r], target_data[r]

    def _initialize_weights(self, input_col, output_row):
        self._w1 = self._rgen.normal(loc=0.0, scale=0.01, size=(input_col + 1, self._hidden_node))
        self._w2 = self._rgen.normal(loc=0.0, scale=0.01, size=(self._hidden_node, output_row + 1))
        self.is_weight_initialized = True

    def fit(self, training_data, target_data):
        # ミニバッチ単位で学習する
        # i+self.batch_sizeがlen(target_data)を超えていても正常に動作する
        for _ in tqdm(range(self.epochs)):
            for i in range(0, len(target_data), self.batch_size):
                last_index = i + self.batch_size if i + self.batch_size <= training_data.shape[0] else \
                training_data.shape[0]
                batch_train_data = training_data[i:last_index, :]
                batch_target_data = target_data[i:last_index, :]

                # 重みの初期化
                if not self.is_weight_initialized:
                    self._initialize_weights(batch_train_data.shape[1], batch_target_data.shape[1])

                training_data_shuffled, target_data_shuffled = self._shuffle(batch_train_data, batch_target_data)

                # バイアス用の列を追加
                training_data_shuffled = np.concatenate([np.ones((training_data_shuffled.shape[0], 1)),
                                                         training_data_shuffled], axis=1)
                target_data_shuffled = np.concatenate([np.ones((target_data_shuffled.shape[0], 1)),
                                                       target_data_shuffled], axis=1)

                error_norm = self._update_weight(training_data_shuffled, target_data_shuffled)

            # 誤差のノルムをリストに格納
            self.error_norm_list.append(error_norm)

        return self

    def _update_weight(self, training_data, target_data):
        # 隠れ層の計算
        first_input = np.dot(training_data, self._w1)
        first_activation = self.activation(first_input)

        # 出力層の計算
        second_input = np.dot(first_activation, self._w2)
        second_activation = self.activation(second_input)

        d_error = second_activation - target_data
        d_second_activation = d_error * self.activation_prime(second_input)
        dw2 = np.dot(first_activation.T, d_second_activation)

        d_second_input = np.dot(d_second_activation, self._w2.T)
        d_first_activation = d_second_input * self.activation_prime(first_input)
        dw1 = np.dot(training_data.T, d_first_activation)

        self._w1 -= self.eta * dw1
        self._w2 -= self.eta * dw2

        error_norm = np.linalg.norm(d_error)
        return error_norm

    def activation(self, X):
        return sigmoid(X)

    def activation_prime(self, X):
        return sigmoid_prime(X)

    def predict(self, test_data):
        result = None
        for i in range(0, test_data.shape[0], self.batch_size):
            last_index = i + self.batch_size if i + self.batch_size <= test_data.shape[0] else test_data.shape[0]
            batch_test_data = test_data[i:last_index, :]
            batch_test_data = np.concatenate([np.ones((batch_test_data.shape[0], 1)), batch_test_data], axis=1)

            first_input = np.dot(batch_test_data, self._w1)
            first_activation = sigmoid(first_input)

            second_input = np.dot(first_activation, self._w2)
            second_activation = sigmoid(second_input)

            if result is None:
                result = np.copy(second_activation[:, 1:])
            else:
                result = np.concatenate([result, second_activation[:, 1:]])

        return result

    def save(self, filename):
        with open(filename, mode='wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        with open(filename, mode='rb') as f:
            return CustomUnpicker(f).load()


class CustomUnpicker(pickle.Unpickler):
    """
    Pickleでモデルデータを読み込む際のエラー回避用
    参考: https://stackoverflow.com/questions/27732354/unable-to-load-files-using-pickle-and-multiple-modules
    """
    def find_class(self, module, name):
        if name == 'NumberClassifier':
            return NumberClassifier

        return super().find_class(module, name)


def image_preprocess(input_data):
    return np.array([data.reshape(28*28) for data in input_data])/255


def label_preprocess(output_data):
    output_reshaped = output_data.reshape(output_data.shape[0], 1)
    output_numbered = np.array([[int(label == num) for num in range(10)] for label in output_reshaped])
    return output_numbered


def learn(images, labels):
    preprocessed_images = image_preprocess(images)
    preprocessed_labels = label_preprocess(labels)

    model = NumberClassifier(eta=0.001, epochs=500)
    model.fit(preprocessed_images, preprocessed_labels)

    NumberClassifier.__module__ = "discriminator"

    # 学習済モデルの保存
    model.save('model.pickle')


def test(images, labels):
    preprocessed_images = image_preprocess(images)

    dump_name = 'model.pickle'
    with open(dump_name, mode='rb') as f:
        model = pickle.load(f)
        result = model.predict(preprocessed_images)

    # 各行で最も値が大きい列のインデックスを取得
    result = np.argmax(result, axis=1)

    show_data = labels - result
    show_data = np.where(show_data != 0, 1, 0)

    correct_rate = (show_data.shape[0] - np.sum(show_data)) / show_data.shape[0]
    print(f'正答率: {round(correct_rate*100, 2)}%')


def recognize(image):
    """
    画像を認識させて結果を返す
    :param image:file_path, file_object (pillow Image.opne関数参照)
    :return: int型
    """
    image_data = Image.open(image)
    input_data = preprocess(image_data)
    model = NumberClassifier.load('number_classifier/model.pickle')
    result = np.argmax(model.predict(input_data), axis=1)[0]

    return result


def preprocess(image):
    """
    画像データの前処理を行う
    1. グレースケール化
    2. 上下左右の空白部分を取り除く
    3. 20×20pxにリサイズ
    4. 重心が中心に来るように28×28pxにリサイズ
    5. 各要素が 0~1 になるように255で除算
    6. 1×784の行列に整形
    :param image: 前処理を行う pillowで読み込んだ画像データ
    :return: 前処理後の画像データ
    """
    # グレースケール化
    if image.mode == 'RGB' or image.mode == 'RGBA':
        image_grayscale = image.convert('L')
        image_grayscale = Image.fromarray(255 - np.asarray(image_grayscale), mode='L')
    else:
        image_grayscale = image

    # 空白マージンを取り除く
    mt, mb, ml, mr = get_image_margin(image_grayscale)
    image_np = np.asarray(image_grayscale)
    row, column = image_np.shape
    image_np = image_np[mt: row - mb, ml:column - mr]
    image_margin_removed = Image.fromarray(image_np, mode='L')

    # 20×20pxに収まるようにリサイズ
    width, height = image_margin_removed.size
    resize_width, resize_height = (20, int(20*(height/width))) if width >= height else (int(20*(width/height)), 20)
    image_resize = image_margin_removed.resize((resize_width, resize_height))

    # 中心に画像の重心が来るようにして28×28pxにリサイズ
    x_com, y_com = calc_center_of_mass(np.array(image_resize))
    x_com = int(round(x_com, 0))
    y_com = int(round(y_com, 0))
    img_ret = Image.new('L', (28, 28))
    img_ret.paste(image_resize, (int((28-resize_width)/2) - x_com, int((28-resize_height)/2) - y_com))

    # 正規化
    img_norm = np.true_divide(np.array(img_ret), 255)
    img_reshape = img_norm.reshape(1, 28*28)

    return img_reshape


def get_image_margin(image):
    """
    イメージのマージンを取得する。
    上下左右端から最初に値が0より大きくなるセルを探し、端からそのセルまでのセル数をマージンとして返す。
    :param image:
    :return: tuple (margin_top, margin_bottom, margin_left, margin_right)
    """
    image_np = np.asarray(image)

    row_sum = image_np.sum(axis=1)
    margin_top = get_first_non_zero_index(row_sum)
    margin_top = margin_top if margin_top else 0
    margin_bottom = get_first_non_zero_index(np.flipud(row_sum))
    margin_bottom = margin_bottom if margin_bottom else 0

    row_column = image_np.sum(axis=0)
    margin_left = get_first_non_zero_index(row_column)
    margin_left = margin_left if margin_left else 0
    margin_right = get_first_non_zero_index(np.flipud(row_column))
    margin_right = margin_right if margin_right else 0

    return margin_top, margin_bottom, margin_left, margin_right


def get_first_non_zero_index(array):
    for idx, i in enumerate(array):
        if i > 0:
            return idx
    return None


def calc_center_of_mass(matrix):
    """
    行列の'中心から'の重心を計算する
    :param matrix: 行列
    :return: 座標の中心を原点とした重心の座標 (x_coordinate, y_coordinate)
             正の方向は行列のインデックスが増加する方向
        例: 4×4行列で1行目２列目の位置が重心の場合の結果は (-1, -2)
    """
    row, column = matrix.shape

    # 1行が ..., -1.5, -0.5, 0.5, 1.5, ...の行列を用意し座標代わりにする
    x_tile = np.arange(column) - (column-1)/2
    x_coordinate = np.tile(x_tile, [row, 1])
    y_tile = np.arange(row) - (row-1)/2
    y_coordinate = np.tile(y_tile, [column, 1]).T

    mat_sum = np.sum(matrix)

    x_com = np.sum(matrix*x_coordinate)/mat_sum
    y_com = np.sum(matrix*y_coordinate)/mat_sum

    return x_com, y_com


def main():
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    start_time = datetime.datetime.now()

    image_data = image_preprocess(train_images)
    label_data = label_preprocess(train_labels)

    model = NumberClassifier(eta=0.001, epochs=500)
    model.fit(image_data, label_data)

    # 学習済モデルの保存
    model.save('model.pickle')

    end_time = datetime.datetime.now()
    elapsed_time = end_time - start_time
    elapsed_time_min = round(elapsed_time.total_seconds() // 60)
    elapsed_time_sec = round(elapsed_time.total_seconds() % 60)
    print(f'実行時間: {elapsed_time_min}分{elapsed_time_sec}秒')

    test(test_images, test_labels)


if __name__ == '__main__':
    main()
