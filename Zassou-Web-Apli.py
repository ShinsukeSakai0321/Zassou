from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import numpy as np
from keras.models import load_model
#from keras.applications.mobilenet import MobileNet,  decode_predictions
from keras.applications.vgg16 import preprocess_input, decode_predictions
from keras.preprocessing.image import img_to_array, load_img

# DB接続用のデータを設定
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)
# アップロード先のフォルダを指定
UPLOAD_FOLDER = './static/image/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
from keras.applications.vgg16 import VGG16
from keras.models import Model
def img_pred(img):
    # 保存したモデルをロード
    #pp_model = load_model('mobile_net_model.h5')
    base_model = VGG16(weights='imagenet')
    pp_model = Model(inputs=base_model.input, 
              outputs=base_model.get_layer('fc2').output)
    
    # 読み込んだ画像を行列に変換
    x = img_to_array(img)
    
    # 3次元を4次元に変換、入力画像は1枚なのでsamples=1
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    # Top2のクラスの予測
    fc2_ft = pp_model.predict(x)
    a_model = load_model('Zassou-vgg-Features.h5')
    # fc2_query200にテストデータの特徴量を保持していると仮定
    # テストデータの所属クラスID(0-19)
    a_pred = a_model.predict_classes(fc2_ft, batch_size=1, verbose=1)
    # テストデータのクラスへの所属確率
    a_pred_P = a_model.predict(fc2_ft, batch_size=1, verbose=0)
    # 20種類の植物クラス
    label = [
        'Butakusa','Dokudami','Enokorogusa', 'Hakobe',
        'Hamasuge','Himejion','Hotarubukuro','Hotokenoza','Katabami',
        'Kayatsurigusa','Ohishiba','OOarechinogiku', 'Seidakaawadachisou',  'Suberihiyu',
        'Sugina', 'Suzumenokatabira',  'Tanpopo'
        ]

    score = np.max(a_pred_P)
    pred_label = label[np.argmax(a_pred_P[0])]
    return score,pred_label


@app.route('/')
def index():
    return render_template('./flask_api_index.html')

@app.route('/result', methods=['POST'])
def result():
    # submitした画像が存在したら処理する
    if request.method=='POST':

        # ファイルを読み込む
        img_file = request.files['image']

        # ファイル名を取得する
        filename = secure_filename(img_file.filename)

        # 画像のアップロード先URLを生成する
        img_url = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        abs_url=os.path.abspath(img_url)

        # 画像をアップロード先に保存する
        img_file.save(img_url)

        # 画像の読み込み
        image_load = load_img(abs_url, target_size=(224,224))

        # クラスの予測をする関数の実行
        score,pred_label = img_pred(image_load)

        return render_template('flask_api_index.html', title='予想クラス', score=score, pred_label=pred_label,result_img=img_url,fname=filename)

if __name__ == '__main__':
    app.debug = True
    app.run(host='localhost', port=5000)