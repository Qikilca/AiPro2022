### 事前準備 ###
#
# AiPro2022フォルダに移動して…
#
# python -m venv .tfflow    ← 新しい仮想環境「.tfflow」を生成する
#
# .tfflow\Scripts\activate.bat    ← 新しい仮想環境「.tfflow」を起動する
#
# (.tfflow) > python -m pip install -r requirements.txt   ← ライブラリの一括インストール
# ※一括インストールは時間がかかりますので(10分～？）休み時間の前などに実行するとよいでしょう
#
# requirements.txt の内容
#   streamlit             ← 毎度、おなじみ
#   typing_extensions     ← これがないとエラーが出ることがあるので一応
#   numpy                 ← 毎度、おなじみ
#   pandas                ← 毎度、おなじみ
#   tensorflow-cpu        ← 深層学習用ライブラリ（容量の関係で今回はCPU版を採用）
#
# (.tfflow) > streamlit run testapp04.py    ← Webアプリ（testapp04.py）を起動

### Teachable Machineでモデルを作成する方法（中級以上） ###
#
#（Teachable Machineの詳細は、「AI理論」の授業で配布された「認識編Ⅰ」を参照）
# 
# ※推奨ブラウザ：Google Chrome（他のブラウザだと、ファイルのD&Dができないかも）
# 
# 1. Teachable Machineにアクセスする → 使ってみる → 画像プロジェクト → 標準の～
#   https://teachablemachine.withgoogle.com/
#   
# 2. 各Classの「アップロード」ボタンからデータを追加する
# 
# 3. トレーニングを実行する
#
# 4. モデルをエクスポートする　→　Tensorflowタブ → Keras → モデルをダウンロード
#
# 5. モデルのファイル（converted_keras.zip）がダウンロードされる
# 
# 6. converted_keras.zip に含まれる「keras_model.h5」を、
#    この testapp04.py ファイルと「同じフォルダ」に格納する
#
# 7. 準備OK！
# 
# （以後、モデルを作り直すたびに、エクスポート → .h5ファイルの上書きをしてください）


# ライブラリのインポート
from glob import glob
import imp
from pydoc import doc
# from turtle import width
import streamlit as st
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import os 


# 画像(img)が属するクラスを推論する関数（'weights_file'は、モデルのファイル名）
# 以下、Teachable Machineのエクスポート時に自動生成されるコードをコピペする(済)
def teachable_machine_classification(img, weights_file):

    # モデルの読み込み
    model = load_model(weights_file)

    # kerasモデルに投入するのに適した形状の配列を作成する。
    # 配列に入れることができる画像の「長さ」または枚数は
    # shapeタプルの最初の位置（この場合は1）で決まる。
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    # これを画像へのパスに置き換える
    # image = Image.open(img)
    image = img

    # Teachable Machineと同じ方法で、224x224にリサイズする。
    # 少なくとも224x224になるように画像をリサイズし、中心から切り取る。
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)

    # 画像をnumpyの配列に変換する
    image_array = np.asarray(image)

    # 画像の正規化
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

    # 画像を配列に読み込む
    data[0] = normalized_image_array

    # 推論を実行する
    prediction = model.predict(data)

    # 推論結果をメインモジュールに戻す
    return prediction.tolist()[0]


# import glob
# import random
# import os 
# import shutil
# import math

# 多かったデータセットのランダム抽出
# def random_sample_file():
#     catfiles = glob.glob("./dataset/images/cat" + '/*.jpg')
#     dogfiles = glob.glob("./dataset/images/dog" + '/*.jpg')
    
#     random_cat_file = random.sample(catfiles,2400)
#     random_dog_file = random.sample(dogfiles,2400)

#     os.makedirs("./dataset/images/choice/cat",exist_ok=True)
#     os.makedirs("./dataset/images/choice/dog",exist_ok=True)
    
#     for file in random_cat_file:
#         shutil.copy2(file,"./dataset/images/choice/cat")
        
#     for file in random_dog_file:
#         shutil.copy2(file,"./dataset/images/choice/dog")


    


# メインモジュール
def main():

    # タイトルの表示
    st.title("Image Classification with Google's Teachable Machine")

    # アップローダの作成
    uploaded_file = st.file_uploader("Choose a Image...", type="jpg")
    

    # 画像がアップロードされた場合...
    if uploaded_file is not None:
        
        # 画像を画面に表示
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        st.write("")
        st.write("Classifying...")

        # teachable_machine_classification関数に画像を引き渡してクラスを推論する
        prediction = teachable_machine_classification(image, 'keras_model.h5')        
        st.caption(f'推論結果：{prediction}番') # 戻り値の確認（デバッグ用）

        classNo = np.argmax(prediction)          # 一番確率の高いクラス番号を算出
        st.caption(f'判定結果：{classNo}番')      # 戻り値の確認（デバッグ用）

        # 推論の確率を小数点以下3桁で丸め×100(%に変換)
        pred0 = round(prediction[0],3) * 100  # 猫の確率(%)
        pred1 = round(prediction[1],3) * 100  # 犬の確率(%)
        pred2 = round(prediction[2],3) * 100  # アザラシの確率(%)

        # 推論で得られたクラス番号(初期値は0)によって出力結果を分岐
        if classNo == 0:
            st.subheader(f"これは{pred0}％の確率で「猫」です！")
        elif classNo == 1:
            st.subheader(f"これは{pred1}％の確率で「犬」です！")
        elif classNo == 2:
            st.subheader(f"これは{pred2}％の確率で「アザラシ」です！")


# mainの起動
if __name__ == "__main__":
    main()


#=============================================================================#
#
# 課題. 画像分類ツールの作成
#
# 初級（必須）：新しく仮想環境（.tfflow）を作成して、このプログラムを稼働させよう
#
# 中級（任意）：「犬」「猫」「アザラシ」を認識できるような新しいモデルを作成しよう
#
# 上級（任意）：画像認識を利用した、オリジナルのWebアプリを作りましょう（自由課題）
#              アイデア1) マニアックな代物を認識してくれる「◯◯鑑定士」アプリ
#              アイデア2) カメラで撮影した顔が、何の動物に似ているか判定するアプリ
#              ※Streamlitの各種API、コンポーネントなどは、自由に使って結構です
# 
# ※中級以上は未完成でも結構です。途中であれば、そこまでのチャレンジを評価いたします
#
# #-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-#
#
# 提出方法：以下の項目を記入の上、Teamsの課題機能でこのコードファイルを提出しましょう
#          中級以上の人は、モデルファイル（keras_model.h5）も一緒に提出してください
#          上級の人は、アプリの概要や使い方などを簡単に下のコメント欄にご記入ください
#
# 締め切り：6/14(火)… 今回は全員に初級を、そしてできれば中・上級に挑戦して欲しいので
#          締め切りを1週間先にしました。じっくりと時間をかけて、取り組んでください！
#
# 氏名：須藤水貴
# 学科：高度情報工学科
# 学年：4年
# コメント：イヌや猫は犬種をなるべく多くして学習させると精度が上がるイメージ。
#
#=============================================================================#
