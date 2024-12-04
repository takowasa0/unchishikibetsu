from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from PIL import Image
import base64
import io
import random  # 仮の💩類似度計算用

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return render_template('index.html')  # templates/index.htmlを提供

@app.route('/upload', methods=['POST'])
def upload_image():
    try:
        # 画像データを取得
        data = request.json
        image_data = data['image']

        # Base64デコードして画像に変換
        image_bytes = base64.b64decode(image_data.split(',')[1])
        image = Image.open(io.BytesIO(image_bytes))

        # 仮の判定処理（ランダムな類似度を返す）
        similarity = random.uniform(0, 100)

        # 結果を返す
        return jsonify({'similarity': round(similarity, 2)})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
