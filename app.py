from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from PIL import Image
import base64
import io
import torch
from torchvision import transforms

# Flaskアプリの初期化
app = Flask(__name__)
CORS(app)

# 学習済みモデルの読み込み
model_path = 'models/poop_classifier.pth'
model = torch.load(model_path, map_location=torch.device('cpu'))  # CPUで実行
model.eval()

# 推論用の画像前処理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

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
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')  # RGB形式に変換

        # 画像を前処理
        input_tensor = transform(image).unsqueeze(0)  # バッチ次元を追加

        # モデルで推論
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            poop_prob = probabilities[1].item()  # 💩クラスの確率

        # 結果を返す
        similarity = poop_prob * 100  # 確率を百分率に変換
        return jsonify({'similarity': round(similarity, 2)})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
