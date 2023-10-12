from flask import Flask, request, render_template, redirect, url_for
import joblib
import pandas as pd
import numpy as np

from modules.dataPreprocess import DataCleansing

cleaner = DataCleansing()


app = Flask(__name__)

# 모델 로드
model = joblib.load('best_rf_model.pkl')  # 모델 파일 이름을 지정하세요

@app.route("/")
def index():
    df = pd.DataFrame()
    return render_template("index.html", df=df)

@app.route('/predict', methods=['POST'])
def predict():
    global df  # 전역 변수로 선언된 df 변수를 사용합니다.
    try:
        if request.method == 'POST':
            # 입력 데이터 받기
            data = request.form
            data = request.form.to_dict()  # 폼 데이터를 딕셔너리로 변환

            # 딕셔너리를 데이터 프레임으로 변환
            df = pd.DataFrame(data, index=[0])

            # 입력 데이터 전처리
            dfs = cleaner.preprocess(df)

            # 입력 데이터 처리 및 모델 예측
            prediction = model.predict(dfs)
            pred_proba =  np.round(model.predict_proba(dfs), 2)

            # 예측 결과를 HTML 템플릿에 전달
            pred_result = prediction
            print("⭐️ 예측 결과입니다.", pred_result, pred_proba)
            
            return render_template("index.html", df=df, prediction=pred_result, pred_proba=pred_proba)
        
        else:
            print("⭐️ 잘못된 접근입니다.")
            return render_template("index.html", df=None, prediction=None, pred_proba=None)
    except Exception as e:
        return render_template("index.html", prediction=f"오류: {str(e)}")

if __name__ == '__main__':
    # app.run(host='0.0.0.0',  port=5001)
    app.run(debug=True)