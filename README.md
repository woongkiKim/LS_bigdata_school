# 프로젝트명 : LS 빅데이터 스쿨 1기 학습자료

## 프로젝트 기간 : 23/10/10 ~ 23/11/14

## 목차

```
1. 기초 분류 모델

    1-1. 타이타닉 생존 예측 모델 (jupyter)

    1-2. 분석 보고서 템플릿 (ppt)

    1-3. 분석 보고서 샘플 (pdf)


2. 전력 사용량 예측 모델


3. 주조 공정 최적화 모델

```

# README 예시 자료

- https://www.makeareadme.com/
- https://www.readme-templates.com/

---

# 프로젝트명 : NLP, data cleansing 및 다중 분류 예측모델 research

    # ** created by : kwk ** #

## 목차

    1. FastText 모델 사용법
    2. peterNorvig 교수 영어 오타 교정기 사용법
    3. peterNorvig 교수 한국말 교정기 사용법 (주류 도메인 한정)기
    4. 편집거리를 이용한 업종명 교정기 사용법(C, G, I 한정)
    5. RNN을 이용한 업종 분류 모델 사용법

## 1. FastText 모델 사용법

#### 1-1. 목적 : FastText 라이브러리 및 여러 모델를 사용하여 주류 품목명 오타처리 가능성 여부 판별

#### 1-2. 정의

    - fastText는 Facebook의 AI 연구소에서 만든 단어 임베딩 및 텍스트 분류 학습을위한 라이브러리입니다.
    - 특정 도메인에 관련된 데이터만 가지고 학습 데이터를 만들면 성능이 더 좋아질 수 있음

#### 1-3. 실행환경

    + python 3.8.7 version (python 3.7.x version에서도 실행 가능)
    - fasttext==0.9.2
    - gensim==4.0.0
    - joblib==1.0.1
    - numpy==1.20.2
    - pandas==1.2.3
    - psutil==5.8.0
    - pybind11==2.6.2
    - python-dateutil==2.8.1
    - pytz==2021.1
    - scikit-learn==0.24.1
    - scipy==1.6.2
    - six==1.15.0
    - smart-open==4.2.0
    - soynlp==0.0.493
    - threadpoolctl==2.1.0

#### 1-4. 실행방법

    1. requirements 라이브러리 설치한다.
        > 명령어 : pip install -r requirements.txt

    2. main.py를 실행한다.
        > 명령어 : python main.py

    3. 안내문구와 노출되면 검색하고 싶은 주류 단어를 입력한다. ex) 첨이슬, 참이술
        > 하단 그림 자료 참조

    ✅ 가상환경 라이브러리 추출하기

    ```
    pip freeze > requirements.txt
    ```

#### 1-5. 결과화면 예시

#### 1-6. 분석 요약

    세금계산서 품목명의 오타를 자동 수정하기 위한 가능성 여부를 판단하고자함.

    주류 클라이언트가 집계를 원하는 주류 품목명 데이터를 학습시켜 오타와 가장 유사한 품목명을 제공함.

#### 1-7. 분석 프로세스

    1. 데이터 준비 : 주류 클라이언트가 집계를 원하는 품목명 데이터 (참이슬, 맥켈란, 조니워커 등등)
    2. [데이터전처리] 학습데이터 생성 : 주류 품목명 자모 분리 (안녕 --> ㅇㅏㄴㄴㅕㅇ) 및 생성
    3. 모델 학습 : 생성한 학습데이터를 fastText 모델에 학습
    4. 모델 사용 : 학습한 모델에 찾고싶은 단어를 입력 (main.py)
    5. 결과값 : 입력한 단어와 유사한 단어 10개 추출

#### 1-8. 코드설명

    + main.py : 주류 품목명 중 오타를 입력하면 오타를 수정한 값에 가장 가까운 단어 10개 추천됨.

#### 1-9. 참고자료

- FastTest 소개1 : https://lovit.github.io/nlp/representation/2018/10/22/fasttext_subword/
- FastTest 소개2 : https://inahjeon.github.io/fasttext/
- 한국어 학습된 모델 : https://fasttext.cc/docs/en/crawl-vectors.html

## 2. peterNorvig 모델 사용법

#### 2-1. 목적 : Peter Norvig 교수의 영문자 오타 교정기의 아이디어를 본다. (조건부확률)

#### 2-2. 정의

    - "학습데이터에 있는 단어"는 오타를 입력했을 때에 정답으로 도출될 "정답 후보군"이 됨

    - 기본 아이디어는 조건부 확률에 기반으로 "c"를 입력하려다가 "w"를 입력할 확률을 기반으로

    - 오타 영단어 입력시, 교정할 영단어로 "얼마나 이동하는 지" 연산하고 최소의 값에 해당하는 단어를 도출한다

#### 2-3. 실행환경

    + 위와 동일

#### 2-4. 실행방법

    1. requirements 라이브러리 설치한다.
        > 명령어 : pip install -r requirements.txt

    2. peterNorvig_main.py를 실행한다.
        > 명령어 : python peterNorvig_sample.py

    3. 안내문구와 노출되면 검색하고 싶은 오타인 영단어를 입력한다. ex) dding, crorrect
        > 하단 그림 자료 참조

#### 2-5. 결과화면 예시

#### 2-6. 분석 요약

    파이썬 기본 코드로만 영어 오탈자 교정기를 만들 수 있다.

    "c"를 입력하려다가 "w"를 입력하게 될 확률을 구하는 일명 "오류모델"을 구하는 확률이론 기반으로 작성되었다.

    조건부 확률 기반이므로 학습된 데이터에 없는 단어는 정답으로 교정될 수 없다.

    만일, 학습 데이터에 오타가 있다면 오타를 정답으로 교정될 위험이 있다.

#### 2-7. 분석 프로세스

    1. 데이터 준비 : 오타가 없는 '깨끗한' 학습 데이터를 준비한다 (big.txt)
    2. [데이터전처리] 학습데이터 생성 : 영어이므로 모든 문자를 소문자로 변경한
    3. 모델 학습 : 생성한 학습데이터를 모델에 학습한다.
    4. 모델 사용 : 학습한 모델에 오타인 영단어를 입력 (peterNorvig_sample.py)
    5. 결과값 : 오타를 교정한 정답을 도출한다.

#### 2-8. 코드설명

    + peterNorvig_sample.py : 오타인 영문자를 입력하면 교정한 정답이 나온다.

#### 2-9. 참고자료

- peter norvig 철자 교정기 한국말 자료 : http://theyearlyprophet.com/spell-correct.html
- 교정기 여러 자료 모음집 : https://wolfgarbe.medium.com/1000x-faster-spelling-correction-algorithm-2012-8701fcd87a5f
- symspllpy : https://americanopeople.tistory.com/349
