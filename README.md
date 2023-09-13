# Note

[ACM Computing Surveys, Vol. 55, No. 9, Article 195. Publication date: January 2023](https://dl.acm.org/doi/full/10.1145/3560815)

논문 제목: Pre-train, Prompt, and Predict: A Systematic Survey of Prompting Methods in Natural Language Processing

을 읽고 정리 및 요약한 내용입니다.

# 목차

1. Sketch(전통적인 지도 학습 vs 프롬프트 기반 모델)
2. Four Paradigms of NLP Progress
    - Architecture Engineering
    - Objective Engineering
    - **Prompt Engineering**
3. Prompting 정의 및 용어 정리
4. 프롬프트 종류
5. 프롬프트 디자인 고려사항
    - 사전학습 모델 선택
    - Prompt Engineering
    - Answer Engineering
    - Expanding the Paradigm
    - 프롬프트를 통한 학습 전략


# 내용

## 1. Sketch(전통적인 지도 학습 vs 프롬프트 기반 모델)

1. Traditional Supervised learning

<img src="./assets/sp_learning.png">

**기존의 NLP task의 경우 사전학습 된 모델을 feature 추출기로 사용하고 해당 task에 맞게 fine-tuning(미세조정)**

2. Prompt-base model

<img src="./assets/prompt_learning.png">

e. g., 

input: 뉴스 기사

Template: lambda x: f"다음 뉴스 기사 요약해줘. {x}"

x': Template에 input(뉴스기사)를 적용한 자연어

$\hat{x}$: x’ + 요약된 뉴스 기사$

y: 요약된 뉴스 기사

**즉, NLP task를 next token 예측(e. g., GPT) 또는 빈칸 맞추기(e. g., BERT)와 같은 언어모델(LM) task로 재정의**

장점

1) 사전학습된 언어모델(LM)을 feature 추출기로만 쓰는 것이 아니라 직접 task를 해결하기 위한 모델로 적용

## 2. Four Paradigms of NLP Progress

### 1 - Feature Engineering

Paradigm: 완전 지도 학습(인공 신경망 사용 X)

Time Period: 2015년 쯤에 가장 유행

특징:

1) 인공 신경망을 주로 사용하지 않음

2) 개발자가 직접 feature 추출하는 것이 필요

예시:

1) Manual features →Linear or SVM

2) Manual features → Conditional Random Fields(CRF)

### 2 - Architecture Engineering

Paradigm: 지도 학습(인공신경망 이용)

Time Period: 대략 2013 ~ 2018년

특징:

1) 인공 신경망에 의존

2) 개발자가 직접 feature 추출하진 않지만 network 구조를 변경해야 됨(e. g.: LSTM, CNN)

3) 때때로 사전학습된 LM 사용하지만 Embedding과 같은 얕은 feature를 주로 이용

예시:

1) CNN을 이용한 Text 분류

### 3 - Objective Engineering

Paradigm: Pre-train, Fine-Tune

Time Period: 대략 2017년 이후부터 현재까지

특징:

1) 다량의 데이터로 사전학습된 언어모델(LM)을 이용하여 feature 추출

2) model 구조에 대한 연구는 이전에 비해 줄었지만 task에 적합하게 engineering 하는 것은 필요

예시:

1) BERT → Fine Tuning

### 4 - Prompt Engineering

Paradigm: Pre-train, Prompt, Predict

Time Period: 대략 2019년 이후부터 현재까지

특징:

1) NLP task를 언어모델(LM)로 재정의

2) LM을 통해 feature 추출, 그리고 예측 모두 진행

3) Prompt Engineering 필요

예시:

1) GPT3

## Prompt 정의
$f_{prompt}(\cdot)$: input string x를 prompt x'으로 변환해주는 함수

- Template

input string x를 위한 빈칸 \[x]와 정답 빈칸 \[z]를 변수로 취급하여 Prompt를 만들어내는 양식
```
e. g., Template: “[x]의 품사는 무엇입니까?: [z]”
    [x]: 뛰어가다
    [z]: 동사
```

- Prompt 형식
    1. Cloze prompt
        
        정답 빈칸 [z]가 template 사이에 존재하는 경우
        
        e. g., [x]는 매우 [z]한 영화다.
        
    2. Prefix prompt
        
        정답 빈칸 [z]가 template 마지막에 존재하는 경우
        
        e. g., [x]를 영어로 번역하시오. [z]
        
- Answer search
    
    정답 빈칸으로 가능한 후보군에 대한 정의를 뜻합니다.
    
    정답 후보군은 모든 토큰(Vocab)일 수 있고 Task에 따라 일부로 제한될 수 있습니다.
    
    Answer search가 필요한 이유: 기존 NLP task는 얇은 Network를 새로 정의하여 task에 맞게 mapping이 이루어져있는 반면 프롬프트 방식의 경우 사전학습된 언어모델을 그대로 이용하기 때문에 모든 토큰(Vocab)에 대해 확률이 부여되기 때문입니다.
    
    e. g., 후보군 Z = {”excellent”, “good”, “ok”, “bad”, “horrible”}
    
    여기서 후보군 Z 중에서 Label에 mapping 시켜야하는 문제가 존재합니다.
    
    예를 들어 기존 감정 분류 task의 경우 가능한 label이 {”Positive”, “Negative”}만 존재한다고 할 때,
    
    {”excellent”, “good”, “ok”} → “Positive”
    
    {“bad”, “horrible”} → “Negative”
    
    로 전환해주는 작업이 필요합니다
    
    따라서 Prompt를 채우는 함수는 prompt x’과 정답 후보군 Z가 필요합니다
    
    filled prompt: $f_{fill}(x’, Z)$

    filled prompt함수를 통해 후보군 중에서 가장 가능성이 높은 후보($\hat{z}$) 선택
    $$\hat{z}=search_{z \inZ} p(f_{fill}(x', z)) $$