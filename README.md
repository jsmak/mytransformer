# 🚀 MyTransformer: Attention Is All You Need from Scratch

> *"Attention Is All You Need (Vaswani et al., 2017)"* 논문의 핵심 메커니즘을 바닥(Scratch)부터 단계별로 직접 구현하고, Multi30k 번역 데이터셋을 활용해 End-to-End 트랜스포머 모델을 학습/평가/BLEU 측정까지 실습하는 튜토리얼 저장소입니다.

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📌 주요 특징

- 🧩 **단계별 모듈 분해 구현**: Scaled Dot-Product Attention부터 Multi-Head Attention, Positional Encoding, LayerNorm, FeedForward, Encoder/Decoder 스택까지 한 단계씩 점진적으로 조립
- 🔍 **수식과 코드의 1:1 매핑**: 논문의 핵심 수식과 아키텍처 다이어그램(Figure 1, Figure 2)을 직접 파이토치 텐서 연산으로 시각화하며 검증
- 🌐 **실제 번역 모델 실습**: Multi30k (영어 $\rightarrow$ 독일어) 데이터셋 기반 토큰화, 어휘 사전 구축, 패딩/마스킹, Greedy Decoding 및 SacreBLEU 점수 평가

---

## 📂 프로젝트 구조

```
mytransformer/
├── notebooks/                                      # 단계별 실습 Jupyter Notebooks
│   ├── 01_Scaled_dot_product_attention_exercise.ipynb # [실습] Scaled Dot-Product Attention
│   ├── 02_Simple_Transformers.ipynb                # [기초] 심플 트랜스포머 원리
│   ├── 03_Attention_scale.ipynb                    # [이론] Attention Scaling factor (√d_k) 분석
│   ├── 04_Multihead_attention_step1.ipynb          # [구현] Multi-Head Attention (1단계: 차원 분할)
│   ├── 05_Multihead_attention_step2.ipynb          # [구현] Multi-Head Attention (2단계: 결합 및 Linear)
│   ├── 06_PositionalEncoding.ipynb                 # [구현] Positional Encoding (Sin/Cos 주기함수)
│   ├── 07_Layernorm.ipynb                          # [구현] Layer Normalization & Residual Connection
│   ├── 08_positionwisefeedForward.ipynb            # [구현] Position-wise Feed-Forward Network
│   ├── 09_Encoder.ipynb                            # [구현] EncoderLayer & Encoder 전체 스택
│   ├── 10_Decoder.ipynb                            # [구현] DecoderLayer & Masked Decoder 스택
│   ├── 11_Transformer.ipynb                        # [구현] Full Transformer 아키텍처 완성
│   ├── 99_MyTransformer.ipynb                      # [종합] Multi30k 영-독 번역 학습 & BLEU 평가
│   └── multi30k_data.pkl                           # Multi30k 번역 데이터셋 (~5.88MB)
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 🗺️ 단계별 학습 커리큘럼

| 번호 | 노트북 파일명 | 핵심 학습 내용 |
| :--- | :--- | :--- |
| **01** | [`01_Scaled_dot_product_attention_exercise.ipynb`](notebooks/01_Scaled_dot_product_attention_exercise.ipynb) | $Q, K, V$ 행렬곱, 마스킹, Scaling, Softmax 가중합 빈칸 채우기 실습 |
| **02** | [`02_Simple_Transformers.ipynb`](notebooks/02_Simple_Transformers.ipynb) | 트랜스포머 기본 메커니즘과 동작 흐름 이해 |
| **03** | [`03_Attention_scale.ipynb`](notebooks/03_Attention_scale.ipynb) | $\frac{1}{\sqrt{d_k}}$ 스케일링이 Softmax Gradient Vanishing을 방지하는 이유 |
| **04** | [`04_Multihead_attention_step1.ipynb`](notebooks/04_Multihead_attention_step1.ipynb) | Head 분할 구조 (`d_model` $\rightarrow$ `num_heads` $\times$ `d_head`) |
| **05** | [`05_Multihead_attention_step2.ipynb`](notebooks/05_Multihead_attention_step2.ipynb) | Multi-Head 출력 Concat 및 최종 Linear 사영(Projection) |
| **06** | [`06_PositionalEncoding.ipynb`](notebooks/06_PositionalEncoding.ipynb) | 순서 정보 부여를 위한 Sinusoidal Positional Encoding 구현 및 시각화 |
| **07** | [`07_Layernorm.ipynb`](notebooks/07_Layernorm.ipynb) | Layer Normalization 수식과 PyTorch 동작 원리 분석 |
| **08** | [`08_positionwisefeedForward.ipynb`](notebooks/08_positionwisefeedForward.ipynb) | Position-wise Feed-Forward Network ($d_{model} \rightarrow d_{ff} \rightarrow d_{model}$) |
| **09** | [`09_Encoder.ipynb`](notebooks/09_Encoder.ipynb) | Self-Attention + FFN + Residual/LayerNorm을 묶은 Encoder 모듈 완성 |
| **10** | [`10_Decoder.ipynb`](notebooks/10_Decoder.ipynb) | Masked Self-Attention + Cross-Attention (Encoder-Decoder) 구조 완성 |
| **11** | [`11_Transformer.ipynb`](notebooks/11_Transformer.ipynb) | 인코더-디코더 연결 및 최종 Vocabulary Linear Projection 완성 |
| **99** | [`99_MyTransformer.ipynb`](notebooks/99_MyTransformer.ipynb) | **[최종 프로젝트]** Multi30k 번역 학습, Loss 곡선, Greedy Decode, BLEU 측정 |

---

## 🏗️ Transformer 아키텍처 구조

```
                      [ Output Probabilities ]
                                 ▲
                          [ Linear Layer ]
                                 ▲
                       ┌───────────────────┐
                       │  Decoder Layer xN │
                       │ ┌───────────────┐ │
                       │ │  Feed Forward │ │
                       │ └───────▲───────┘ │
                       │ ┌───────┴───────┐ │
    ┌─────────────────►│ │Cross Attention│ │
    │                  │ └───────▲───────┘ │
┌───┴───────────────┐  │ ┌───────┴───────┐ │
│  Encoder Layer xN │  │ │Masked Self-Att│ │
│ ┌───────────────┐ │  │ └───────▲───────┘ │
│ │  Feed Forward │ │  └─────────┼─────────┘
│ └───────▲───────┘ │            │
│ ┌───────┴───────┐ │            │
│ │ Self-Attention│ │            │
│ └───────▲───────┘ │            │
└─────────┼─────────┘            │
          ▲                      ▲
[ Positional Encoding ]  [ Positional Encoding ]
          ▲                      ▲
  [ Input Embedding ]    [ Target Embedding ]
          ▲                      ▲
    [ Source Text ]        [ Target Text ]
```

---

## ⚡ 빠른 시작 (Quick Start)

### 1. 가상환경 및 패키지 설치

```bash
# 레포지토리 클론
git clone https://github.com/<YOUR_GITHUB_USERNAME>/mytransformer.git
cd mytransformer

# 가상환경 생성 (선택)
python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
# source venv/bin/activate

# 의존성 패키지 설치
pip install -r requirements.txt
```

### 2. Jupyter Notebook 실행

```bash
jupyter notebook notebooks/
```

- 단계별로 `01`번부터 `11`번까지 순서대로 실행하며 컴포넌트별 동작을 확인합니다.
- 마지막 `99_MyTransformer.ipynb`를 실행하여 직접 번역 모델을 학습시키고 평가 결과를 확인합니다.

---

## 🛠️ Requirements

- Python >= 3.9
- PyTorch >= 2.0.0
- NumPy >= 1.24.0
- Matplotlib >= 3.7.0
- SacreBLEU >= 2.3.0
- Jupyter >= 1.0.0

---

## 📜 References

- Vaswani, A., et al. (2017). *Attention Is All You Need*. Advances in Neural Information Processing Systems (NeurIPS 2017). [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
