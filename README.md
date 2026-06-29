# AI-Generated Reviews & Perceived Helpfulness
Code and analysis for the MSc Business Analytics dissertation:
>**"AI-generated reviews and perceived usefulness: the effects of polarity and length"**
> Elise BERGER - NEOMA Business School - MSc Business Analytics - 2026
> Supervisor: Diego NASCIMENTO


## Repository structure
    .
    ├── README.md
    ├── requirements.txt
    ├── .gitignore
    ├── 01_data_prep.ipynb        # main notebook (download + clean + merge + filter)
    ├── 02_pilot_test.ipynb       # test the dataset
    ├── 03_analysis.ipynb         # descriptives statistics + figures + regression
    ├── 04_run_roberta.py         # run the RoBERTa Model
    ├── data/                     # raw + processed data (not tracked — see "Data")
    ├── figures/                  # generated figures (.png, 300 dpi)
    └── outputs/                  # generated tables (.csv, .txt)


## Data
- **Source:** Amazon-Reviews-2023, McAuley Lab, Hugging Face
— **Category:** `All_Beauty`
- **Sample:** 701 528 raw reviews -> **700 808** after cleaning (removed empty text or missing rating)
- The data is **downloaded automatically** by the first step. Raw files are not committed to the repository (see `.gitignore`)

## Methodology summary
- **Dependent variable:** `helpful_vote` — number of "helpful" votes a review received from other users (proxy for perceived helpfulness).
- **Independent variable:** `AI_generated_score` — probability the review is AI-generated.
- **Moderators:** `rating` (polarity) and `text_length`.
- **Model:** `helpful_vote ~ AI_c * rating_c + AI_c * log_length_c` (continuous predictors are mean-centered).
- **Estimator:** Negative Binomial GLM, selected over Poisson and OLS because the dependent variable is an overdispersed count (dispersion statistic = 21.8).


## AI-generation model

- Model: `Hello-SimpleAI/chatgpt-detector-roberta`, a RoBERTa classifier fine-tuned on the HC3 corpus (Human–ChatGPT Comparison Corpus).
- **Label mapping verified:** `{0: 'Human', 1: 'ChatGPT'}` → the score uses index 1 = P(AI-generated), checked with an assertion at load time.
- Inference settings: `batch_size = 32`, `max_length = 256` tokens, checkpoint every 5,000 reviews, `random_state = 42`.
- Runs on Apple Silicon (MPS), NVIDIA (CUDA), or CPU (auto-detected).

## Key results

- A higher AI-generation probability **reduces** perceived helpfulness (about 19% fewer helpful votes).
- The penalty is **stronger for five-star reviews** and **for longer reviews**.
- All three hypotheses are **rejected**: the effects are statistically significant but run opposite to the predicted (positive) direction.

## Setup
Developed with Python 3.14.

## AI usage disclosure

A generative AI tool was used **only for language editing, proofreading, and code structuring**. It did not generate the core concepts, the analysis, or the scientific
arguments. See the dissertation's "Declaration on the Use of AI" for the full statement.
