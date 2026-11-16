# T5-Based Review Augmentation for Cold-Start E-Commerce Recommendation

Undergraduate thesis (Skripsi), Informatics, Universitas Sebelas Maret, 2026
*Augmentasi Ulasan Berbasis T5 untuk Mengatasi Cold-Start pada Sistem Rekomendasi E-Commerce*

New products have no reviews, so review-based recommenders have nothing to learn from. This project fine-tunes **T5-Base** to write synthetic reviews from a product's metadata (title, features, description), then feeds those reviews to three review-based recommenders and measures how much they help on cold-start items.

## Results

Best configuration compared with the metadata-only baseline (Scenario A):

| Model | RMSE | MSE | NDCG@K | HR@K |
|---|---|---|---|---|
| DeepCoNN | +2.55% (worse) | +5.17% (worse) | **+54.01%** | **+109.76%** |
| NARRE | **−9.29%** | **−17.71%** | **+29.67%** | **+53.64%** |
| MPCN | **−21.97%** | **−39.12%** | **+57.11%** | **+100.00%** |

- Best rating accuracy: NARRE, T5 reviews only, K=20 (RMSE 1.1176, MSE 1.2490)
- Best ranking: NARRE, metadata + T5, K=20 (NDCG@K 0.0708); MPCN, T5 reviews only, K=20 (HR@K 0.1944)
- Synthetic review quality at K=5: BERTScore 0.8402, Distinct-2 0.3609
- Human evaluation (5 annotators, 60 samples): average 3.81 / 5; K=20 scored highest (4.03)

Augmentation consistently improved ranking for all three models. It improved rating error for NARRE and MPCN but not for DeepCoNN.

## Method

1. **Data**: Amazon Reviews 2023, Electronics category ([McAuley-Lab/Amazon-Reviews-2023](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023)). Items were split 80% warm / 20% cold (`random_state=42`).
2. **Review generation**: T5-Base fine-tuned on 50,000 metadata–review pairs from warm items (90/10 train/eval split, 3 epochs, lr 2e-5, batch 32, FP16, RTX 4090). Prompt: `generate review: title: … features: … description: …`. For each cold item, K ∈ {5, 10, 20} reviews were generated with top-p sampling.
3. **Recommenders**: DeepCoNN, NARRE and MPCN with FastText embeddings, tested in three input scenarios:
   - A: metadata only
   - B: T5 synthetic reviews only
   - C: metadata + T5 reviews
4. **Evaluation**: RMSE and MSE for rating prediction; NDCG@K and HR@K for ranking, using 1 relevant item vs. 99 negatives sampled from the cold-item pool. Text quality: BLEU, ROUGE-L, BERTScore, Distinct-1/2, plus human evaluation (relevance, fluency, factual consistency, sentiment match).

## Notebooks

| Notebook | Contents |
|---|---|
| [`01_t5_review_generation.ipynb`](notebooks/01_t5_review_generation.ipynb) | Download and clean the dataset, warm/cold split, fine-tune T5-Base, generate K=5/10/20 synthetic reviews for cold items |
| [`02_recommender_evaluation.ipynb`](notebooks/02_recommender_evaluation.ipynb) | Model architectures, loading trained weights, 1-vs-99 ranking set, final RMSE/MSE/NDCG@K/HR@K evaluation, ablation tables, metadata/DeepCoNN input validation |
| [`03_case_study_per_product.ipynb`](notebooks/03_case_study_per_product.ipynb) | Earlier run of the evaluation notebook plus per-product case studies: one cold item, its full metadata, the T5 reviews for each K, and every model's rating prediction for several users |

The notebooks were run on Vast.ai (notebook 01) and Kaggle (notebooks 02 and 03) and are kept as run, with their outputs. They expect the intermediate files (parquet data, vocabularies, embedding matrices and trained weights) at the paths used there; those files are too large for this repository.

## Tech stack

Python, PyTorch, Hugging Face Transformers and Datasets, pandas, NumPy, scikit-learn, FastText embeddings, evaluate / rouge_score / bert_score.

## Author

**Muhammad Aspa Zikra** · [GitHub](https://github.com/aspazikra) · [LinkedIn](https://www.linkedin.com/in/muhammad-aspa-zikra) · m.asfazikra@gmail.com
