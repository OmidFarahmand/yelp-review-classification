# Yelp Review Sentiment Classification – EECS 4412 Project Report

---

## 📌 Introduction & Objectives

The goal of this project was to classify Yelp reviews into three sentiment categories: **positive**, **negative**, and **neutral**. Two classification models were developed using different text representation techniques:

1. **Bag of Words (BoW)** – A traditional approach based on word frequency.
2. **Embeddings** – A more modern technique using sentence-level semantic representations.

The objective was to evaluate and compare the **accuracy** and **practicality** of each approach in the context of sentiment analysis.

---

## 🧾 Bag of Words (BoW) Approach

### 🔄 Preprocessing & Representation

Using the `TfidfVectorizer` from `scikit-learn`, raw Yelp reviews were converted into numerical feature vectors.

Steps included:
- **Stop Word Removal:** Words like “the”, “is”, “and” were removed to reduce noise.
- **TF-IDF Vectorization:** Emphasized words unique to each review while down-weighting common terms.
- **Feature Limitation:** Vocabulary limited to 20,000 most important words.
- **Feature Selection:** Top 10,000 features selected using a **Chi-squared (χ²)** test, improving accuracy and performance.

### 🧪 Model Selection & Results

We applied **5-fold cross-validation** to evaluate three classifiers:

| Classifier            | Accuracy | Std. Dev | Notes                     |
|-----------------------|----------|----------|---------------------------|
| Logistic Regression   | 84.73%   | ±0.19%   | Best performance overall  |
| Linear SVC            | 84.63%   | ±0.23%   | Slightly less stable      |
| Random Forest         | 80.74%   | Higher   | Weak performance on text  |

➡️ **Final Model:** Logistic Regression (`max_iter=1000`)

---

## 🤖 Embedding-Based Approach

### 🔄 Preprocessing & Representation

We used the `sentence-transformers` library and the **`all-MiniLM-L6-v2`** model to generate embeddings.

Steps included:
- **Sentence Embeddings:** Each review converted into a 384-dimensional vector.
- **Model Selection:** `all-MiniLM-L6-v2` chosen for speed and strong semantic representation.

Embeddings helped capture deeper contextual and semantic relationships between words that BoW might miss.

### 🧪 Model Selection & Results

Three classifiers were evaluated using **5-fold stratified cross-validation**:

| Classifier            | Accuracy | Std. Dev | Notes                        |
|-----------------------|----------|----------|------------------------------|
| Linear SVC            | 82.62%   | ±0.14%   | Most stable and accurate     |
| Logistic Regression   | 82.58%   | ±0.16%   | Very close second            |
| Random Forest         | 78.34%   | ±0.31%   | Weakest due to overfitting   |

➡️ **Final Model:** Linear SVC (`max_iter=2000`)

---

## 💬 Discussion & Conclusion

### Key Observations:

- **Accuracy:**  
  - BoW (84.73%) slightly outperformed the embedding model (82.62%)  
  - Yelp reviews are often repetitive and structured, favoring frequency-based approaches.

- **Computational Efficiency:**  
  - BoW was faster and more lightweight.  
  - Embedding methods, while slower, provided richer representations of semantic meaning.

- **Classifier Performance:**  
  - Random Forest consistently underperformed in both approaches due to its tendency to overfit sparse, high-dimensional text data.

### Final Takeaway:

While BoW remains a highly effective method for structured, review-style text classification, embeddings offer promising results for tasks requiring deeper semantic understanding. For real-world scenarios where nuance and context are key, embedding-based methods may provide more value despite the computational cost.
