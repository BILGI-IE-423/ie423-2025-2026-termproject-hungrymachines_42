# IE 423 Term Project Proposal

## Team Information

* Bersu Yılmaz - 123203069

* Emirhan Karaca - 122203009

* Mert Ada Demirbaş - 123203026
  

## Dataset Description

We use the IMDB Dataset of 50K Movie Reviews, obtained from [Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews).

This dataset contains 50,000 movie reviews labeled for binary sentiment classification (positive and negative), providing a balanced dataset for text analysis and machine learning applications.

Analyzing the relationship between document length and sentiment is an important step in exploratory data analysis and feature engineering. This study examines whether word count has predictive value and whether variations in review length introduce bias in classification models, potentially associating longer texts with negative sentiment.

The IMDB dataset provides a suitable basis for this analysis, enabling us to evaluate whether review length should be used as a feature to improve model performance or normalized to reduce potential bias.


## Dataset Access and Location
The dataset is stored in: data/raw/IMDB Dataset.csv

If the dataset needs to be downloaded manually, it is available at:

https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

After downloading, place the file inside:

data/raw/

## Research Questions

### Research Question 1 
Can traditional machine learning algorithms reliably categorize extreme sentiments to automate initial quality control in customer feedback loops?

**Explanation:** 
In large-scale customer feedback systems, quickly identifying strongly positive or negative opinions is crucial for effective decision-making and prioritization. This study examines whether traditional machine learning models can serve as practical tools for automating early-stage quality control by focusing on extreme sentiment detection. The key objective is to evaluate whether simpler and more interpretable models are sufficient for identifying highly polarized feedback in real-world scenarios. The IMDB 50K dataset provides a suitable test environment, as it includes diverse reviews with varying emotional intensity, enabling a realistic assessment of model performance.

### Research Question 2
Is there a statistically significant correlation between the review length (word count) and the sentiment polarity, indicating whether dissatisfied viewers write more exhaustive reviews?

**Explanation:**
Analyzing the relationship between document length and class labels is a fundamental step in exploratory data analysis (EDA) and feature engineering. This inquiry investigates whether simple structural metadata (word count) holds predictive power for the target variable (sentiment), and whether varying text lengths introduce a distribution bias that could cause classification models to essentially associate verbosity with the negative class. The IMDB dataset is ideal for evaluating this dynamic; its 50,000 labeled reviews provide the necessary volume to test if document length should be utilized as an engineered feature to improve model accuracy, or if length normalization is required to prevent algorithmic bias.

### Research Question 3
Are misclassified reviews concentrated in specific structural or behavioral patterns, such as review length, use of rare words, or emotionally mixed content?

**Explanation:**
Systematic error analysis is essential in the machine learning lifecycle, as it helps identify model weaknesses beyond overall accuracy. By examining misclassified reviews, this study explores how traditional models handle complex text structures, including rare words and emotionally mixed content. The IMDB dataset provides a suitable setting due to its diverse vocabulary and varying review lengths, allowing clear identification of False Positives and False Negatives. Analyzing these errors will help reveal patterns that negatively affect model performance and guide future improvements.

## Project Proposal
The primary objective of this project is to evaluate the effectiveness of traditional machine learning methods in sentiment analysis, with a focus on detecting extreme opinions and understanding factors that influence model performance.

The study will begin with data preprocessing, including text cleaning, normalization, and transformation into numerical representations such as Bag-of-Words and TF-IDF. This will be followed by exploratory data analysis to examine sentiment distribution, review length patterns, and potential relationships between textual features and sentiment polarity.

In line with the research questions, models such as Logistic Regression, Naive Bayes, and Support Vector Machines which may be implemented. Additionally, statistical analyses will be conducted to evaluate the relationship between review length and sentiment, alongside error analysis to identify patterns in misclassified reviews.

The project aims not only to achieve reliable classification performance but also to generate interpretable insights, particularly regarding the usability of simpler models for detecting extreme sentiments in practical applications.

Potential challenges include high-dimensional text data, bias related to review length, and difficulties in handling ambiguous or mixed sentiments.

## Initial Outputs

## Reproducibility Instructions

### 1. Clone the repository
```text
git clone [https://github.com/BILGI-IE-423/ie423-2025-2026-termproject-hungrymachines_42]
cd [ie423-2025-2026-termproject-hungrymachines_42]
```

### 2. Install required packages
```text
pip install -r requirements.txt
```

### 3. Place the dataset
```text
Download the dataset from:

https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

Put the dataset file inside:
```text
data/raw/
```


### 4. Run the scripts
```text
python scripts/01_load_data.py
python scripts/02_preprocess_data.py
python scripts/03_basic_eda.py
```
