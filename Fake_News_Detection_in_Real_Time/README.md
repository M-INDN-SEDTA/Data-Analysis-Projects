# Fake News Detection in Real Time

### 1. **Project Title & Description**

* **Name:** `Fake_News_Detection_in_Real_Time`

* **Type:** Data Analysis / Machine Learning Project

* **Description:**

  > This project focuses on analyzing, detecting, and predicting Fake vs. True news articles. Using Python for data preprocessing, feature engineering, and exploratory analysis, it implements **TF-IDF + Logistic Regression** for classification. Insights are further visualized using **Power BI dashboards** and summarized in detailed PDF reports for stakeholders.

* **Dataset:** https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets/data

* **Key Highlights:**

  * Integrated **data cleaning, feature extraction, and EDA** for robust analysis.
  * Implemented **TF-IDF Vectorizer + Logistic Regression** to detect fake news.
  * Stored results in **MySQL database** with SQL queries for aggregation.
  * Developed **interactive Power BI dashboards** to visualize trends.
  * Delivered **static reports (PDF)** summarizing findings.

---

### 2. **Project Structure (Tree View)**

```markdown
📂 Fake_News_Detection_in_Real_Time
├── datasets/                                       # Raw & processed datasets
│   ├── Fake.csv                                    # Raw Fake news data
│   ├── True.csv                                    # Raw True news data
│   ├── true_and_fake_news_clean.csv                # Cleaned dataset
│   └── true_and_fake_news_clean_with_features.csv  # Feature-engineered dataset
│
├── model/                                          # Trained ML models
│   ├── fake_news_model.pkl                         # Logistic Regression model
│   └── tfidf_vectorizer.pkl                        # Saved TF-IDF vectorizer
│
├── Fake_News_Detection_in_Real_Time.ipynb          # Jupyter Notebook (EDA + ML)
├── Fake_News_Detection_in_Real_Time.py             # Python script version
├── Fake_News_Detection_in_Real_Time_Prediction.py  # Prediction script (loads model)
│
├── Fake_News_Detection_in_Real_Time.pbix           # Power BI Dashboard
├── Fake_News_Detection_in_Real_Time_dashboard.pdf  # Exported dashboard (PDF)
├── Fake_News_Detection_in_Real_Time_Report.pdf     # Detailed analysis report
│
├── Fake_News_Detection_in_Real_Time_SQL_Queries.sql # SQL queries for dataset
├── load_news_dataset_to_mysql_database.ipynb       # Load CSV → MySQL (Notebook)
├── load_news_dataset_to_mysql_database.py          # Load CSV → MySQL (Script)
├── Power_Bi_Dax_Queries.txt                        # DAX measures for dashboard
└── requirements.txt                                # Python dependencies
```

---

### 3. **Objective / Problem / Goal**

Fake news can mislead the public and spread rapidly across social media.
The goal of this project is to **detect Fake vs. True news articles** using ML, analyze linguistic/textual patterns, and visualize insights via **dashboards and reports**.

---

### 4. **Data Source**

Dataset: **Kaggle – Fake and Real News Dataset**

* Contains \~45k news articles across multiple subjects.
* Each article has fields: `title`, `text`, `subject`, `date`, and `label` (Fake/True).

---

### 5. **Data Cleaning & Preprocessing**

* Removed nulls, duplicates, and invalid entries.
* Converted `date` to datetime (extracted month/year).
* Created new features:

  * `word_count`, `char_count`, `avg_word_length`, `polarity`.
  * Cleaned text (lowercase, removed stopwords, punctuation).
* Encoded labels → `1 = True`, `0 = Fake`.

---

### 6. **Exploratory Data Analysis (EDA)**

* **Class Balance:** Fake (52%) vs. True (48%) – nearly balanced.
* **Top Subjects:** `politicsNews`, `worldnews`, `News`, `politics`.
* **Word Stats:** True articles slightly longer than Fake.
* **Sentiment:** Both neutral (\~0.05 avg polarity).
* **Vocabulary:** \~72k unique words each; 39k unique per class.
* **Frequent Words:**

  * True → “said”, “reuters”, “state”, “government”
  * Fake → “trump”, “obama”, “clinton”, “american”

**Dashboards (Power BI):**

* Monthly distribution of Fake vs. True news.
* Subject-wise proportions.
* Trend analysis with time dimension.
* True vs. Fake by word length & polarity.

---

### 7. **Modeling**

* **TF-IDF Vectorizer** for feature extraction.
* **Logistic Regression** chosen as final model.
* Achieved **balanced performance** in detecting Fake vs. True.
* Models stored:

  * `fake_news_model.pkl`
  * `tfidf_vectorizer.pkl`

---

### 8. **SQL & Power BI Integration**

* Data loaded into **MySQL** using Python scripts.
* Wrote **SQL queries** for aggregations (subject-wise, monthly).
* Built **Power BI dashboard** using DAX measures.
* Exported `.pbix` and `.pdf` for reporting.

---

### 9. **Installation**

* Clone repo:

```bash
git clone https://github.com/M-INDN-SEDTA/Data-Analysis-Projects/tree/main/Fake_News_Detection_in_Real_Time
cd Fake_News_Detection_in_Real_Time
```

* Install dependencies:

```bash
pip install -r requirements.txt
```

* Run Jupyter Notebook:

```bash
jupyter notebook Fake_News_Detection_in_Real_Time.ipynb
```

---

### 10. **Visualization & Reporting**

* **Interactive Dashboards:** `Fake_News_Detection_in_Real_Time.pbix`
* **Static Dashboard Report:** `Fake_News_Detection_in_Real_Time_dashboard.pdf`
* **Full Analysis Report:** `Fake_News_Detection_in_Real_Time_Report.pdf`

---

### 11. **Future Scope**

* Deploy trained model as a **web app (Flask/FastAPI/Streamlit)**.
* Add **real-time scraping** of news + live predictions.
* Improve accuracy with **deep learning (LSTM, BERT, RoBERTa)**.
* Integrate with **browser plugin** to flag fake news instantly.

