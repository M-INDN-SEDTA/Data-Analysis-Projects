# Student Performance Prediction & Adaptive Learning System

### 1. **Project Title & Description**

* **Name:** `Student Performance Prediction and Adaptive Learning`

* **Type:** Data Analysis / Machine Learning / Educational Analytics Project

* **Description:**

> This project focuses on predicting student performance and creating an adaptive learning framework to enhance learning outcomes. Using Python for data preprocessing, feature engineering, and exploratory analysis, it implements machine learning models to predict performance and generate personalized learning suggestions. Insights are visualized using **Power BI dashboards** and summarized in detailed PDF reports.

* **Dataset:**
  Source: [https://www.deepdatalake.com/details.php?dataset_id=93](https://www.deepdatalake.com/details.php?dataset_id=93)
  (DeepDataLake)

* **Key Highlights:**

  * Integrated **data cleaning, feature extraction, and EDA** for robust analysis.
  * Built **machine learning models** for predicting performance.
  * Stored results in **SQL database** for further querying.
  * Developed **Power BI dashboards** for interactive visualization.
  * Delivered **static reports (PDF)** summarizing analysis and predictions.

---

### 2. **Project Structure (Tree View)**

```markdown
📂 Student_Performance_Prediction_and_Adaptive_Learning
├── datasets/
│   ├── student_performance.csv
│   └── student_performance_expanded_cols.csv
│
├── Power BI dax quries/                                             # DAX queries for Power BI dashboard
│
├── requirements.txt                                                 # Python dependencies
│
├── Student Performance Dashboard.pdf                                # PDF report of dashboard
├── Student Performance Dataset To SQL DB.ipynb                      # Notebook to load dataset into SQL DB
├── Student Performance Prediction & Adaptive Learning Report.pdf    # Full project report
├── Student Performance Prediction and Adaptive Learning.ipynb       # Jupyter Notebook
├── Student Performance Prediction and Adaptive Learning.py          # Python script version
├── Student Performance SQL Queries.sql                              # SQL queries for analysis
└── Student Performance.pbix                                         # Power BI dashboard
```

---

### 3. **Objective / Problem / Goal**

Improving student outcomes requires understanding the factors affecting performance and delivering personalized learning interventions.
The goal of this project is to **predict student performance** and develop an **adaptive learning system** to improve academic results.

---

### 4. **Data Source**

Dataset: **Student Performance Dataset**

* Contains student demographics, study habits, and exam scores.
* Features include: `gender`, `race/ethnicity`, `parental level of education`, `lunch`, `test preparation course`, and scores in `math`, `reading`, and `writing`.

---

### 5. **Data Cleaning & Preprocessing**

* Removed duplicates and null values.
* Converted categorical features into numerical encoding.
* Added engineered features such as:

  * `total_score`, `average_score`, `score_category`.
* Normalized numeric features for modeling.

---

### 6. **Exploratory Data Analysis (EDA)**

Key insights include:

* Performance variations based on **parental education level** and **test preparation course completion**.
* Score distribution analysis for different demographics.
* Correlation analysis between features and scores.

**Power BI Dashboards include:**

* Performance trends across demographics.
* Score distribution by category.
* Adaptive learning suggestions based on performance patterns.

---

### 7. **Modeling**

* **Feature Engineering** for richer insights.
* **Machine Learning Models** for performance prediction.
* Achieved robust accuracy in performance prediction.
* Models stored for later use in adaptive learning.

---

### 8. **SQL & Power BI Integration**

* Data loaded into a **SQL database** via Python scripts.
* Created SQL queries for performance analysis.
* Built **Power BI dashboards** using these queries and DAX formulas.
* Exported dashboard and reports for stakeholder use.

---

### 9. **Installation**

* Clone repo:

```bash
git clone https://github.com/M-INDN-SEDTA/Data-Analysis-Projects/tree/main/Student_Performance_Prediction_and_Adaptive_Learning
cd Student_Performance_Prediction_and_Adaptive_Learning
```

* Install dependencies:

```bash
pip install -r requirements.txt
```

* Run Jupyter Notebook:

```bash
jupyter notebook "Student Performance Prediction and Adaptive Learning.ipynb"
```
Note: If the above code give errors while executing please rename its filename to "Student_Performance_Prediction_and_Adaptive_Learning.ipynb"
and use this name instead.

---

### 10. **Visualization & Reporting**

* **Interactive Dashboards:** `Student Performance.pbix`
* **Static Dashboard Report:** `Student Performance Dashboard.pdf`
* **Full Analysis Report:** `Student Performance Prediction & Adaptive Learning Report.pdf`

---

### 11. **Future Scope**

* Deploy as a **web application** for real-time student performance predictions.
* Integrate with **learning management systems** for adaptive recommendations.
* Expand dataset with additional behavioral data.
* Implement advanced models such as **XGBoost, LightGBM, and Deep Learning**.
* Develop a **mobile app** for instant performance feedback.

