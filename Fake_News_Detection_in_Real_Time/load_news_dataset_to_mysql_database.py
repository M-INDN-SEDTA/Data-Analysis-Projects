import pandas as pd
import mysql.connector
from dotenv import load_dotenv
import os

# 1. Load .env credentials
load_dotenv(override=True)

username = os.getenv("DB_USERNAME")
password = os.getenv("DB_PASSWORD")
host = "localhost"
port = 3306
database = "true_and_fake_news_detection_db"

# 2. Load cleaned CSV file
csv_path = "./datasets/true_and_fake_news_clean_with_features.csv"
df = pd.read_csv(csv_path)
print("CSV loaded with shape:", df.shape)

# 3. Connect to MySQL
conn = mysql.connector.connect(
    host=host,
    user=username,
    password=password,
    database=database,
    port=port
)
cursor = conn.cursor()

# 4. Create table if not exists
cursor.execute("""
CREATE TABLE IF NOT EXISTS news_articles (
    id INT AUTO_INCREMENT PRIMARY KEY,
    title TEXT,
    text LONGTEXT,
    subject VARCHAR(255),
    date DATE,
    label INT,
    clean_text LONGTEXT,
    label_str VARCHAR(10),
    word_count INT,
    char_count INT,
    avg_word_length FLOAT,
    sentence_count INT,
    polarity FLOAT
)
""")

# 5. Prepare insert statement
sql = """
INSERT IGNORE INTO news_articles (
    title, text, subject, date, label,
    clean_text, label_str, word_count, char_count,
    avg_word_length, sentence_count, polarity
) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
"""

# 6. Preprocess DataFrame
# Replace NaN with None
df = df.replace({pd.NA: None, pd.NaT: None})
df = df.where(pd.notnull(df), None)

# Clean whitespace
df["date"] = df["date"].str.strip()
# Now convert properly
df["date"] = pd.to_datetime(df["date"], format="%B %d, %Y", errors="coerce").dt.date

# 7. Convert DataFrame rows to list of tuples (handling NaN/NaT)
values = [tuple(None if pd.isna(x) else x for x in row)
          for row in df[[
              "title", "text", "subject", "date", "label",
              "clean_text", "label_str", "word_count", "char_count",
              "avg_word_length", "sentence_count", "polarity"
          ]].to_numpy()]

# 7. Insert rows in batches
batch_size = 1000  # due to large dataset we cant insert all at once
for i in range(0, len(values), batch_size):
    batch = values[i:i+batch_size]
    cursor.executemany(sql, batch)
    conn.commit()
    print(f"Inserted batch {i+1} to {i+len(batch)}")

# 8. Close cursor and connection
cursor.close()
conn.close()
print("All batches inserted successfully.")