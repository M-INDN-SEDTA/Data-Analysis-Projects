-- 1. Dataset Overview
SELECT COUNT(*) AS total_records FROM news_articles;
SELECT COUNT(DISTINCT subject) AS num_subjects FROM news_articles;
SELECT DISTINCT subject FROM news_articles;

-- 2. Class Distribution
SELECT label_str, COUNT(*) AS count,
       ROUND(100 * COUNT(*) / (SELECT COUNT(*) FROM news_articles), 2) AS percentage
FROM news_articles
GROUP BY label_str;

-- 3. Subject Distribution (Top 10)
SELECT subject, COUNT(*) AS count
FROM news_articles
GROUP BY subject
ORDER BY count DESC
LIMIT 10;

-- 4. Word & Character Statistics
SELECT 
    ROUND(AVG(word_count), 2) AS avg_word_count,
    MAX(word_count) AS max_word_count,
    MIN(word_count) AS min_word_count,
    ROUND(AVG(char_count), 2) AS avg_char_count,
    ROUND(AVG(sentence_count), 2) AS avg_sentence_count,
    ROUND(AVG(avg_word_length), 2) AS avg_word_length
FROM news_articles;

-- 5. Word & Character Stats by Class
SELECT label_str,
    ROUND(AVG(word_count), 2) AS avg_word_count,
    ROUND(AVG(char_count), 2) AS avg_char_count,
    ROUND(AVG(sentence_count), 2) AS avg_sentence_count
FROM news_articles
GROUP BY label_str;

-- 6. Sentiment (Polarity)
SELECT 
    ROUND(AVG(polarity), 3) AS overall_avg_polarity
FROM news_articles;

SELECT 
    label_str,
    ROUND(AVG(polarity), 3) AS avg_polarity
FROM news_articles
GROUP BY label_str;

-- 7. Articles by Year
SELECT YEAR(date) AS year, COUNT(*) AS count
FROM news_articles
GROUP BY YEAR(date)
ORDER BY year;

-- 8. Articles by Month-Year
SELECT DATE_FORMAT(date, '%Y-%m') AS month, COUNT(*) AS count
FROM news_articles
GROUP BY month
ORDER BY month;

-- 9. Highest Polarity Article
SELECT id, title, polarity
FROM news_articles
ORDER BY polarity DESC
LIMIT 1;

-- 10. Lowest Polarity Article
SELECT id, title, polarity
FROM news_articles
ORDER BY polarity ASC
LIMIT 1;

-- 11. Top 5 Longest Articles (Word Count)
SELECT id, title, word_count
FROM news_articles
ORDER BY word_count DESC
LIMIT 5;

-- 12. Top 5 Shortest Articles (Word Count)
SELECT id, title, word_count
FROM news_articles
ORDER BY word_count ASC
LIMIT 5;

-- 13. Average Polarity by Subject
SELECT subject, ROUND(AVG(polarity), 3) AS avg_polarity
FROM news_articles
GROUP BY subject
ORDER BY avg_polarity DESC;

-- 14. Articles per Subject & Label
SELECT subject, label_str, COUNT(*) AS count
FROM news_articles
GROUP BY subject, label_str
ORDER BY subject, count DESC;
