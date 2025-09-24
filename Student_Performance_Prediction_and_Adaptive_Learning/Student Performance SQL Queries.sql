-- Exam Scores by Gender
SELECT g.text AS Gender, ROUND(AVG(sp.Final_Exam_Score),2) AS Avg_Final_Exam_Score
FROM student_performance sp
JOIN gender g ON sp.Gender = g.id
GROUP BY g.text;

-- Dropout Rate by Education Level
SELECT e.text AS Education_Level,
       ROUND(SUM(CASE WHEN Dropout_Likelihood='Yes' THEN 1 ELSE 0 END)*100.0/COUNT(*),2) AS Dropout_Rate_Percent
FROM student_performance sp
JOIN education_level e ON sp.Education_Level = e.id
GROUP BY e.text;

-- Average Exam Score by Course
SELECT c.text AS Course_Name, ROUND(AVG(Final_Exam_Score),2) AS Avg_Final_Exam_Score
FROM student_performance sp
JOIN course_name c ON sp.Course_Name = c.id
GROUP BY c.text;

-- Engagement vs Feedback
SELECT el.text AS Engagement_Level, ROUND(AVG(Feedback_Score),2) AS Avg_Feedback_Score
FROM student_performance sp
JOIN engagement_level el ON sp.Engagement_Level = el.id
GROUP BY el.text;

-- Exam Score by Learning Style
SELECT l.text AS Learning_Style, ROUND(AVG(Final_Exam_Score),2) AS Avg_Final_Exam_Score
FROM student_performance sp
JOIN learning_style l ON sp.Learning_Style = l.id
GROUP BY l.text;

-- Dropout Distribution
SELECT Dropout_Likelihood, ROUND(COUNT(*)*100.0/(SELECT COUNT(*) FROM student_performance),2) AS Percentage
FROM student_performance
GROUP BY Dropout_Likelihood;

-- Avg Exam Score by Quiz Attempts Group
SELECT 
  CASE Quiz_Scores_Quartile
       WHEN 1 THEN 'Q1-Low'
       WHEN 2 THEN 'Q2'
       WHEN 3 THEN 'Q3'
       WHEN 4 THEN 'Q4-High'
  END AS Quiz_Scores_Quartile,
  ROUND(AVG(Final_Exam_Score),2) AS Avg_Final_Exam_Score
FROM (
  SELECT sp.*, 
         NTILE(4) OVER (ORDER BY Quiz_Scores) AS Quiz_Scores_Quartile
  FROM student_performance sp
) t
GROUP BY Quiz_Scores_Quartile;

-- Avg Exam Score by Video Quartiles 
SELECT 
  CASE Video_Time_Quartile
       WHEN 1 THEN 'Q1-Low'
       WHEN 2 THEN 'Q2'
       WHEN 3 THEN 'Q3'
       WHEN 4 THEN 'Q4-High'
  END AS Video_Time_Quartile,
  ROUND(AVG(Final_Exam_Score),2) AS Avg_Final_Exam_Score
FROM (
  SELECT sp.*,
         NTILE(4) OVER (ORDER BY Time_Spent_on_Videos) AS Video_Time_Quartile
  FROM student_performance sp
) t
GROUP BY Video_Time_Quartile;

-- Pass Rate by Course
SELECT c.text AS Course_Name,
       ROUND(SUM(CASE WHEN Final_Exam_Score>=50 THEN 1 ELSE 0 END)/COUNT(*),2) AS Pass_Rate
FROM student_performance sp
JOIN course_name c ON sp.Course_Name = c.id
GROUP BY c.text;



-- Avg Exam Score by Dropout
SELECT Dropout_Likelihood, ROUND(AVG(Final_Exam_Score),2) AS Avg_Final_Exam_Score
FROM student_performance
GROUP BY Dropout_Likelihood;

-- Crosstab: Learning Style vs Engagement Level
SELECT l.text AS Learning_Style, el.text AS Engagement_Level, 
       ROUND(COUNT(*)*100.0 / SUM(COUNT(*)) OVER (PARTITION BY l.text),2) AS Percentage
FROM student_performance sp
JOIN learning_style l ON sp.Learning_Style = l.id
JOIN engagement_level el ON sp.Engagement_Level = el.id
GROUP BY l.text, el.text;

-- Avg Exam Score by Quartiles 
SELECT Quiz_Scores_Quartile, ROUND(AVG(Final_Exam_Score),2) AS Avg_Final_Exam_Score
FROM (
  SELECT sp.*, NTILE(4) OVER (ORDER BY Quiz_Scores) AS Quiz_Scores_Quartile
  FROM student_performance sp
) t
GROUP BY Quiz_Scores_Quartile;

-- Score per Hour bassed on Course nmae
SELECT c.text AS Course_Name, ROUND(AVG(Final_Exam_Score/NULLIF(Time_Spent_on_Videos,0)),2) AS Avg_Score_per_Hour
FROM student_performance sp
JOIN course_name c ON sp.Course_Name = c.id
GROUP BY c.text;

-- Score Variance by Course
SELECT c.text AS Course_Name, ROUND(VAR_SAMP(Final_Exam_Score),2) AS Score_Variance
FROM student_performance sp
JOIN course_name c ON sp.Course_Name = c.id
GROUP BY c.text;

-- Score Range by Learning Style
SELECT l.text AS Learning_Style, MIN(Final_Exam_Score) AS Min_Score, MAX(Final_Exam_Score) AS Max_Score
FROM student_performance sp
JOIN learning_style l ON sp.Learning_Style = l.id
GROUP BY l.text;

-- Dropout Rate by Engagement Level
SELECT el.text AS Engagement_Level,
       ROUND(SUM(CASE WHEN Dropout_Likelihood='Yes' THEN 1 ELSE 0 END)*100.0/COUNT(*),2) AS Dropout_Rate_Percent
FROM student_performance sp
JOIN engagement_level el ON sp.Engagement_Level = el.id
GROUP BY el.text;


-- Combined Engagement Level
SELECT 
  CASE Engagement_Level_Cat
       WHEN 1 THEN 'Low'
       WHEN 2 THEN 'Medium'
       WHEN 3 THEN 'High'
  END AS Engagement_Level_Cat,
  ROUND(AVG(Final_Exam_Score),2) AS Avg_Final_Exam_Score
FROM (
  SELECT sp.*, 
         NTILE(3) OVER (ORDER BY (Time_Spent_on_Videos+Assignment_Completion_Rate+Forum_Participation)) AS Engagement_Level_Cat
  FROM student_performance sp
) t
GROUP BY Engagement_Level_Cat;

-- Video vs Assignment Quartiles
SELECT 
  CASE Video_Quartile
       WHEN 1 THEN 'Q1'
       WHEN 2 THEN 'Q2'
       WHEN 3 THEN 'Q3'
       WHEN 4 THEN 'Q4'
  END AS Video_Quartile,
  CASE Assignment_Quartile
       WHEN 1 THEN 'Q1'
       WHEN 2 THEN 'Q2'
       WHEN 3 THEN 'Q3'
       WHEN 4 THEN 'Q4'
  END AS Assignment_Quartile,
  ROUND(AVG(Final_Exam_Score),2) AS Avg_Final_Exam_Score
FROM (
  SELECT sp.*,
         NTILE(4) OVER (ORDER BY Time_Spent_on_Videos) AS Video_Quartile,
         NTILE(4) OVER (ORDER BY Assignment_Completion_Rate) AS Assignment_Quartile
  FROM student_performance sp
) t
GROUP BY Video_Quartile, Assignment_Quartile;

-- Students at Risk
SELECT Student_ID, Final_Exam_Score, Dropout_Likelihood
FROM student_performance
WHERE Dropout_Likelihood='Yes' AND Final_Exam_Score < 50
LIMIT 10;

-- Courses by Avg Score
SELECT c.text AS Course_Name, ROUND(AVG(Final_Exam_Score),2) AS Avg_Final_Exam_Score
FROM student_performance sp
JOIN course_name c ON sp.Course_Name = c.id
GROUP BY c.text
ORDER BY Avg_Final_Exam_Score DESC
LIMIT 5;

-- Bottom 5 Courses by Avg Score
SELECT c.text AS Course_Name, ROUND(AVG(Final_Exam_Score),2) AS Avg_Final_Exam_Score
FROM student_performance sp
JOIN course_name c ON sp.Course_Name = c.id
GROUP BY c.text
ORDER BY Avg_Final_Exam_Score ASC
LIMIT 5;

-- Pass/Fail Count
SELECT SUM(CASE WHEN Final_Exam_Score>=50 THEN 1 ELSE 0 END) AS Pass,
       SUM(CASE WHEN Final_Exam_Score<50 THEN 1 ELSE 0 END) AS Fail
FROM student_performance;

-- Top & Bottom 5 Students
SELECT c.text AS Course_Name, sp.Final_Exam_Score, l.text AS Learning_Style
FROM student_performance sp
JOIN course_name c ON sp.Course_Name = c.id
JOIN learning_style l ON sp.Learning_Style = l.id
ORDER BY sp.Final_Exam_Score DESC
LIMIT 5;

-- Bottom 5
SELECT c.text AS Course_Name, sp.Final_Exam_Score, l.text AS Learning_Style
FROM student_performance sp
JOIN course_name c ON sp.Course_Name = c.id
JOIN learning_style l ON sp.Learning_Style = l.id
ORDER BY sp.Final_Exam_Score ASC
LIMIT 5;

-- Dropout vs Performance
SELECT Dropout_Likelihood,
       COUNT(*) AS Students,
       ROUND(AVG(Final_Exam_Score),2) AS Avg_Score,
       MIN(Final_Exam_Score) AS Min_Score,
       MAX(Final_Exam_Score) AS Max_Score
FROM student_performance
GROUP BY Dropout_Likelihood;

-- Learning Style Effect
SELECT l.text AS Learning_Style,
       ROUND(AVG(Final_Exam_Score),2) AS Avg_Score,
       MIN(Final_Exam_Score) AS Min_Score,
       MAX(Final_Exam_Score) AS Max_Score
FROM student_performance sp
JOIN learning_style l ON sp.Learning_Style = l.id
GROUP BY l.text;
