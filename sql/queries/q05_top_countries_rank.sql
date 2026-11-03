-- Задача: топ-3 страны по выручке в каждой группе и их доля в выручке группы.
-- Конструкция: RANK() OVER (PARTITION BY group ...) нумерует страны внутри группы.
-- Фильтровать по результату оконной функции в том же SELECT нельзя, поэтому
-- ранжирование вынесено в CTE. SUM() OVER даёт знаменатель для доли.
-- Результат: 6 строк (3 страны x 2 группы).
WITH country_revenue AS (
    SELECT group_name, country,
           COUNT(*)     AS users,
           SUM(revenue) AS revenue
    FROM user_analytics
    GROUP BY group_name, country
),
ranked AS (
    SELECT *,
           RANK() OVER (PARTITION BY group_name ORDER BY revenue DESC) AS revenue_rank,
           ROUND(100.0 * revenue / SUM(revenue) OVER (PARTITION BY group_name), 1)
               AS share_of_group_pct
    FROM country_revenue
)
SELECT group_name, revenue_rank, country, users,
       ROUND(revenue, 2) AS revenue, share_of_group_pct
FROM ranked
WHERE revenue_rank <= 3
ORDER BY group_name, revenue_rank;
