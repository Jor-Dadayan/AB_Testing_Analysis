-- Задача: посчитать эффект treatment относительно control одной строкой.
-- Конструкция: CTE group_stats считает метрики по группам, CTE pivoted с помощью
-- CASE WHEN разворачивает две строки в колонки, чтобы вычитать значения друг из друга.
-- Результат: 1 строка. Это только разница в выборке, значимость проверяется в Python.
WITH group_stats AS (
    SELECT group_name,
           AVG(converted::numeric) AS conversion,
           AVG(revenue)            AS revenue_per_user
    FROM user_analytics
    GROUP BY group_name
),
pivoted AS (
    SELECT
        MAX(CASE WHEN group_name = 'control'   THEN conversion END)       AS conv_control,
        MAX(CASE WHEN group_name = 'treatment' THEN conversion END)       AS conv_treatment,
        MAX(CASE WHEN group_name = 'control'   THEN revenue_per_user END) AS rpu_control,
        MAX(CASE WHEN group_name = 'treatment' THEN revenue_per_user END) AS rpu_treatment
    FROM group_stats
)
SELECT
    ROUND(conv_control, 4)                              AS conv_control,
    ROUND(conv_treatment, 4)                            AS conv_treatment,
    ROUND(conv_treatment - conv_control, 4)             AS abs_lift_conversion,
    ROUND((conv_treatment / conv_control - 1) * 100, 2) AS rel_lift_conversion_pct,
    ROUND(rpu_control, 4)                               AS rpu_control,
    ROUND(rpu_treatment, 4)                             AS rpu_treatment,
    ROUND(rpu_treatment - rpu_control, 4)               AS abs_lift_rpu,
    ROUND((rpu_treatment / rpu_control - 1) * 100, 2)   AS rel_lift_rpu_pct
FROM pivoted;
