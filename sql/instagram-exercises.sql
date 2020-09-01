-- 1. Finding 5 oldest users
SELECT
    username, created_at
FROM users
ORDER BY created_at
LIMIT 5;


-- 2. Most popular registration date
SELECT
    DAYNAME(created_at) AS 'day',
    COUNT(*) AS 'total'
FROM users
GROUP BY DAYNAME(created_at)
ORDER BY 2 DESC;


-- 3. Find users that have never posted a photo
SELECT
    users.username
FROM users
LEFT JOIN photos
    ON users.id = photos.user_id
WHERE photos.image_url IS NULL;


-- 4. User with most likes on a single photo
SELECT
    photos.id,
    users.username,
    photos.image_url,
    COUNT(*) AS total
FROM photos
INNER JOIN likes
    ON likes.photo_id = photos.id
INNER JOIN users
    ON photos.user_id = users.id
GROUP BY photos.id
ORDER BY total DESC
LIMIT 10;


-- 5. How many times does the average user post
SELECT
    AVG(tabl.total) AS 'avg'
FROM (
    SELECT
        users.username AS username,
        CASE
            WHEN AVG(photos.image_url) IS NULL THEN 0
            ELSE COUNT(*)
        END AS total
    FROM users
    LEFT JOIN photos
        ON users.id = photos.user_id
    GROUP BY users.id
    ORDER BY total
) AS tabl;


-- 6. Most used hashtags
SELECT
    tags.tag_name,
    COUNT(*) AS tag_count
FROM tags
INNER JOIN photo_tags
    ON tags.id = photo_tags.tag_id
GROUP BY tags.id
ORDER BY tag_count DESC
LIMIT 5;


-- 7. Find users who have liked every single photo on the site
SELECT
    tabl.username,
    tabl.total
FROM (
    SELECT
        users.username,
        COUNT(*) AS total
    FROM users
    INNER JOIN likes
        ON users.id = likes.user_id
    GROUP BY users.id
    ORDER BY total DESC
    ) AS tabl
WHERE tabl.total = (SELECT COUNT(*) FROM photos);
