/* CASE examples */
SELECT title, released_year,
    CASE
        WHEN released_year >= 2000 THEN 'Modern Lit'
        ELSE '20th Century Lit'
    END AS 'genre'
FROM books;


SELECT
    title,
    released_year AS 'year',
    CASE
        WHEN stock_quantity <= 50 THEN '*'
        WHEN stock_quantity <= 100 THEN '**'
        ELSE '***'
    END AS 'stock'
FROM books;


-- evaluate
SELECT 10 != 10;
SELECT 15 > 14 AND 99-5 <= 94;
SELECT 1 IN (5,3) OR 9 BETWEEN 8 AND 10;

-- select all books before 1980
SELECT title FROM books
WHERE released_year < 1980;

-- 
SELECT title, author_fname, author_lname FROM books
WHERE author_lname IN ('Eggers', 'Chabon');

-- SELECT books written by lahiri and after 2000
SELECT title,
    CONCAT(author_fname, ' ', author_lname) AS 'author'
FROM books
WHERE
    author_lname = 'Lahiri' AND
    released_year > 2000;


-- 
SELECT title, pages FROM books
WHERE pages BETWEEN 100 AND 200;


-- get every author whose lastname starts with c or s
SELECT title, 
    CONCAT(author_fname, ' ', author_lname) AS 'author'
FROM books
WHERE
    author_lname LIKE 'C%' OR
    author_lname LIKE 'S%';


-- Case statements example
SELECT
    title,
    author_lname,
    CASE
        WHEN title IN ('Just Kids', 'A Heartbreaking Work of Staggering Genius') THEN 'Memoir'
        WHEN title LIKE '%stories%' THEN 'Short Stories'
        ELSE 'Novel' 
    END AS 'type'
FROM books;


-- Make this happen...
SELECT
    author_fname,
    author_lname,
    CASE
        WHEN COUNT(*) = 1 THEN '1 book'
        ELSE CONCAT(COUNT(*), ' books')
    END AS 'count'
FROM books
GROUP BY author_fname, author_lname
ORDER BY author_lname;
