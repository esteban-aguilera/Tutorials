/* number of books in the database */
SELECT COUNT(*) FROM books;


/* number of books by year */
SELECT
    released_year AS year,
    COUNT(*) AS 'number of books'
FROM books
GROUP BY released_year;


/* Total number of books in stock */
SELECT SUM(stock_quantity) FROM books;


/* Average released year of each author */
SELECT
    author_fname
    author_lname,
    AVG(released_year) AS 'number of books'
FROM books
GROUP BY author_fname, author_lname;


/* full name of the author who wrote the longest book */
SELECT CONCAT(author_fname, ' ', author_lname) AS 'full name'
FROM books
ORDER BY pages DESC
LIMIT 1;

SELECT CONCAT(author_fname, ' ', author_lname) AS 'full name'
FROM books
WHERE pages = (
SELECT MAX(pages) FROM books
);


/* Obtain table with [year, number of books released in that year, avg pages of the books that year] */
SELECT
    released_year AS 'year',
    COUNT(*) AS '# books',
    AVG(pages) AS 'avg pages'
FROM books
GROUP BY released_year
ORDER BY released_year;
