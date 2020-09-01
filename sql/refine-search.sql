/* find the books that have the work stories in the title */
SELECT title FROM books WHERE title LIKE '%stories%';

/* find the longest book */
SELECT title, pages FROM books
ORDER BY pages DESC LIMIT 1;

/* summary of the three most recent books */
SELECT CONCAT
    (
        title,
        ' - ',
        released_year
    ) AS 'summary'
FROM books
ORDER BY released_year DESC LIMIT 3;

/* find all the books whose author's last name contains a space */
SELECT title, author_lname
FROM books
WHERE author_lname LIKE '% %';

/* find the three books with the lowest number in stock quantity */
SELECT title, released_year, stock_quantity
FROM books
ORDER BY stock_quantity
LIMIT 3;

/* print title and author's last name.  Sort them by author's last name and then by title */
SELECT title, author_lname
FROM books
ORDER BY author_lname, title;

/* Make it happen and sort it by the author's last name */
SELECT
    CONCAT(
        'MY FAVORITE AUTHOR IS ',
        UPPER(
            CONCAT(
                author_fname,
                ' ',
                author_lname
            )
        ),
        '!'
    ) AS yell
FROM books
ORDER BY author_lname;
