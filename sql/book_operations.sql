/* concatenate */
SELECT
    CONCAT(author_fname, ' ', author_lname)
FROM books;


/* show some columns and concatenate*/
SELECT
    author_fname AS first,
    author_fname AS last,
    CONCAT(author_fname, ' ', author_lname) AS full
FROM books;


/* concatenate with separators */
SELECT
    author_fname AS first,
    author_fname AS last,
    CONCAT_WS(', ', author_fname, author_lname) AS full
FROM books;


/* take a substring and concatenate */
SELECT
    CONCAT
    (
        SUBSTRING(title, 1, 10),
        '...'
    )
    AS 'short title'
FROM books;


/* replace every string 'e' with a '3' */
SELECT
    SUBSTRING(REPLACE(title, 'e', '3'),1,10)
    AS 'weird short title'
FROM books;
