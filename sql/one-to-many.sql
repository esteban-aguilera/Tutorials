-- create tables
CREATE TABLE customers(
    id INT AUTO_INCREMENT PRIMARY KEY,
    first_name VARCHAR(100),
    last_name VARCHAR(100),
    email VARCHAR(255)
);

CREATE TABLE orders(
    id INT AUTO_INCREMENT PRIMARY KEY,
    order_date DATETIME,
    amount DECIMAL(8,2),
    customer_id INT,
    FOREIGN KEY(customer_id)
        REFERENCES customers(id)
        ON DELETE SET NULL
);


-- insert values
INSERT INTO customers (first_name, last_name, email)  VALUES
    ('Boy', 'George', 'george@gmail.com'),
    ('George', 'Michael', 'gm@gmail.com'),
    ('David', 'Bowie', 'david@gmail.com'),
    ('Blue', 'Steele', 'blue@gmail.com'),
    ('Bette', 'Davis', 'bette@aol.com');
        
INSERT INTO orders (order_date, amount, customer_id) VALUES
    ('2016/02/10', 99.99, 1),
    ('2017/11/11', 35.50, 1),
    ('2014/12/12', 800.67, 2),
    ('2015/01/03', 12.50, 2),
    ('1999/04/11', 450.25, 5);


-- find orders from a particular customer
SELECT * FROM orders WHERE 
    customer_id = (
        SELECT id FROM customers
        WHERE last_name = 'George'
    );


-- implicit inner join
SELECT customers.first_name, customers.last_name, orders.order_date, orders.amount
FROM customers, orders
WHERE customers.id = orders.customer_id;


-- explicit inner join
SELECT customers.first_name, customers.last_name, orders.order_date, orders.amount FROM customers
JOIN orders
    ON customers.id = orders.customer_id;


-- left join
SELECT customers.first_name,
    customers.last_name,
    IFNULL(SUM(orders.amount), 0)
FROM customers
LEFT JOIN orders
    ON customers.id = orders.customer_id
GROUP BY customers.id
ORDER BY SUM(orders.amount) DESC;


















-- EXERCISES
CREATE TABLE students(
    id INT AUTO_INCREMENT PRIMARY KEY,
    first_name VARCHAR(100)
);

CREATE TABLE papers(
    id INT AUTO_INCREMENT PRIMARY KEY,
    title VARCHAR(255),
    grade INT,
    student_id INT,
    FOREIGN KEY(student_id)
        REFERENCES students(id)
        ON DELETE SET NULL
);

INSERT INTO students (first_name) VALUES 
('Caleb'), ('Samantha'), ('Raj'), ('Carlos'), ('Lisa');

INSERT INTO papers (student_id, title, grade ) VALUES
(1, 'My First Book Report', 60),
(1, 'My Second Book Report', 75),
(2, 'Russian Lit Through The Ages', 94),
(2, 'De Montaigne and The Art of The Essay', 98),
(4, 'Borges and Magical Realism', 89);


-- print (first_name, paper title, grade)
SELECT
    students.first_name,
    papers.title,
    papers.grade
FROM students
INNER JOIN papers
    ON students.id = papers.student_id
ORDER BY papers.grade DESC;

-- print (first_name, paper title, grade) and include stundents with no papers
SELECT
    students.first_name,
    papers.title,
    papers.grade
FROM students
LEFT JOIN papers
    ON students.id = papers.student_id
ORDER BY papers.grade DESC;

-- print (first_name, paper title, grade) and include stundents with no papers and replace NULL with 0
SELECT
    students.first_name,
    IFNULL(papers.title, 'MISSING') AS 'title',
    IFNULL(papers.grade, 0) AS 'grade'
FROM students
LEFT JOIN papers
    ON students.id = papers.student_id
ORDER BY papers.grade DESC;


-- take the average of papers
SELECT
    students.first_name,
    IFNULL(AVG(papers.grade), 0) AS 'average',
FROM students
LEFT JOIN papers
    ON students.id = papers.student_id
GROUP BY students.id
ORDER BY IFNULL(AVG(papers.grade), 0) DESC;


-- add passing status
SELECT
    students.first_name,
    IFNULL(AVG(papers.grade), 0) AS 'average',
    CASE
        WHEN IFNULL(AVG(papers.grade), 0) >= 75 THEN 'PASSING'
        ELSE 'FAILING'
    END AS 'passing_status'
FROM students
LEFT JOIN papers
    ON students.id = papers.student_id
GROUP BY students.id
ORDER BY IFNULL(AVG(papers.grade), 0) DESC;
