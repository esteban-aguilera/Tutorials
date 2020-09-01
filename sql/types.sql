/* fill in the blanks */
CREATE TABLE inventory (
    item_name VARCHAR(100),
    price DECIMAL(5,2),
    quantity INT
);


/* format curent date */
SELECT NOW();
SELECT CURDATE();
SELECT DAYOFWEEK(CURDATE());
SELECT DAYNAME(CURDATE());
SELECT DATE_FORMAT(NOW(), '%d/%m/%Y');
SELECT DATE_FORMAT(NOW(), '%M %D at %h:%i');

/* create tweets table */
CREATE TABLE tweets (
    content VARCHAR(140),
    username VARCHAR(20),
    created_at TIMESTAMP DEFAULT NOW()
);
