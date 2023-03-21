DROP SCHEMA IF EXISTS upgrad;

CREATE SCHEMA upgrad;

USE upgrad;

CREATE TABLE salaries(name VARCHAR(30),salary INT);

INSERT INTO salaries VALUES
('Declan Johnson', 95000),
('Nitya Chandra', 50000),
('Lucas Bell', 107000),
('Ellis Shaw', 24000),
('Garrett Atkins', 92000),
('Phoenix Richards', 33000),
('Alexander Wang', 96000),
('Jamie Wilson', 62000),
('Kiran Delaney', 20000),
('Marita Estrada', 26000);

CREATE TABLE messi_goals(Year INT,Goals INT);

INSERT INTO messi_goals VALUES
(2010, 60),
(2011, 59),
(2012, 91),
(2013, 45),
(2014, 58),
(2015, 52),
(2016, 59),
(2017, 54),
(2018, 51),
(2019, 50),
(2020, 27),
(2021, 43);

CREATE TABLE student(stu_id INT,first_name VARCHAR(30), middle_name VARCHAR(30), last_name VARCHAR(30), hobby VARCHAR(30));

INSERT INTO student VALUES
(13, 'William', 'Mae', 'Stewart', 'Reading'),
(17, 'Emma',     NULL, 'Jenkins', 'Swimming'),
(18, 'Claire',  'Mary', 'Martinez', NULL),
(29, 'Alice',   NULL,	NULL, 		'Knitting');