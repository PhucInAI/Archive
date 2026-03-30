-- DS QUESTION:
-- What is the relationship between day of week and party drivers being under the influence?

SELECT * FROM parties LIMIT 10;
SELECT DISTINCT party_sobriety FROM parties;

SELECT * FROM collisions LIMIT 10;

-- SQL QUESTION:
-- For the different days of the week, what is the total number of cases
-- where party_sobriety is 'had been drinking, under influence' and party_type is 'driver'?

-- Solution Approach:
-- Join parties and collisions, add column on collision day of week to parties table
-- Filter by party_sobriety and party_type
-- Group joined table by day of week of collision to find the number of parties

-- Q1
-- Join the parties and collisions table, and output the parties table with all its original columns, plus
-- columns on day_of_week (1,2,3 ...) and day_of_week_name (Sunday, Monday, Tuesday ...)
-- HINT (Look up usage for functions DAYOFWEEK() and DAYNAME() online)



-- Q2
-- For each day of the week,
-- find out the number of parties that were under driving the influence at collision time
