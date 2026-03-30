-- DS QUESTION:
-- What is the relationship between cellphone usage and age for parties at fault for collisions in Los Angeles?

SELECT * FROM parties LIMIT 10;

SELECT * FROM collisions LIMIT 10;

-- SQL QUESTION:
-- For different categories of cell_phone_use_type, what is the frequency and average age of at fault parties?
-- county_location should be "los angeles"

-- Solution Approach:
-- GROUP BY cellphone_use_type, find average age and freq of each group
-- Filter above result to only include collisions with case_id corresponding to collision in Los Angeles


-- Q1
-- Find the average age of at fault parties
SELECT AVG(party_age) AS avg_party_age
FROM   parties
WHERE  at_fault = 1
;


-- Q2
-- For different categories of cell_phone_use_type,
-- find the number of at fault parties and the average age of at fault parties
SELECT 		COUNT(*) as count_parties,
			AVG(party_age) AS avg_party_age
FROM   		parties
WHERE  		at_fault = 1
GROUP BY 	cellphone_use_type
ORDER BY	cellphone_use_type
;

-- Q3
-- Do the same analysis as Q2, but only for collisions for which county_location is "los angeles".
-- For different categories of cell_phone_use_type,
-- find the number of at fault parties and the average age of at fault parties
SELECT 		cellphone_use_type,
			COUNT(*) AS count_parties,
			AVG(party_age) AS avg_party_age
FROM   		parties
WHERE  		at_fault = 1 AND case_id IN	(
											SELECT case_id
                                            FROM	collisions
                                            WHERE	county_location = 'loss angles'
										)
GROUP BY 	cellphone_use_type
ORDER BY	cellphone_use_type
;

-- Q3 (With CTE)

WITH los_angeles_conllision AS (
	select *
    FROM collisions
    WHERE county_location = 'los angeles'
    )
SELECT 		cellphone_use_type,
			COUNT(*) AS count_parties,
			AVG(party_age) AS avg_party_age
FROM   		parties
WHERE  		at_fault = 1 AND case_id IN (SELECT case_id FROM los_angeles_conllision)
GROUP BY 	cellphone_use_type
ORDER BY	cellphone_use_type
;
