This folder contains basic data analysis done on the regional yearly rainfall data from 1901 to 2017 taken from "https://data.gov.in". 

Observations from region-wise scatter plot of annual rainfall VS JJAS(monsoon) rainfall:
1. Strong positive linear relationship indicates that JJAS (monsoon months) contribute the major portion of the annual rainfall across all regions.
2. In North-eastern regions, more spread/scatter is observed which indicates that rainfall outside JJAS months also plays a vital role.
3. Some South-East and South regions (like Andaman & Nicobar, Tamil Nadu, Rayalaseema) show deviations which is explained by their dependency on Northeast Monsoon (Oct–Dec).

Observations from box plot showing seasonal rainfall (Considering 1955 as the beginning of computer-era in India, we compare seasonal wise rainfall for pre- VS post- computer era using box plot):
1. Most regions show considerable variability in rainfall across seasons with JJAS(monsoon months) taking up maximum rainfall value.
2. Post-Computer era shows slightly lower median rainfall in many regions but has more extreme high rainfall outliers in some regions (like NE, SW, S).
3. Variability is observed to change regionally. Some regions have narrower post-computer rainfall distribution (such as SE, N), while others (such as NE, SW, S) see more spread and extreme events.

Observations from bar plot comparing the rainfall pattern of decade 1901-1910 and 2008-2017 for the unique regional entries of 'SUBDIVISION' column:
1. Coastal Karnataka, Andaman & Nicobar Islands, Kerala, and Konkan & Goa consistently receive highest rainfall in both periods. Similarly, West Rajasthan and Saurashtra & Kutch remain the driest regions.
2. Some regions (e.g., Gangetic West Bengal, Jharkhand, Bihar) appear to have decreased in ranking in terms of rainfall. Chhattisgarh and Orissa also appear to have slightly lower values or rankings in the second period. This suggests possible drying trends in eastern-central India.
3. Western Ghats regions (like Konkan) have slightly increased rainfall in 2008-2017 decade.
4. The middle tier (e.g., Telangana, East Madhya Pradesh, Madhya Maharashtra) shows shuffling in ranks, indicating local changes in rainfall patterns

Observations from heatmap showing regional mean seasonal rainfall correlation:
1. SW-W, C-N and SW-S pairs show strong positive correlation, illustrating similar rainfall patterns.
2. In annual rainfall, NE-W pair has the most negative correlation portraying vastly different rainfall behavior.
3. NE is an outlier with minimal similarity to others.
