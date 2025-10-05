This folder contains an exploratory data analysis done on the regional yearly rainfall data from 1901 to 2017 taken from "https://data.gov.in".

**Dataset**
Source: data.gov.in
Time Period: 1901–2017
Coverage: Regional (based on “SUBDIVISION” classification)
Key Variables: Annual rainfall (mm), Seasonal rainfall (e.g., JJAS – monsoon months) and Region/Subdivision identifiers

**Objective**
The main objectives of this analysis are to:
1. Explore the contribution of monsoon (JJAS) rainfall to annual rainfall.
2. Compare seasonal rainfall characteristics before and after the “computer era” (pre-1955 vs post-1955).
3. Examine decadal changes in rainfall distribution across regions.
4. Identify correlations and similarities among regional rainfall patterns.

**Observations**

**_Annual vs. JJAS Rainfall (Scatter Plot)_**
1. Strong positive linear relationship indicates that JJAS (monsoon months) contribute the major portion of the annual rainfall across all regions.
2. In North-eastern regions, more spread/scatter is observed which indicates that rainfall outside JJAS months also plays a vital role.
3. Some South-East and South regions (like Andaman & Nicobar, Tamil Nadu, Rayalaseema) show deviations which is explained by their dependency on Northeast Monsoon (Oct–Dec).

**_Seasonal Rainfall Comparison for Pre- vs Post-Computer Era (Box Plot)_** 
(Considering 1955 as the beginning of computer-era in India, we compare seasonal wise rainfall for pre- VS post- computer era using box plot):
1. Most regions show considerable variability in rainfall across seasons with JJAS(monsoon months) taking up maximum rainfall value.
2. Post-Computer era shows slightly lower median rainfall in many regions but has more extreme high rainfall outliers in some regions (like NE, SW, S).
3. Variability is observed to change regionally. Some regions have narrower post-computer rainfall distribution (such as SE, N), while others (such as NE, SW, S) see more spread and extreme events.

**_Decadal Comparison: 1901–1910 vs 2008–2017 (Bar Plot)_**
1. Coastal Karnataka, Andaman & Nicobar Islands, Kerala, and Konkan & Goa consistently receive highest rainfall in both periods. Similarly, West Rajasthan and Saurashtra & Kutch remain the driest regions.
2. Some regions (e.g., Gangetic West Bengal, Jharkhand, Bihar) appear to have decreased in ranking in terms of rainfall. Chhattisgarh and Orissa also appear to have slightly lower values or rankings in the second period. This suggests possible drying trends in eastern-central India.
3. Western Ghats regions (like Konkan) have slightly increased rainfall in 2008-2017 decade.
4. The middle tier (e.g., Telangana, East Madhya Pradesh, Madhya Maharashtra) shows shuffling in ranks, indicating local changes in rainfall patterns

**_Regional Correlation Analysis (Heatmap)_**
1. SW-W, C-N and SW-S pairs show strong positive correlation, illustrating similar rainfall patterns.
2. In annual rainfall, NE-W pair has the most negative correlation portraying vastly different rainfall behavior.
3. NE is an outlier with minimal similarity to others.

**Conclusion**
1. Monsoon months (JJAS) dominate Indian annual rainfall across all regions.
2. Post-1955, rainfall variability has increased in some regions where extreme rainfall events are exhibited.
3. Some eastern and central regions show signs of drying, while western coastal areas exhibit stable or slightly increasing rainfall.
4. Regional correlations highlight distinct climatic zones, with the North-East remaining unique in its rainfall behavior.
