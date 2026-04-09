"""Analyze all source datasets and their overlap."""
import pandas as pd

d1 = pd.read_csv('d:/Navya Singhal/dattaset/symbipredict_2022.csv', nrows=0)
d2 = pd.read_csv('d:/Navya Singhal/dattaset/Disease and symptoms dataset.csv', nrows=0)
d3 = pd.read_csv('d:/Navya Singhal/dattaset/Indian-Healthcare-Symptom-Disease-Dataset - Sheet1 (2).csv')

s1 = set(c.lower().strip().replace(' ','_') for c in d1.columns if c != 'prognosis')
s2 = set(c.lower().strip().replace(' ','_') for c in d2.columns if c != 'diseases')

print(f'Dataset 1 (symbipredict) symptoms: {len(s1)}')
print(f'Dataset 2 (Disease&symptoms) symptoms: {len(s2)}')
overlap = s1 & s2
print(f'Overlap: {len(overlap)} symptoms in common')
print(f'Combined unique: {len(s1 | s2)}')
if overlap:
    print(f'Overlapping names (first 20): {sorted(overlap)[:20]}')

# Indian Healthcare metadata
print(f'\nIndian Healthcare metadata: {len(d3)} symptom entries')
print(f'Severity values: {d3["Severity"].unique().tolist()}')
print(f'Sample:')
print(d3[['Symptom','Severity','Common in Region']].head(8).to_string())

# Check disease overlap
d1f = pd.read_csv('d:/Navya Singhal/dattaset/symbipredict_2022.csv')
d2f = pd.read_csv('d:/Navya Singhal/dattaset/Disease and symptoms dataset.csv', usecols=['diseases'], low_memory=False)

dis1 = set(d1f['prognosis'].str.lower().str.strip().unique())
dis2 = set(d2f['diseases'].str.lower().str.strip().unique())
print(f'\nDataset 1 diseases: {len(dis1)}')
print(f'Dataset 2 diseases: {len(dis2)}')
print(f'Disease overlap: {len(dis1 & dis2)}')
print(f'Combined unique diseases: {len(dis1 | dis2)}')
