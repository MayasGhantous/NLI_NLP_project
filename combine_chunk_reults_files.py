import json
files = ['evaluation_chunk_results_asel.json', 'evaluation_chunk_results2.json', 'evaluation_chunk_results_mo.json']
combined = {}
corrent = 0
incorrect = 0
langauges = 0
for file in files:
    with open(file) as f:
        data = json.load(f)
        for key in data:
            if key in combined.keys():
                continue
            if key == 'total':
                continue
            if key == 'overall_accuracy':
                continue
            langauges+=1
            combined[key] = data[key]
            corrent += data[key]['correct']
            incorrect += data[key]['incorrect']
combined['total'] = {'correct': corrent, 'incorrect': incorrect, 'accuracy': corrent/(corrent+incorrect)}
print(langauges)
with open('evaluation_chunk_results_combined.json', 'w') as f:
    json.dump(combined, f)
print(corrent/(corrent+incorrect))