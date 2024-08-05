import json
output_file = 'evaluation_chunk_results.json'

results={}
with open(output_file, 'r') as file:
    results = json.load(file)
count = 0
for key in results.keys():
    if key == "overall_accuracy":
        continue
    print(key)
    print(results[key]["accuracy"])
    count += results[key]["accuracy"]
    print("\n")
print("Average accuracy: ", count/23)
    