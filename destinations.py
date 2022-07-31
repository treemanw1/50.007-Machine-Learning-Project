import json

# Opening JSON file
with open('destinations.json', encoding="utf8") as json_file:
    data = json.load(json_file)

    # Print the type of data variable
    print("Type:", type(data))

# print(len(data))
# data = list(filter(None, data))  # remove empty dicts (non apparently)
print(len(data))
# remove entries without "term" key
data_terms = []
for place in data:
    if "term" in place.keys():
        data_terms.append(place)
print(len(data_terms))
# remove entries with term != city
data_terms = [x for x in data_terms if x["type"]=="city"]
print(len(data_terms))

# convert back to json
output = json.dumps(data_terms, indent=2)
f = open('destinations_processed.json', 'w')
f.write(output)
