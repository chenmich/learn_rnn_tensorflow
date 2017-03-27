import json
json_content ={}
with open('data/result_data/tmp.json', mode='w') as jsonfile:
    json_content["a"] = [[1, 2, 3], [2, 3, 4]]
    json.dump(json_content, jsonfile)
    json_content = {}
    json_content["b"] = [[5, 6, 7], [8,9,10]]
    json.dump(json_content, jsonfile)

