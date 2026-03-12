import json

with open("/raid/home/dgx959/nayan/origin/Drywall-Join-Detect-2/train/_annotations.coco.json", "r") as f:
    data = json.load(f)


print(data.keys())
print(type(data['categories']))
print((data['categories'][0]))