import urllib.request, json, base64
req = urllib.request.Request('https://api.github.com/repos/yzhao062/anomaly-detection-resources/readme')
res = urllib.request.urlopen(req)
data = json.loads(res.read())
with open("resources_readme.md", "w", encoding="utf-8") as f:
    f.write(base64.b64decode(data['content']).decode('utf-8'))
