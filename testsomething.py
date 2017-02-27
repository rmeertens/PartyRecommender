import requests

hostname = "http://inforbuntu:5009/predict"


payload = "{\"text\":\"groen dieren geen kolen \"}"
headers = {
    'origin': "http://verkiezingsradar.bigdatarepublic.nl",
    'x-devtools-emulate-network-conditions-client-id': "b55ebfed-6d07-4e1c-a42e-be325f4cf5ae",
    'user-agent': "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/57.0.2987.74 Safari/537.36",
    'content-type': "application/json",
    'accept': "*/*",
    'referer': "http://verkiezingsradar.bigdatarepublic.nl/",
    'accept-encoding': "gzip, deflate",
    'accept-language': "en-US,en;q=0.8",
    'cache-control': "no-cache",
    'postman-token': "30436b82-4674-804e-f23d-449f3c084097",
    'predicttext':'groen gras'
    }

response = requests.request("GET", hostname, data=payload, headers=headers)

print(response.text)