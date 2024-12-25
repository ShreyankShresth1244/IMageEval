import requests

url = "https://img.pixelz.com/blog/white-background-photography/product-photo-liqueur-bottle-1000.jpg"  # Replace with the actual URL
response = requests.get(url)

if response.status_code == 404:
    print("Image not found (HTTP 404).")
else:
    print(f"Image fetched successfully with status code: {response.status_code}")
