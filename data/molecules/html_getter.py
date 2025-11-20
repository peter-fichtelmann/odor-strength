import requests

def get_static_html(url):
    resp = requests.get(url)
    html = resp.content
    return html