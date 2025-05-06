import os
import requests
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor

# Load API key from .env
load_dotenv()
API_KEY = os.getenv("SERPAPI_KEY")

def google_search(query, site):
    """Search Google for products on a specific e-commerce site."""
    params = {
        "q": f"{query} site:{site}",
        "api_key": API_KEY,
        "engine": "google",
        "num": 5  # Limit to top 5 results
    }
    try:
        response = requests.get("https://serpapi.com/search", params=params)
        response.raise_for_status()  # Raise HTTP errors
        return {
            "site": site,
            "results": response.json().get("organic_results", [])
        }
    except requests.exceptions.RequestException as e:
        print(f"Error searching {site}: {e}")
        return {
            "site": site,
            "results": []
        }

def search_all_platforms(query):
    """Search across multiple e-commerce platforms simultaneously."""
    sites = ["amazon.in", "myntra.com", "flipkart.com"]
    
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(lambda site: google_search(query, site), sites))
    
    return results

# Example usage
if __name__ == "__main__":
    query = "pink shirt with white pocket"  # Replace with your image-to-text model's output
    all_results = search_all_platforms(query)
    
    for platform in all_results:
        print(f"\n=== Results from {platform['site'].upper()} ===")
        if not platform['results']:
            print("No results found")
            continue
            
        for idx, product in enumerate(platform['results'], 1):
            print(f"{idx}. {product.get('title')}")
            if product.get('price'):
                print(f"   Price: {product.get('price')}")
            print(f"   Link: {product.get('link')}\n")