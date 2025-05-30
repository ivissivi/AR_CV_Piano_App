import requests
from bs4 import BeautifulSoup
import json
from datetime import datetime
import time
import random

class AlfaRomeoScraper:
    def __init__(self):
        self.base_url = "https://www.autotrader.co.uk/car-search?make=ALFA%20ROMEO"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        self.cars = []

    def scrape_listings(self, max_pages=5):
        print(f"\n🚗 Starting Alfa Romeo scraper...")
        
        for page in range(1, max_pages + 1):
            try:
                url = f"{self.base_url}&page={page}"
                print(f"\n📄 Scraping page {page}...")
                
                response = requests.get(url, headers=self.headers)
                if response.status_code != 200:
                    print(f"❌ Error accessing page {page}: {response.status_code}")
                    continue
                
                soup = BeautifulSoup(response.text, 'html.parser')
                listings = soup.find_all('div', class_='listing')
                
                if not listings:
                    print(f"ℹ️ No more listings found on page {page}")
                    break
                
                for listing in listings:
                    try:
                        car_data = self._extract_car_data(listing)
                        if car_data:
                            self.cars.append(car_data)
                            print(f"✅ Scraped: {car_data['title']}")
                    except Exception as e:
                        print(f"❌ Error processing listing: {str(e)}")
                
                # Random delay between requests to be respectful
                time.sleep(random.uniform(2, 4))
                
            except Exception as e:
                print(f"❌ Error scraping page {page}: {str(e)}")
                continue
        
        print(f"\n✨ Scraping completed! Found {len(self.cars)} Alfa Romeos")
        return self.cars

    def _extract_car_data(self, listing):
        try:
            title = listing.find('h2', class_='listing-title').text.strip()
            price = listing.find('div', class_='price').text.strip()
            year = listing.find('div', class_='year').text.strip()
            mileage = listing.find('div', class_='mileage').text.strip()
            location = listing.find('div', class_='location').text.strip()
            
            # Get the image URL if available
            img_tag = listing.find('img')
            image_url = img_tag['src'] if img_tag and 'src' in img_tag.attrs else None
            
            # Get the listing URL
            link_tag = listing.find('a', class_='listing-link')
            listing_url = link_tag['href'] if link_tag and 'href' in link_tag.attrs else None
            
            return {
                'title': title,
                'price': price,
                'year': year,
                'mileage': mileage,
                'location': location,
                'image_url': image_url,
                'listing_url': listing_url,
                'scraped_at': datetime.now().isoformat()
            }
        except Exception as e:
            print(f"❌ Error extracting car data: {str(e)}")
            return None

    def save_to_json(self, filename='alfa_romeos.json'):
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self.cars, f, indent=4, ensure_ascii=False)
            print(f"\n💾 Saved {len(self.cars)} cars to {filename}")
        except Exception as e:
            print(f"❌ Error saving to JSON: {str(e)}")

def main():
    scraper = AlfaRomeoScraper()
    cars = scraper.scrape_listings()
    scraper.save_to_json()

if __name__ == "__main__":
    main() 