import requests
import time
import json

# Official "Gamma" API endpoint for markets
BASE_URL = "https://gamma-api.polymarket.com/markets"

# Our keywords for "Middle East" filtering
MIDDLE_EAST_KEYWORDS = [
    'middle east', 'israel', 'gaza', 'hamas', 'hezbollah', 'idf',
    'iran', 'lebanon', 'syria', 'yemen', 'houthi', 'palestin', # Catches 'palestine'/'palestinian'
    'saudi', 'uae', 'qatar', 'red sea', 'netanyahu', 'sinwar'
]

def fetch_all_markets():
    """
    This function collects *all* relevant markets, filters them
    by keyword, sorts them by 24h volume, and returns
    a sorted list.
    """
    
    all_matching_markets = []
    limit = 100  # 100 markets per request
    offset = 0
    
    print("Starting collection and filtering of all open markets...")

    while True:
        try:
            # We ask the server to pre-filter for us
            params = {
                'limit': limit,
                'offset': offset,
                'closed': 'false' # Get only open markets
            }
            
            response = requests.get(BASE_URL, params=params)
            response.raise_for_status()
            
            markets_batch = response.json()
            
            if not markets_batch:
                print(f"Collection complete. Found {len(all_matching_markets)} relevant markets.")
                break
                
            for market in markets_batch:
                
                if market.get('archived'):
                    continue

                question_lower = market.get('question', '').lower()
                is_middle_east_market = any(keyword in question_lower for keyword in MIDDLE_EAST_KEYWORDS)

                if not is_middle_east_market:
                    continue
                
                try:
                    volume_24hr = float(market.get('volume24hr', 0))
                except (ValueError, TypeError):
                    volume_24hr = 0.0

                # --- Handle *both* API price formats ---
                answers = []
                
                # Format 1: New markets (use 'tokens' array)
                if 'tokens' in market:
                    for token in market.get('tokens', []):
                        try:
                            outcome_name = token.get('outcome', 'N/A')
                            outcome_price = float(token.get('price', 0))
                            outcome_percent = outcome_price * 100
                            answers.append(f"{outcome_name}: ${outcome_price:.2f} ({outcome_percent:.0f}%)")
                        except Exception:
                            continue
                
                # Format 2: Old markets (use stringified 'outcomes'/'outcomePrices')
                elif 'outcomes' in market and 'outcomePrices' in market:
                    try:
                        outcomes_list = json.loads(market['outcomes'])
                        prices_list = json.loads(market['outcomePrices'])
                        
                        for outcome, price_str in zip(outcomes_list, prices_list):
                            outcome_price = float(price_str)
                            outcome_percent = outcome_price * 100
                            answers.append(f"{outcome}: ${outcome_price:.2f} ({outcome_percent:.0f}%)")
                    except Exception as e:
                        print(f"(Warning: could not parse old prices for market {market.get('id')})")

                
                market_data = {
                    "id": market.get('id'),
                    "question": market.get('question', 'N/A'),
                    "volume_24hr": volume_24hr,
                    "liquidity": float(market.get('liquidityNum', 0)),
                    "answers": answers,
                    "url": f"https://polymarket.com/market/{market.get('slug', '')}"
                }
                
                all_matching_markets.append(market_data)
            
            offset += limit
            time.sleep(0.1) 

        except requests.exceptions.RequestException as e:
            print(f"\n--- ERROR fetching markets: {e} ---")
            break
        except json.JSONDecodeError:
            print(f"\n--- ERROR decoding JSON from server. ---")
            break

    # --- The Sort ---
    print("Sorting markets by 24h volume (high to low)...")
    sorted_markets = sorted(
        all_matching_markets, 
        key=lambda m: m['volume_24hr'], 
        reverse=True
    )
    
    return sorted_markets

def main():
    """
    Main function to run the script.
    """
    print(f"--- Starting market data update at {time.ctime()} ---")
    
    sorted_markets = fetch_all_markets()
    
    # Save all collected data to a JSON file
    try:
        with open('data.json', 'w', encoding='utf-8') as f:
            json.dump(sorted_markets, f, indent=2, ensure_ascii=False)
        print(f"Successfully saved {len(sorted_markets)} markets to data.json")
    except Exception as e:
        print(f"--- ERROR saving data to JSON file: {e} ---")

if __name__ == "__main__":
    main()