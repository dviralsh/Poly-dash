import requests
import time
import json
import os
import math
from datetime import datetime, timedelta

# Official "Gamma" API endpoint for markets
BASE_URL = "https://gamma-api.polymarket.com/markets"

# Our keywords for "Middle East" filtering
MIDDLE_EAST_KEYWORDS = [
    'middle east', 'israel', 'gaza', 'hamas', 'hezbollah', 'idf',
    'iran', 'lebanon', 'syria', 'yemen', 'houthi', 'palestin', # Catches 'palestine'/'palestinian'
    'saudi', 'uae', 'qatar', 'red sea', 'netanyahu', 'sinwar'
]

STATE_FILE = 'model_state.json'
DATA_FILE = 'data.json'

class MarketPredictor:
    def __init__(self):
        self.state = {
            "weights": {
                "bias": 0.0,
                "price": 0.5,
                "whale_index": 0.1,
                "one_day_change": 0.2,
                "spread": -0.1
            },
            "pending_predictions": [],
            "learning_rate": 0.01,
            "last_run": 0
        }
        self.load_state()

    def load_state(self):
        if os.path.exists(STATE_FILE):
            try:
                with open(STATE_FILE, 'r') as f:
                    self.state = json.load(f)
                    print("Loaded model state.")
            except Exception as e:
                print(f"Could not load state: {e}. Starting fresh.")

    def save_state(self):
        try:
            with open(STATE_FILE, 'w') as f:
                json.dump(self.state, f, indent=2)
            print("Saved model state.")
        except Exception as e:
            print(f"Error saving state: {e}")

    def predict(self, features):
        w = self.state["weights"]
        # Simple linear regression: y = w*x + b
        # Ensure result is between 0 and 1 (probability)
        raw_pred = (
            w["bias"] +
            w["price"] * features["price"] +
            w["whale_index"] * features["whale_index"] +
            w["one_day_change"] * features["one_day_change"] +
            w["spread"] * features["spread"]
        )
        return max(0.0, min(1.0, raw_pred))

    def update_weights(self, features, predicted, actual):
        # Gradient Descent: w = w + lr * (error) * input
        error = actual - predicted
        lr = self.state["learning_rate"]
        w = self.state["weights"]
        
        w["bias"] += lr * error * 1.0
        w["price"] += lr * error * features["price"]
        w["whale_index"] += lr * error * features["whale_index"]
        w["one_day_change"] += lr * error * features["one_day_change"]
        w["spread"] += lr * error * features["spread"]
        
        self.state["weights"] = w
        print(f"  [Learning] Error: {error:.4f} -> Weights updated.")

    def process_market(self, market_id, current_price, features):
        current_time = time.time()
        
        # 1. Check for matured predictions to learn from
        remaining_predictions = []
        for p in self.state["pending_predictions"]:
            if p["market_id"] == market_id and p["target_time"] <= current_time:
                # This prediction has matured!
                print(f"  Validating prediction for {market_id}...")
                self.update_weights(p["features"], p["predicted_price"], current_price)
            else:
                remaining_predictions.append(p)
        
        self.state["pending_predictions"] = remaining_predictions

        # 2. Make new predictions
        times = {
            "15m": 15 * 60,
            "1h": 60 * 60,
            "24h": 24 * 60 * 60,
            "1w": 7 * 24 * 60 * 60
        }
        
        new_preds = {}
        for label, seconds in times.items():
            target_time = current_time + seconds
            pred_price = self.predict(features)
            
            # Record it for future validation
            self.state["pending_predictions"].append({
                "market_id": market_id,
                "target_time": target_time,
                "predicted_price": pred_price,
                "features": features,
                "label": label
            })
            new_preds[label] = pred_price
            
        return new_preds

def calculate_whale_index(market):
    """
    Identifies 'Whale' activity based on Volume-to-Liquidity ratio and Capital depth.
    A high ratio suggests heavy trading relative to market depth (aggressive positioning).
    """
    try:
        vol = float(market.get('volume24hr', 0) or 0)
        liq = float(market.get('liquidity', 0) or 0)
        
        if liq < 1: return 0
        
        # Ratio of volume to liquidity. 
        # Normal markets might be 0.1 - 0.5. 
        # Whale action might spike this > 1.0 or 2.0.
        ratio = vol / liq 
        
        # Logarithmic scale to dampen extreme outliers
        return math.log1p(ratio) 
    except:
        return 0

def fetch_all_markets():
    """
    Collects markets, filters, detects whales, and runs ML predictions.
    """
    predictor = MarketPredictor()
    all_matching_markets = []
    limit = 100
    offset = 0
    
    print("Starting collection, whale detection, and ML forecasting...")

    while True:
        try:
            params = {
                'limit': limit,
                'offset': offset,
                'closed': 'false' 
            }
            
            response = requests.get(BASE_URL, params=params)
            response.raise_for_status()
            
            markets_batch = response.json()
            
            if not markets_batch:
                print(f"Collection complete. Found {len(all_matching_markets)} relevant markets.")
                break
                
            for market in markets_batch:
                if market.get('archived'): continue

                question_lower = market.get('question', '').lower()
                is_middle_east_market = any(keyword in question_lower for keyword in MIDDLE_EAST_KEYWORDS)

                if not is_middle_east_market: continue
                
                # --- Feature Extraction ---
                try:
                    # Parse primary price (usually the 'yes' token or first outcome)
                    # For simplicity in this ML demo, we track the FIRST outcome's price.
                    # Complex markets might require tracking all outcomes.
                    current_price = 0.5 # Default fallback
                    tokens = market.get('tokens', [])
                    if tokens:
                        current_price = float(tokens[0].get('price', 0.5))
                    elif 'outcomePrices' in market:
                        # Fallback for old markets format
                        try:
                            prices = json.loads(market.get('outcomePrices', '[]'))
                            if prices: current_price = float(prices[0])
                        except: pass
                    
                    vol_24hr = float(market.get('volume24hr', 0) or 0)
                    liquidity = float(market.get('liquidityNum', 0) or 0)
                    change_24h = float(market.get('oneDayPriceChange', 0) or 0)
                    
                    # Spread calculation (Bid/Ask)
                    best_bid = float(market.get('bestBid', 0) or 0)
                    best_ask = float(market.get('bestAsk', 0) or 0)
                    spread = best_ask - best_bid
                    
                    whale_idx = calculate_whale_index(market)
                    
                    features = {
                        "price": current_price,
                        "whale_index": whale_idx,
                        "one_day_change": change_24h,
                        "spread": spread
                    }
                    
                    # --- ML & Prediction ---
                    # Learn from past & Predict future
                    predictions = predictor.process_market(market.get('id'), current_price, features)
                    
                except Exception as e:
                    print(f"Skipping ML for market {market.get('id')}: {e}")
                    continue

                # --- Formatting Output ---
                answers = []
                # ... same formatting logic as before ...
                if 'tokens' in market:
                    for token in market.get('tokens', []):
                        try:
                            outcome_name = token.get('outcome', 'N/A')
                            outcome_price = float(token.get('price', 0))
                            outcome_percent = outcome_price * 100
                            answers.append(f"{outcome_name}: ${outcome_price:.2f} ({outcome_percent:.0f}%)")
                        except: continue
                elif 'outcomes' in market and 'outcomePrices' in market:
                    try:
                        outcomes_list = json.loads(market['outcomes'])
                        prices_list = json.loads(market['outcomePrices'])
                        for outcome, price_str in zip(outcomes_list, prices_list):
                            outcome_price = float(price_str)
                            outcome_percent = outcome_price * 100
                            answers.append(f"{outcome}: ${outcome_price:.2f} ({outcome_percent:.0f}%)")
                    except: pass

                market_data = {
                    "id": market.get('id'),
                    "question": market.get('question', 'N/A'),
                    "volume_24hr": vol_24hr,
                    "liquidity": liquidity,
                    "whale_index": f"{whale_idx:.2f}",
                    "answers": answers,
                    "predictions": {k: f"{v:.2f}" for k,v in predictions.items()},
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

    # Save ML State
    predictor.save_state()

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