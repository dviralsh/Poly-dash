import requests
import json
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import time

# --- CONFIGURATION ---
BASE_URL = "https://gamma-api.polymarket.com/markets"
DATA_FILE = 'data.json'

KEYWORDS = [
    'middle east', 'israel', 'gaza', 'hamas', 'hezbollah', 'idf',
    'iran', 'lebanon', 'syria', 'yemen', 'houthi', 'palestin',
    'saudi', 'uae', 'qatar', 'red sea', 'netanyahu', 'sinwar', 
    'ceasefire', 'tehran', 'beirut', 'jerusalem', 'tel aviv',
    'war', 'peace', 'hostage', 'biden', 'trump', 'nuclear', 'missile'
]

class AlphaMarketBrain:
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.vectorizer = TfidfVectorizer(stop_words='english')

    def fetch_markets(self):
        print("--- 🧠 Starting Smart Money Scan ---")
        all_markets = []
        limit = 100
        offset = 0
        
        # Deep Scan (6 Pages)
        for _ in range(6): 
            try:
                params = {
                    'limit': limit, 'offset': offset, 'closed': 'false',
                    'order': 'volume24hr', 'ascending': 'false'
                }
                resp = requests.get(BASE_URL, params=params).json()
                if not resp: break
                
                for m in resp:
                    if m.get('archived'): continue
                    text = (str(m.get('question')) + str(m.get('description'))).lower()
                    if any(k in text for k in KEYWORDS):
                        all_markets.append(m)
                
                offset += limit
                time.sleep(0.2)
            except Exception as e:
                print(f"Fetch Error: {e}")
                break
        
        print(f"Captured {len(all_markets)} raw markets.")
        return all_markets

    def get_best_outcome(self, market):
        """ Returns the most relevant outcome/price tuple """
        try:
            outcomes = json.loads(market.get('outcomes', '[]'))
            prices = json.loads(market.get('outcomePrices', '[]'))
            if not outcomes or not prices: return None
            
            prices = [float(p) for p in prices]
            best_idx = np.argmax(prices) # Track the favorite
            return outcomes[best_idx], prices[best_idx]
        except:
            return "Yes", 0.5

    def calculate_fair_value(self, current_price, confidence, trend_24h, peer_momentum):
        """
        The "Conservative Alpha" Engine.
        Calculates 'Fair Value' by adjusting Market Price based on 'Smart Money' flow.
        """
        # 1. Determine Direction of Whale Flow
        # If Price moved UP with high volume -> Buying Pressure (Positive)
        # If Price moved DOWN with high volume -> Selling Pressure (Negative)
        # We use the 24h trend to determine the SIGN of the whale vector.
        
        # If trend is 0 (flat), we assume momentum follows the peer group
        if abs(trend_24h) < 0.01:
            flow_direction = 1 if peer_momentum > 0 else -1
        else:
            flow_direction = 1 if trend_24h > 0 else -1

        # 2. Calculate "Smart Weight" (Logarithmic Dampening)
        # Confidence (Vol/Liq) usually ranges 0.1 to 10.0
        # We want a max influence of around 20% (0.20)
        # log1p(10) ~= 2.4.  2.4 * 0.08 ~= 0.19.
        whale_impact = np.log1p(confidence) * 0.08 * flow_direction

        # 3. Peer Influence (Cluster Gravity)
        # If all related markets are moving, this one should too.
        # Max influence ~10%
        cluster_impact = peer_momentum * 0.5 

        # 4. Fair Value Calculation
        # We anchor to the Current Price, then tilt it.
        fair_value = current_price + whale_impact + cluster_impact
        
        # 5. Sanity Limits (The "No Hallucination" Clause)
        # The AI cannot predict a price more than 25% away from reality 
        # unless conviction is absolutely massive.
        max_deviation = 0.25
        fair_value = max(current_price - max_deviation, min(current_price + max_deviation, fair_value))
        
        # Hard clamps for probability bounds
        fair_value = max(0.02, min(0.98, fair_value))
        
        # Determine Label
        direction = "Hold"
        if fair_value > current_price + 0.03: direction = "Up"
        elif fair_value < current_price - 0.03: direction = "Down"
        
        return fair_value, direction

    def explain_prediction(self, row):
        current = row['price']
        target = row['predicted_price']
        outcome = row['outcome_name']
        diff = target - current
        
        text = f"Analyzing <b>'{outcome}'</b> ({current*100:.0f}%). "
        
        # Explaining the Logic
        reasons = []
        
        # Whale Logic
        if row['money_confidence'] > 1.5:
            action = "accumulating" if row['change_raw'] >= 0 else "dumping"
            reasons.append(f"Smart money is {action} (High Vol/Liq).")
        
        # Peer Logic
        if abs(row['peer_momentum']) > 0.05:
            trend = "bullish" if row['peer_momentum'] > 0 else "bearish"
            reasons.append(f"Sector sentiment is {trend}.")

        if reasons:
            text += " ".join(reasons)
            
            # The Prediction Conclusion
            if row['direction'] == "Up":
                text += f" 📈 <b>Adjustment:</b> Buying pressure suggests true value is higher (~{target*100:.0f}%)."
            elif row['direction'] == "Down":
                text += f" 📉 <b>Adjustment:</b> Selling pressure suggests true value is lower (~{target*100:.0f}%)."
            else:
                text += " Market appears efficiently priced."
        else:
            text += "No significant anomalies. Market price reflects consensus."
            
        return text

    def run_analysis(self, raw_markets):
        rows = []
        for m in raw_markets:
            try:
                outcome_data = self.get_best_outcome(m)
                if not outcome_data: continue
                name, price = outcome_data
                if price == 0: continue

                vol = float(m.get('volume24hr') or 0)
                liq = float(m.get('liquidityNum') or 0)
                change = float(m.get('oneDayPriceChange', 0) or 0)
                
                # Confidence Metric
                money_confidence = 0
                if liq > 0: money_confidence = vol / liq
                
                spread = float(m.get('bestAsk',0)) - float(m.get('bestBid',0))

                rows.append({
                    'id': m['id'],
                    'question': m['question'],
                    'outcome_name': name,
                    'price': price,
                    'volume': vol,
                    'liquidity': liq,
                    'change_raw': change, # Signed Change (Direction)
                    'change_abs': abs(change),
                    'spread': spread,
                    'money_confidence': money_confidence,
                    'slug': m.get('slug'),
                    'raw_json': json.dumps(m)
                })
            except: continue
            
        df = pd.DataFrame(rows)
        if df.empty: return []

        # Cluster Analysis
        print("Analyzing clusters...")
        if not df.empty:
            tfidf = self.vectorizer.fit_transform(df['question'])
            sim = cosine_similarity(tfidf)
            peers_mom = []
            for i in range(len(df)):
                peers = np.where(sim[i] > 0.25)[0]
                if len(peers) > 1:
                    others = [p for p in peers if p != i]
                    peers_mom.append(df.iloc[others]['change_raw'].mean())
                else: peers_mom.append(0)
            df['peer_momentum'] = peers_mom
        else:
            df['peer_momentum'] = 0

        # Run "Fair Value" Model
        predictions = []
        directions = []
        for _, row in df.iterrows():
            pred, direct = self.calculate_fair_value(
                row['price'], 
                row['money_confidence'], 
                row['change_raw'], # Pass direction!
                row['peer_momentum']
            )
            predictions.append(pred)
            directions.append(direct)
            
        df['predicted_price'] = predictions
        df['direction'] = directions

        # Score based on how "Active" the situation is
        # We combine Spread + Confidence + Gap to find interesting markets
        df['gap'] = abs(df['predicted_price'] - df['price'])
        df['final_score'] = (
            (df['gap'] * 0.5) + 
            (np.log1p(df['money_confidence']) * 0.3) +
            (df['change_abs'] * 0.2)
        )
        df['final_score'] = self.scaler.fit_transform(df[['final_score']]).flatten()

        output = []
        for _, row in df.iterrows():
            score = float(row['final_score'])
            hue = 120 * (1 - score)
            
            output.append({
                "question": row['question'],
                "outcome": row['outcome_name'],
                "price": f"{row['price']:.2f}",
                "predicted": f"{row['predicted_price']:.2f}",
                "delta": row['predicted_price'] - row['price'],
                "score_float": score,
                "color": f"hsl({hue:.0f}, 90%, 45%)",
                "narrative": self.explain_prediction(row),
                "url": f"https://polymarket.com/market/{row['slug']}",
                "raw_data": row['raw_json']
            })
            
        output.sort(key=lambda x: x['score_float'], reverse=True)
        return output

def main():
    brain = AlphaMarketBrain()
    markets = brain.fetch_markets()
    results = brain.run_analysis(markets)
    with open(DATA_FILE, 'w') as f:
        json.dump(results, f, indent=2)
    print("Done.")

if __name__ == "__main__":
    main()