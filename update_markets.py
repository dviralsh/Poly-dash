import requests
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import time

# --- CONFIGURATION ---
BASE_URL = "https://gamma-api.polymarket.com/markets"
DATA_FILE = 'data.json'

# Expanded Keywords to catch more markets
KEYWORDS = [
    'middle east', 'israel', 'gaza', 'hamas', 'hezbollah', 'idf',
    'iran', 'lebanon', 'syria', 'yemen', 'houthi', 'palestin',
    'saudi', 'uae', 'qatar', 'red sea', 'netanyahu', 'sinwar', 
    'ceasefire', 'tehran', 'beirut', 'jerusalem', 'tel aviv',
    'war', 'peace', 'hostage', 'biden', 'trump' # Broader context
]

class AdvancedMarketBrain:
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.vectorizer = TfidfVectorizer(stop_words='english')

    def fetch_markets(self):
        """
        Fetches markets with robust pagination and deep keyword searching.
        """
        print("--- 🧠 Starting Deep Market Scan ---")
        all_markets = []
        limit = 100
        offset = 0
        
        # Scan up to 500 active markets to build a solid dataset
        for _ in range(5): 
            try:
                params = {
                    'limit': limit, 
                    'offset': offset, 
                    'closed': 'false',
                    'order': 'volume24hr', # Get active ones first
                    'ascending': 'false'
                }
                resp = requests.get(BASE_URL, params=params).json()
                
                if not resp: break
                
                for m in resp:
                    if m.get('archived'): continue
                    
                    # Fuzzy Search: Check Question + Description + Tags
                    text_corpus = (
                        str(m.get('question')) + " " + 
                        str(m.get('description')) + " " + 
                        str(m.get('tags'))
                    ).lower()
                    
                    if any(k in text_corpus for k in KEYWORDS):
                        all_markets.append(m)
                
                offset += limit
                time.sleep(0.2) # API Rate limit kindness
                
            except Exception as e:
                print(f"Fetch Error: {e}")
                break
        
        print(f"Captured {len(all_markets)} relevant markets.")
        return all_markets

    def parse_price(self, market):
        """
        FIXED: Robustly extracts the 'Yes' price or main outcome price.
        """
        try:
            # Method 1: OutcomePrices array (Most reliable)
            if 'outcomePrices' in market:
                prices = json.loads(market['outcomePrices'])
                return float(prices[0]) # Usually 'Yes' or first option
            
            # Method 2: Token Price
            if 'tokens' in market and len(market['tokens']) > 0:
                return float(market['tokens'][0].get('price', 0))
                
            return 0.0
        except:
            return 0.0

    def analyze_clusters(self, df):
        """
        Calculates 'Peer Pressure'. 
        If similar markets are moving, boost the score of this market.
        """
        if df.empty: return df
        
        # Vectorize Questions
        tfidf = self.vectorizer.fit_transform(df['question'])
        similarity = cosine_similarity(tfidf)
        
        peer_momentum = []
        
        for idx in range(len(df)):
            # Find similar markets (similarity > 0.2)
            sim_scores = similarity[idx]
            peers = np.where(sim_scores > 0.2)[0]
            
            if len(peers) > 1:
                # Average volume change of PEERS (excluding self)
                peer_vol = df.iloc[peers]['change_abs'].mean()
                peer_momentum.append(peer_vol)
            else:
                peer_momentum.append(0)
                
        df['peer_momentum'] = peer_momentum
        return df

    def run_multi_factor_model(self, df):
        """
        The Core Logic: Combines 4 Factors into one Anomaly Score.
        """
        if len(df) < 5: return df # Not enough data for stats

        # --- Factor 1: Liquidity Stress ---
        # Low Liquidity + High Volume = Explosive Potentail
        # We use log scale to normalize
        df['liquidity_stress'] = np.log1p(df['volume']) / (np.log1p(df['liquidity']) + 1)

        # --- Factor 2: Price Velocity ---
        # 24h Change weighted by spread (Change is more significant if spread is tight)
        df['velocity'] = df['change_abs'] * (1 - df['spread'])

        # --- Factor 3: ML Anomaly Detection ---
        features = ['velocity', 'liquidity_stress', 'peer_momentum', 'spread']
        X = df[features].fillna(0)
        
        iso = IsolationForest(contamination=0.15, random_state=42)
        iso.fit(X)
        # Invert scores: -1 (anomaly) becomes high score
        raw_anomaly = iso.decision_function(X) * -1
        df['ml_score'] = self.scaler.fit_transform(raw_anomaly.reshape(-1, 1)).flatten()

        # --- FINAL COMPOSITE SCORE ---
        # We blend the ML "Black Box" score with our transparent "White Box" factors
        # This creates a more stable, explainable signal.
        df['final_score'] = (
            (df['ml_score'] * 0.4) +         # ML detection (40%)
            (df['liquidity_stress'] * 0.3) + # Whale/Volume (30%)
            (df['peer_momentum'] * 0.2) +    # Cluster Trend (20%)
            (df['velocity'] * 0.1)           # Pure Price Action (10%)
        )
        
        # Normalize final score to 0-1
        df['final_score'] = self.scaler.fit_transform(df[['final_score']]).flatten()
        
        return df

    def get_color(self, score):
        # Continuous Gradient: Green -> Yellow -> Red
        # Hue: 120 (Green) -> 60 (Yellow) -> 0 (Red)
        hue = 120 * (1 - score)
        return f"hsl({hue:.0f}, 100%, 40%)"

def main():
    brain = AdvancedMarketBrain()
    
    # 1. Fetch
    raw_markets = brain.fetch_markets()
    if not raw_markets: return

    # 2. Parse Data
    rows = []
    for m in raw_markets:
        try:
            price = brain.parse_price(m)
            if price == 0: continue # Skip broken markets
            
            vol = float(m.get('volume24hr') or 0)
            liq = float(m.get('liquidityNum') or 0)
            change = float(m.get('oneDayPriceChange', 0) or 0)
            
            # Spread (Bid-Ask)
            bid = float(m.get('bestBid') or 0)
            ask = float(m.get('bestAsk') or 0)
            spread = ask - bid if (ask > 0 and bid > 0) else 0.01

            rows.append({
                'id': m['id'],
                'question': m['question'],
                'price': price,
                'volume': vol,
                'liquidity': liq,
                'change_abs': abs(change),
                'spread': spread,
                'slug': m.get('slug')
            })
        except: continue
        
    df = pd.DataFrame(rows)
    
    # 3. Intelligent Processing
    print("running Cluster Analysis...")
    df = brain.analyze_clusters(df)
    
    print("Running Multi-Factor Model...")
    df = brain.run_multi_factor_model(df)
    
    # 4. Export
    output = []
    for _, row in df.iterrows():
        score = float(row['final_score'])
        
        # Generate Text Analysis based on what triggered the score
        reasons = []
        if row['peer_momentum'] > df['peer_momentum'].mean(): reasons.append("Cluster Trend")
        if row['liquidity_stress'] > df['liquidity_stress'].mean() + 0.5: reasons.append("Volume Spike")
        if row['ml_score'] > 0.8: reasons.append("Stat Anomaly")
        
        if not reasons: reasons.append("Normal Trading")

        output.append({
            "question": row['question'],
            "price": f"{row['price']:.2f}", # REAL Price now
            "score_float": score,
            "color": brain.get_color(score),
            "analysis": " + ".join(reasons[:2]), # Keep it short
            "url": f"https://polymarket.com/market/{row['slug']}",
            "volume": row['volume']
        })
        
    # Sort by Signal Strength
    output.sort(key=lambda x: x['score_float'], reverse=True)
    
    with open(DATA_FILE, 'w') as f:
        json.dump(output, f, indent=2)
    print("Done.")

if __name__ == "__main__":
    main()