import requests
import json
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
import time
import re

# --- CONFIGURATION ---
BASE_URL = "https://gamma-api.polymarket.com/markets"
DATA_FILE = 'data.json'

# Strict Topic Buckets
TOPICS = {
    'GAZA': ['gaza', 'hamas', 'rafah', 'sinwar', 'hostage', 'khan younis'],
    'NORTH': ['lebanon', 'hezbollah', 'beirut', 'nasrallah', 'litani', 'northern'],
    'IRAN': ['iran', 'tehran', 'nuclear', 'khamenei', 'irgc'],
    'YEMEN': ['yemen', 'houthi', 'red sea', 'aden'],
    'US_RELATIONS': ['biden', 'trump', 'blinken', 'white house', 'aid'],
    'DOMESTIC': ['netanyahu', 'gantz', 'lapid', 'election', 'knesset', 'bengvir']
}

ESCALATION_TERMS = ['war', 'strike', 'attack', 'invasion', 'escalation', 'conflict', 'missile', 'bomb', 'military', 'operation', 'invade', 'killed', 'assassination']
DEESCALATION_TERMS = ['ceasefire', 'peace', 'truce', 'agreement', 'diplomacy', 'release', 'treaty', 'deal', 'negotiation', 'hostage release']

class MultiOutcomeBrain:
    def __init__(self):
        self.scaler = MinMaxScaler()

    def fetch_markets(self):
        print("--- 🧠 Starting Yes-Only Alpha Scan ---")
        all_markets = []
        limit = 100
        offset = 0
        
        for _ in range(5): 
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
                    
                    relevant = False
                    for topic_tags in TOPICS.values():
                        if any(t in text for t in topic_tags):
                            relevant = True
                            break
                    
                    if relevant:
                        all_markets.append(m)
                
                offset += limit
                time.sleep(0.2)
            except Exception as e:
                print(f"Fetch Error: {e}")
                break
        
        print(f"Captured {len(all_markets)} relevant markets.")
        return all_markets

    def assign_metadata(self, text):
        text = text.lower()
        topic = "GENERAL"
        for key, tags in TOPICS.items():
            if any(t in text for t in tags):
                topic = key
                break
        
        polarity = 0
        if any(t in text for t in ESCALATION_TERMS): polarity = 1
        if any(t in text for t in DEESCALATION_TERMS): polarity = -1
        
        return topic, polarity

    def process_outcomes(self, market):
        rows = []
        try:
            outcomes = json.loads(market.get('outcomes', '[]'))
            prices = json.loads(market.get('outcomePrices', '[]'))
            
            if not outcomes and 'tokens' in market:
                outcomes = ["Yes", "No"]
                p = float(market['tokens'][0].get('price', 0))
                prices = [p, 1-p]

            if not outcomes or not prices: return []
            
            vol = float(market.get('volume24hr') or 0)
            liq = float(market.get('liquidityNum') or 0)
            full_text = f"{market.get('question')} {market.get('description')}"
            topic, polarity = self.assign_metadata(full_text)

            # --- STANDARDIZATION LOGIC ---
            # Check if this is a standard Binary Market (Yes/No)
            # If so, we FORCE the loop to only accept "Yes" to ensure readability.
            is_binary = False
            if len(outcomes) == 2 and "Yes" in outcomes and "No" in outcomes:
                is_binary = True

            for i, outcome in enumerate(outcomes):
                try:
                    # READABILITY FIX: Skip 'No' outcomes in binary markets
                    if is_binary and outcome != "Yes":
                        continue

                    price = float(prices[i])
                    if price < 0.01: continue 

                    est_vol = vol * price 
                    est_liq = liq * price
                    
                    outcome_conf = 0
                    if est_liq > 10:
                        outcome_conf = est_vol / est_liq

                    rows.append({
                        'id': f"{market['id']}_{i}",
                        'question': market['question'],
                        'outcome_name': outcome,
                        'price': price,
                        'est_volume': est_vol,
                        'money_confidence': outcome_conf,
                        'topic': topic,
                        'polarity': polarity,
                        'slug': market.get('slug'),
                        'raw_json': json.dumps(market)
                    })
                except: continue
                
        except Exception as e:
            pass
            
        return rows

    def calculate_topic_momentum(self, df):
        if df.empty: return {}
        
        topic_pressure = {}
        for topic in TOPICS.keys():
            topic_rows = df[df['topic'] == topic]
            if topic_rows.empty:
                topic_pressure[topic] = 0
                continue
            
            active_rows = topic_rows[topic_rows['money_confidence'] > 0.5]
            if not active_rows.empty:
                pressure = (active_rows['money_confidence'] * active_rows['polarity']).mean()
                topic_pressure[topic] = pressure
            else:
                topic_pressure[topic] = 0
        return topic_pressure

    def predict_fair_value(self, row, topic_pressure):
        current = row['price']
        conf = row['money_confidence']
        topic = row['topic']
        polarity = row['polarity']
        
        # 1. Whale Push
        whale_push = np.log1p(conf) * 0.10
        
        # 2. Topic Push
        pressure = topic_pressure.get(topic, 0)
        topic_impact = pressure * polarity * 0.15 
        
        fair_value = current + whale_push + topic_impact
        
        # Clamps
        fair_value = max(current - 0.2, min(current + 0.2, fair_value))
        fair_value = max(0.01, min(0.99, fair_value))
        
        return fair_value

    def generate_narrative(self, row, pressure):
        outcome = row['outcome_name']
        conf = row['money_confidence']
        topic = row['topic']
        pol = row['polarity']
        
        text = f"Analyzing <b>{outcome}</b>. "
        
        if conf > 1.5:
            text += f"🐳 <b>Whale Alert:</b> Volume > Liquidity ({conf:.1f}x). "
        elif conf > 0.8:
            text += "Solid institutional volume. "
            
        if abs(pressure) > 0.1:
            if pressure > 0: trend = "Escalation"
            else: trend = "De-escalation"
            
            is_aligned = (pressure > 0 and pol == 1) or (pressure < 0 and pol == -1)
            
            if is_aligned and pol != 0:
                text += f"✅ Aligns with <b>{topic} {trend}</b>. "
            elif not is_aligned and pol != 0:
                text += f"⚠️ Contradicts <b>{topic} {trend}</b>. "
        
        return text

    def run_analysis(self):
        raw_markets = self.fetch_markets()
        
        all_rows = []
        for m in raw_markets:
            outcomes = self.process_outcomes(m)
            all_rows.extend(outcomes)
            
        df = pd.DataFrame(all_rows)
        if df.empty: return []
        
        print("Calculating Semantic Correlations...")
        topic_pressure = self.calculate_topic_momentum(df)
        
        results = []
        for _, row in df.iterrows():
            if row['money_confidence'] < 0.3 and row['price'] < 0.4:
                continue
                
            pred = self.predict_fair_value(row, topic_pressure)
            
            gap = abs(pred - row['price'])
            score = (gap * 0.7) + (row['money_confidence'] * 0.3)
            norm_score = min(score * 2, 1.0)
            hue = 120 * (1 - norm_score)
            
            results.append({
                "question": row['question'],
                "outcome": row['outcome_name'],
                "price": f"{row['price']:.2f}",
                "predicted": f"{pred:.2f}",
                "delta": pred - row['price'],
                "score_float": norm_score,
                "color": f"hsl({hue:.0f}, 90%, 45%)",
                "narrative": self.generate_narrative(row, topic_pressure.get(row['topic'], 0)),
                "url": f"https://polymarket.com/market/{row['slug']}",
                "raw_data": row['raw_json']
            })
            
        results.sort(key=lambda x: x['score_float'], reverse=True)
        return results

def main():
    brain = MultiOutcomeBrain()
    results = brain.run_analysis()
    with open(DATA_FILE, 'w') as f:
        json.dump(results, f, indent=2)
    print("Done.")

if __name__ == "__main__":
    main()