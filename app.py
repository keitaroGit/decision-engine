import os, requests, json, time, re
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)

ALPHA_KEY  = os.environ.get('ALPHA_VANTAGE_KEY', '')
FRED_KEY   = os.environ.get('FRED_API_KEY', '')
CLAUDE_KEY = os.environ.get('CLAUDE_API_KEY', '')

client = Anthropic(api_key=CLAUDE_KEY)

cache = {}
CACHE_TTL = 900

def cached(key, fn):
    now = time.time()
    if key in cache and now - cache[key]['ts'] < CACHE_TTL:
        return cache[key]['data']
    data = fn()
    cache[key] = {'data': data, 'ts': now}
    return data

def av(params):
    params['apikey'] = ALPHA_KEY
    try:
        r = requests.get('https://www.alphavantage.co/query', params=params, timeout=10)
        return r.json()
    except:
        return {}

def get_quote(ticker):
    def fn():
        d = av({'function': 'GLOBAL_QUOTE', 'symbol': ticker})
        q = d.get('Global Quote', {})
        return {
            'price':      q.get('05. price', 'N/A'),
            'change_pct': q.get('10. change percent', 'N/A'),
            'volume':     q.get('06. volume', 'N/A'),
        }
    return cached('quote_' + ticker, fn)

def get_overview(ticker):
    def fn():
        d = av({'function': 'OVERVIEW', 'symbol': ticker})
        return {
            'name':           d.get('Name', ticker),
            'sector':         d.get('Sector', 'N/A'),
            'industry':       d.get('Industry', 'N/A'),
            'pe_ratio':       d.get('PERatio', 'N/A'),
            'pb_ratio':       d.get('PriceToBookRatio', 'N/A'),
            'ps_ratio':       d.get('PriceToSalesRatioTTM', 'N/A'),
            'profit_margin':  d.get('ProfitMargin', 'N/A'),
            'op_margin':      d.get('OperatingMarginTTM', 'N/A'),
            'revenue_growth': d.get('QuarterlyRevenueGrowthYOY', 'N/A'),
            'eps_growth':     d.get('QuarterlyEarningsGrowthYOY', 'N/A'),
            'eps_ttm':        d.get('EPS', 'N/A'),
            'beta':           d.get('Beta', 'N/A'),
            'mkt_cap':        d.get('MarketCapitalization', 'N/A'),
            '52w_high':       d.get('52WeekHigh', 'N/A'),
            '52w_low':        d.get('52WeekLow', 'N/A'),
            'analyst_target': d.get('AnalystTargetPrice', 'N/A'),
        }
    return cached('overview_' + ticker, fn)

def get_earnings(ticker):
    def fn():
        d = av({'function': 'EARNINGS', 'symbol': ticker})
        quarters = d.get('quarterlyEarnings', [])[:3]
        result = []
        for q in quarters:
            result.append({
                'date':         q.get('fiscalDateEnding', ''),
                'reported_eps': q.get('reportedEPS', ''),
                'est_eps':      q.get('estimatedEPS', ''),
                'surprise':     q.get('surprisePercentage', ''),
            })
        return result
    return cached('earnings_' + ticker, fn)

def fred_val(series_id):
    def fn():
        try:
            r = requests.get(
                'https://api.stlouisfed.org/fred/series/observations',
                params={
                    'series_id': series_id,
                    'api_key': FRED_KEY,
                    'file_type': 'json',
                    'limit': 1,
                    'sort_order': 'desc'
                },
                timeout=8
            )
            obs = r.json().get('observations', [])
            return obs[0].get('value', 'N/A') if obs else 'N/A'
        except:
            return 'N/A'
    return cached('fred_' + series_id, fn)

def get_macro():
    return {
        'us10y':  fred_val('DGS10'),
        'oil':    fred_val('DCOILWTICO'),
        'gold':   fred_val('GOLDAMGBD228NLBM'),
        'usdyen': fred_val('DEXJPUS'),
        'eurusd': fred_val('DEXUSEU'),
    }

SYSTEM_PROMPT = """You are an investment analyst using a 3-layer framework.
Respond ONLY with a JSON object. No markdown. No explanation. No text before or after the JSON.
Use only simple ASCII characters in all string values. No smart quotes, no em dashes, no special characters.
Use simple apostrophes (') not curly quotes. Use hyphens (-) not dashes.

The JSON must follow this exact structure:
{"verdict":"BUY","verdict_ja":"migi","confidence":70,"summary_en":"text","summary_ja":"text","layer1":{"score":6,"signal":"NEUTRAL","key_points_en":["p1","p2"],"key_points_ja":["p1","p2"]},"layer2":{"score":7,"signal":"STRONG","key_points_en":["p1","p2"],"key_points_ja":["p1","p2"]},"layer3":{"score":5,"signal":"FAIR","key_points_en":["p1","p2"],"key_points_ja":["p1","p2"]},"distortion":{"found":false,"description_en":null,"description_ja":null},"risks_en":["r1","r2"],"risks_ja":["r1","r2"],"catalysts_en":["c1","c2"],"catalysts_ja":["c1","c2"]}

Allowed verdict values: STRONG BUY, BUY, WATCH, PASS, STRONG PASS
Allowed verdict_ja values: kyoi kaikomi, kaikomi, yousu mi, miokuri, kyoi miokuri
Allowed layer1 signal: TAILWIND, NEUTRAL, HEADWIND
Allowed layer2 signal: STRONG, NEUTRAL, WEAK
Allowed layer3 signal: UNDERVALUED, FAIR, OVERVALUED"""

def run_analysis(ticker, overview, quote, earnings, macro, horizon='mid'):
    horizon_map = {
        'short': 'SHORT TERM 1-3 months: focus on momentum and near-term catalysts',
        'mid':   'MID TERM 3-12 months: focus on earnings trajectory and margins',
        'long':  'LONG TERM 1-3 years: focus on competitive moat and macro cycle',
    }
    htext = horizon_map.get(horizon, horizon_map['mid'])

    lines = [
        "Analyze " + ticker + " | Horizon: " + htext,
        "Sector: " + str(overview.get('sector', '')),
        "Price: " + str(quote.get('price', '')) + " | Change: " + str(quote.get('change_pct', '')),
        "52W High: " + str(overview.get('52w_high', '')) + " | Low: " + str(overview.get('52w_low', '')) + " | Target: " + str(overview.get('analyst_target', '')),
        "PE: " + str(overview.get('pe_ratio', '')) + " | PB: " + str(overview.get('pb_ratio', '')) + " | PS: " + str(overview.get('ps_ratio', '')),
        "EPS: " + str(overview.get('eps_ttm', '')) + " | EPS Growth: " + str(overview.get('eps_growth', '')),
        "Rev Growth: " + str(overview.get('revenue_growth', '')) + " | Op Margin: " + str(overview.get('op_margin', '')),
        "Beta: " + str(overview.get('beta', '')) + " | Profit Margin: " + str(overview.get('profit_margin', '')),
        "Earnings surprises: " + str(earnings),
        "US 10Y: " + str(macro.get('us10y', '')) + "% | Oil: " + str(macro.get('oil', '')) + " | USD/JPY: " + str(macro.get('usdyen', '')),
    ]
    prompt = "\n".join(lines) + "\n\nOutput JSON only."

    try:
        msg = client.messages.create(
            model='claude-sonnet-4-6',
            max_tokens=1200,
            system=SYSTEM_PROMPT,
            messages=[{'role': 'user', 'content': prompt}]
        )
        raw = msg.content[0].text.strip()

        # Extract JSON
        start = raw.find('{')
        end = raw.rfind('}')
        if start == -1 or end == -1:
            return {'error': 'No JSON found in response'}
        text = raw[start:end + 1]

        # Clean non-ASCII
        text = text.encode('ascii', 'ignore').decode('ascii')

        # Fix common JSON issues
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', text)

        return json.loads(text)

    except json.JSONDecodeError as e:
        return {'error': 'JSON parse error: ' + str(e)}
    except Exception as e:
        return {'error': str(e)}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/analyze', methods=['POST'])
def analyze():
    data    = request.get_json()
    ticker  = data.get('ticker', '').upper().strip()
    horizon = data.get('horizon', 'mid')

    if not ticker:
        return jsonify({'error': 'Ticker required'}), 400

    overview = get_overview(ticker)
    quote    = get_quote(ticker)
    earnings = get_earnings(ticker)
    macro    = get_macro()

    if overview.get('name') == ticker and overview.get('sector') == 'N/A':
        return jsonify({'error': 'Ticker not found: ' + ticker}), 404

    result = run_analysis(ticker, overview, quote, earnings, macro, horizon)

    if 'error' in result:
        return jsonify(result), 500

    return jsonify({
        'ticker':   ticker,
        'name':     overview.get('name'),
        'price':    quote.get('price'),
        'change':   quote.get('change_pct'),
        'horizon':  horizon,
        'analysis': result,
        'raw': {
            'overview': overview,
            'quote':    quote,
            'earnings': earnings,
            'macro':    macro,
        }
    })

@app.route('/api/health')
def health():
    return jsonify({'status': 'ok', 'ticker_cache': len(cache)})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
