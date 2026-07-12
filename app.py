import os, requests, json, time
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

# ── Cache ─────────────────────────────────────────────────────────────────────
cache = {}
CACHE_TTL = 900  # 15 min

def cached(key, fn):
    now = time.time()
    if key in cache and now - cache[key]['ts'] < CACHE_TTL:
        return cache[key]['data']
    data = fn()
    cache[key] = {'data': data, 'ts': now}
    return data

# ── Alpha Vantage helpers ─────────────────────────────────────────────────────
def av(params):
    params['apikey'] = ALPHA_KEY
    try:
        r = requests.get('https://www.alphavantage.co/query', params=params, timeout=10)
        return r.json()
    except:
        return {}

def get_quote(ticker):
    def fn():
        d = av({'function':'GLOBAL_QUOTE','symbol':ticker})
        q = d.get('Global Quote', {})
        return {
            'price':       q.get('05. price', 'N/A'),
            'change_pct':  q.get('10. change percent', 'N/A'),
            'volume':      q.get('06. volume', 'N/A'),
            'prev_close':  q.get('08. previous close', 'N/A'),
        }
    return cached(f'quote_{ticker}', fn)

def get_overview(ticker):
    def fn():
        d = av({'function':'OVERVIEW','symbol':ticker})
        return {
            'name':          d.get('Name', ticker),
            'sector':        d.get('Sector', 'N/A'),
            'industry':      d.get('Industry', 'N/A'),
            'pe_ratio':      d.get('PERatio', 'N/A'),
            'pb_ratio':      d.get('PriceToBookRatio', 'N/A'),
            'ps_ratio':      d.get('PriceToSalesRatioTTM', 'N/A'),
            'profit_margin': d.get('ProfitMargin', 'N/A'),
            'op_margin':     d.get('OperatingMarginTTM', 'N/A'),
            'revenue_growth':d.get('QuarterlyRevenueGrowthYOY', 'N/A'),
            'eps_growth':    d.get('QuarterlyEarningsGrowthYOY', 'N/A'),
            'eps_ttm':       d.get('EPS', 'N/A'),
            'dividend_yield':d.get('DividendYield', 'N/A'),
            'beta':          d.get('Beta', 'N/A'),
            'mkt_cap':       d.get('MarketCapitalization', 'N/A'),
            '52w_high':      d.get('52WeekHigh', 'N/A'),
            '52w_low':       d.get('52WeekLow', 'N/A'),
            'analyst_target':d.get('AnalystTargetPrice', 'N/A'),
            'description':   d.get('Description', '')[:400],
        }
    return cached(f'overview_{ticker}', fn)

def get_earnings(ticker):
    def fn():
        d = av({'function':'EARNINGS','symbol':ticker})
        quarters = d.get('quarterlyEarnings', [])[:4]
        return [{
            'date':           q.get('fiscalDateEnding'),
            'reported_eps':   q.get('reportedEPS'),
            'estimated_eps':  q.get('estimatedEPS'),
            'surprise_pct':   q.get('surprisePercentage'),
        } for q in quarters]
    return cached(f'earnings_{ticker}', fn)

# ── FRED helpers ──────────────────────────────────────────────────────────────
def fred(series_id):
    def fn():
        try:
            url = 'https://api.stlouisfed.org/fred/series/observations'
            params = {
                'series_id':  series_id,
                'api_key':    FRED_KEY,
                'file_type':  'json',
                'limit':      1,
                'sort_order': 'desc',
            }
            r = requests.get(url, params=params, timeout=8)
            obs = r.json().get('observations', [])
            return obs[0].get('value', 'N/A') if obs else 'N/A'
        except:
            return 'N/A'
    return cached(f'fred_{series_id}', fn)

def get_macro():
    return {
        'us10y':    fred('DGS10'),      # US 10Y Treasury
        'cpi_yoy':  fred('CPIAUCSL'),   # CPI
        'pce':      fred('PCEPI'),      # PCE
        'oil':      fred('DCOILWTICO'), # WTI Oil
        'gold':     fred('GOLDAMGBD228NLBM'),
        'usdyen':   fred('DEXJPUS'),    # USD/JPY
        'eurusd':   fred('DEXUSEU'),
    }

# ── Claude analysis ───────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are Keitaro's personal investment decision engine — an elite analyst who thinks like a top Wall Street professional.

You apply a strict 3-layer decision framework:

LAYER 1 - MACRO ENVIRONMENT (Market Gravity)
Check: US 10Y yield, CPI/PCE inflation, oil price, currency rates.
Key question: Is macro a headwind or tailwind? Rising rates = multiple compression. High oil = cost-push inflation risk.

LAYER 2 - FUNDAMENTAL ENGINE (Company's Earning Power)  
Check: EPS growth, revenue growth, operating margin, earnings surprises vs consensus, forward guidance.
Key question: Does this company have pricing power? Is it beating expectations consistently? Is guidance strong?

LAYER 3 - SUPPLY/DEMAND & MARKET PSYCHOLOGY
Check: Valuation (PE/PB/PS vs sector average), beta, 52-week position, analyst targets.
Key question: Is the price a fair reflection of value? Is there a "gap" between price and fundamental value?

THE ULTIMATE BUY SIGNAL: When macro fear drives the price down, but the company's fundamentals are at all-time highs — that's the "inter-layer gap" (歪み/Distortion) you look for.

VERDICTS:
- STRONG BUY: Clear inter-layer distortion — macro/sentiment driven selloff but fundamentals pristine
- BUY: Solid fundamentals, reasonable valuation, macro manageable  
- WATCH: Good company but timing or valuation not ideal — wait for better entry
- PASS: Fundamental weakness, expensive, or macro headwinds too strong
- STRONG PASS: Multiple red flags across layers

Always output in this EXACT JSON format:
{
  "verdict": "BUY|STRONG BUY|WATCH|PASS|STRONG PASS",
  "verdict_ja": "買い|強い買い|様子見|見送り|強い見送り",
  "confidence": 75,
  "summary_en": "2-3 sentence overall verdict explanation",
  "summary_ja": "2-3文の総合判断（日本語）",
  "layer1": {
    "score": 7,
    "signal": "TAILWIND|NEUTRAL|HEADWIND",
    "key_points_en": ["point1", "point2", "point3"],
    "key_points_ja": ["ポイント1", "ポイント2", "ポイント3"]
  },
  "layer2": {
    "score": 8,
    "signal": "STRONG|NEUTRAL|WEAK",
    "key_points_en": ["point1", "point2", "point3"],
    "key_points_ja": ["ポイント1", "ポイント2", "ポイント3"]
  },
  "layer3": {
    "score": 6,
    "signal": "UNDERVALUED|FAIR|OVERVALUED",
    "key_points_en": ["point1", "point2", "point3"],
    "key_points_ja": ["ポイント1", "ポイント2", "ポイント3"]
  },
  "distortion": {
    "found": true,
    "description_en": "Describe the inter-layer gap if found, or null",
    "description_ja": "歪みの説明（日本語）またはnull"
  },
  "risks_en": ["risk1", "risk2"],
  "risks_ja": ["リスク1", "リスク2"],
  "catalysts_en": ["catalyst1", "catalyst2"],
  "catalysts_ja": ["カタリスト1", "カタリスト2"]
}"""

def run_analysis(ticker, overview, quote, earnings, macro):
    prompt = f"""Analyze {ticker} using the 3-layer framework.

COMPANY DATA:
Name: {overview.get('name')} | Sector: {overview.get('sector')} | Industry: {overview.get('industry')}
Market Cap: {overview.get('mkt_cap')} | Beta: {overview.get('beta')}
Current Price: ${quote.get('price')} | Change: {quote.get('change_pct')}
52W High: {overview.get('52w_high')} | 52W Low: {overview.get('52w_low')}
Analyst Target: {overview.get('analyst_target')}

LAYER 2 - FUNDAMENTALS:
PE Ratio: {overview.get('pe_ratio')} | PB: {overview.get('pb_ratio')} | PS: {overview.get('ps_ratio')}
EPS TTM: {overview.get('eps_ttm')} | EPS Growth (YOY): {overview.get('eps_growth')}
Revenue Growth (YOY): {overview.get('revenue_growth')}
Operating Margin: {overview.get('op_margin')} | Profit Margin: {overview.get('profit_margin')}
Dividend Yield: {overview.get('dividend_yield')}

RECENT EARNINGS SURPRISES (last 4 quarters):
{json.dumps(earnings, indent=2)}

LAYER 1 - MACRO ENVIRONMENT:
US 10Y Yield: {macro.get('us10y')}%
CPI: {macro.get('cpi_yoy')} | PCE: {macro.get('pce')}
WTI Oil: ${macro.get('oil')}/bbl
Gold: ${macro.get('gold')}
USD/JPY: {macro.get('usdyen')} | EUR/USD: {macro.get('eurusd')}

Company Description: {overview.get('description')}

Apply the 3-layer framework. Output ONLY valid compact JSON (no markdown, no line breaks in strings). Keep each key_points list to max 2 items. Keep descriptions under 100 chars each."""

    try:
        msg = client.messages.create(
            model='claude-sonnet-4-6',
            max_tokens=3000,
            system=SYSTEM_PROMPT,
            messages=[{'role': 'user', 'content': prompt}]
        )
        text = msg.content[0].text.strip()
        # Strip markdown code blocks
        if '```' in text:
            parts = text.split('```')
            for part in parts:
                part = part.strip()
                if part.startswith('json'):
                    part = part[4:].strip()
                if part.startswith('{'):
                    text = part
                    break
        # Find JSON object boundaries
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1:
            text = text[start:end+1]
        try:
            return json.loads(text)
        except Exception:
            text = text.replace('\n', ' ').replace('\r', '')
            return json.loads(text)
    except Exception as e:
        return {'error': str(e)}

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/analyze', methods=['POST'])
def analyze():
    data    = request.get_json()
    ticker  = data.get('ticker', '').upper().strip()
    if not ticker:
        return jsonify({'error': 'Ticker required'}), 400

    overview = get_overview(ticker)
    quote    = get_quote(ticker)
    earnings = get_earnings(ticker)
    macro    = get_macro()

    if overview.get('name') == ticker and overview.get('sector') == 'N/A':
        return jsonify({'error': f'Ticker "{ticker}" not found. Please check the symbol.'}), 404

    result = run_analysis(ticker, overview, quote, earnings, macro)
    if 'error' in result:
        return jsonify(result), 500

    return jsonify({
        'ticker':   ticker,
        'name':     overview.get('name'),
        'price':    quote.get('price'),
        'change':   quote.get('change_pct'),
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
    return jsonify({'status': 'ok', 'keys': {
        'alpha': bool(ALPHA_KEY),
        'fred':  bool(FRED_KEY),
        'claude':bool(CLAUDE_KEY),
    }})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
