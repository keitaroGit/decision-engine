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
        d = av({'function':'GLOBAL_QUOTE','symbol':ticker})
        q = d.get('Global Quote', {})
        return {
            'price':      q.get('05. price', 'N/A'),
            'change_pct': q.get('10. change percent', 'N/A'),
            'volume':     q.get('06. volume', 'N/A'),
            'prev_close': q.get('08. previous close', 'N/A'),
        }
    return cached('quote_'+ticker, fn)

def get_overview(ticker):
    def fn():
        d = av({'function':'OVERVIEW','symbol':ticker})
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
            'dividend_yield': d.get('DividendYield', 'N/A'),
            'beta':           d.get('Beta', 'N/A'),
            'mkt_cap':        d.get('MarketCapitalization', 'N/A'),
            '52w_high':       d.get('52WeekHigh', 'N/A'),
            '52w_low':        d.get('52WeekLow', 'N/A'),
            'analyst_target': d.get('AnalystTargetPrice', 'N/A'),
            'description':    d.get('Description', '')[:300],
        }
    return cached('overview_'+ticker, fn)

def get_earnings(ticker):
    def fn():
        d = av({'function':'EARNINGS','symbol':ticker})
        quarters = d.get('quarterlyEarnings', [])[:4]
        return [{
            'date':          q.get('fiscalDateEnding'),
            'reported_eps':  q.get('reportedEPS'),
            'estimated_eps': q.get('estimatedEPS'),
            'surprise_pct':  q.get('surprisePercentage'),
        } for q in quarters]
    return cached('earnings_'+ticker, fn)

def fred_val(series_id):
    def fn():
        try:
            r = requests.get('https://api.stlouisfed.org/fred/series/observations', params={
                'series_id': series_id, 'api_key': FRED_KEY,
                'file_type': 'json', 'limit': 1, 'sort_order': 'desc'
            }, timeout=8)
            obs = r.json().get('observations', [])
            return obs[0].get('value', 'N/A') if obs else 'N/A'
        except:
            return 'N/A'
    return cached('fred_'+series_id, fn)

def get_macro():
    return {
        'us10y':  fred_val('DGS10'),
        'cpi':    fred_val('CPIAUCSL'),
        'oil':    fred_val('DCOILWTICO'),
        'gold':   fred_val('GOLDAMGBD228NLBM'),
        'usdyen': fred_val('DEXJPUS'),
        'eurusd': fred_val('DEXUSEU'),
    }

SYSTEM_PROMPT = """You are Keitaro's personal investment decision engine — an elite analyst applying a strict 3-layer framework.

LAYER 1 - MACRO: Interest rates, inflation, oil, currency impact on the stock.
LAYER 2 - FUNDAMENTALS: EPS growth, revenue, margins, earnings surprises, guidance quality.
LAYER 3 - VALUATION/DEMAND: PE/PB/PS vs peers, price position, analyst targets.

THE ULTIMATE BUY SIGNAL: Macro fear drives price down but fundamentals are at all-time highs = inter-layer distortion.

Output ONLY this exact JSON structure, no markdown, no extra text:
{"verdict":"BUY","verdict_ja":"買い","confidence":75,"summary_en":"summary here","summary_ja":"要約","layer1":{"score":7,"signal":"TAILWIND","key_points_en":["point1","point2"],"key_points_ja":["ポイント1","ポイント2"]},"layer2":{"score":8,"signal":"STRONG","key_points_en":["point1","point2"],"key_points_ja":["ポイント1","ポイント2"]},"layer3":{"score":6,"signal":"FAIR","key_points_en":["point1","point2"],"key_points_ja":["ポイント1","ポイント2"]},"distortion":{"found":false,"description_en":null,"description_ja":null},"risks_en":["risk1","risk2"],"risks_ja":["リスク1","リスク2"],"catalysts_en":["cat1","cat2"],"catalysts_ja":["カタリスト1","カタリスト2"]}

Verdict options: STRONG BUY / BUY / WATCH / PASS / STRONG PASS
verdict_ja options: 強い買い / 買い / 様子見 / 見送り / 強い見送り
signal options - layer1: TAILWIND/NEUTRAL/HEADWIND, layer2: STRONG/NEUTRAL/WEAK, layer3: UNDERVALUED/FAIR/OVERVALUED"""

def run_analysis(ticker, overview, quote, earnings, macro, horizon='mid'):
    horizon_labels = {
        'short': 'SHORT TERM (1-3 months) — weight Layer3 highest',
        'mid':   'MID TERM (3-12 months) — weight Layer2 highest',
        'long':  'LONG TERM (1-3 years) — weight Layer1 + moat highest',
    }
    horizon_contexts = {
        'short': 'Focus on momentum, near-term catalysts, short interest, IV. Is NOW a good entry?',
        'mid':   'Focus on earnings trajectory, margin trends, sector rotation. Next earnings cycle upside?',
        'long':  'Focus on competitive moat, TAM, management, balance sheet strength. 3-year compounder?',
    }
    hl = horizon_labels.get(horizon, horizon_labels['mid'])
    hc = horizon_contexts.get(horizon, horizon_contexts['mid'])

    prompt = (
        "Analyze " + ticker + " for " + hl + ". " + hc + "\n\n"
        "COMPANY: " + overview.get('name','') + " | " + overview.get('sector','') + "\n"
        "Price: $" + str(quote.get('price','N/A')) + " | Change: " + str(quote.get('change_pct','N/A')) + "\n"
        "52W High: " + str(overview.get('52w_high','')) + " | Low: " + str(overview.get('52w_low','')) + " | Target: $" + str(overview.get('analyst_target','')) + "\n"
        "PE: " + str(overview.get('pe_ratio','')) + " | PB: " + str(overview.get('pb_ratio','')) + " | PS: " + str(overview.get('ps_ratio','')) + "\n"
        "EPS TTM: " + str(overview.get('eps_ttm','')) + " | EPS Growth: " + str(overview.get('eps_growth','')) + "\n"
        "Rev Growth: " + str(overview.get('revenue_growth','')) + " | Op Margin: " + str(overview.get('op_margin','')) + "\n"
        "Beta: " + str(overview.get('beta','')) + " | MktCap: " + str(overview.get('mkt_cap','')) + "\n"
        "Recent earnings surprises: " + json.dumps(earnings) + "\n"
        "MACRO: US10Y=" + str(macro.get('us10y','')) + "% | Oil=$" + str(macro.get('oil','')) + " | USD/JPY=" + str(macro.get('usdyen','')) + "\n"
        "Output ONLY valid JSON as specified."
    )

    try:
        msg = client.messages.create(
            model='claude-sonnet-4-6',
            max_tokens=1500,
            system=SYSTEM_PROMPT,
            messages=[{'role': 'user', 'content': prompt}]
        )
        text = msg.content[0].text.strip()
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1:
            text = text[start:end+1]
        return json.loads(text)
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
        return jsonify({'error': 'Ticker "' + ticker + '" not found.'}), 404

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
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
