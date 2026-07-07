"""
═══════════════════════════════════════════════════════
  BUY THE DIP SCANNER — NASDAQ-100
  Modo completo: datos reales + email HTML + Telegram
  Filtro SMA 200: solo señales donde precio > SMA(200)
  Filtro TP mínimo: solo señales con objetivo Y >= TP_MIN  (nuevo)
═══════════════════════════════════════════════════════
  Uso:
    python btd_scanner.py              → escáner diario
    python btd_scanner.py --calibrar   → fuerza recalibración XYZ
    python btd_scanner.py --test       → modo test (no envía)

  Filtro TP:
    - Variable de entorno BTD_TP_MIN (ej. 7) en Render, o "tp_min" en config.json.
    - Por defecto 7: descarta señales cuyo objetivo de beneficio sea < 7%.
    - Poner 0 lo desactiva (comportamiento antiguo).
"""

import yfinance as yf
import pandas as pd
import numpy as np
from itertools import product
from datetime import datetime, timedelta, timezone
import json, os, sys, smtplib, requests, argparse
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

CFG_FILE    = Path("config.json")
PARAMS_FILE = Path("parametros_xyz.json")
HIST_FILE   = Path("historico_señales.csv")
LOG_FILE    = Path("btd_log.txt")

TP_MIN_DEFAULT = 7.0   # objetivo de beneficio mínimo (%) para aceptar una señal

# Detección de "señal nueva" SIN memoria (evita re-entrar en caídas libres):
# una entrada es nueva si el ticker NO era señal en las N sesiones previas.
SESIONES_LIMPIAS = 1   # 1 = regla validada (re-entrar solo si faltó al menos 1 sesión). Sube a 2-3 para exigir más.

TICKERS = [
    'NVDA','AAPL','MSFT','AMZN','GOOGL','GOOG','META','AVGO','TSLA','COST',
    'NFLX','ASML','AMD','MU','CSCO','LRCX','TMUS','AMAT','ISRG','LIN',
    'PEP','QCOM','AMGN','INTU','BKNG','KLAC','TXN','MRVL','ADBE','ADP',
    'PANW','GILD','SNPS','CDNS','REGN','MELI','CRWD','CTAS','ANET','ORLY',
    'MNST','FTNT','PCAR','CEG','TEAM','ROP','PYPL','PAYX','WDAY','ROST',
    'FAST','VRSK','CPRT','IDXX','ODFL','DXCM','ON','ABNB','ARM','PLTR',
    'TTD','ZS','DASH','CHTR','MCHP','BIIB','NXPI','SMCI','ILMN','WBD',
    'GEHC','GFS','FANG','SIRI','DLTR','DDOG','RIVN','LCID','MTCH','WBA',
    'ALGN','TTWO','ENPH','MDLZ','CSX','ADSK','CMCSA','KDP','AZN','VRTX',
    'SBUX','EBAY','EA','XEL','EXC','CTSH','MSTR','APP','COIN','HOOD'
]

RECALIBRAR_CADA_N_SESIONES = 10

# ── Robustez de la calibración (anti-sobreajuste / anti-PF-explotado) ──
MIN_SIGNALS    = 6     # mínimo de operaciones en el backtest para fiarnos de una combinación
MIN_PERDEDORAS = 2     # mínimo de operaciones PERDEDORAS (sin pérdidas reales el PF no es fiable)
PF_DISPLAY_CAP = 99.0  # tope de visualización del profit factor (por seguridad; ya no debería alcanzarse)

def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")

def cargar_config():
    # Modo nube (Render): lee credenciales desde variables de entorno
    if os.environ.get("BTD_EMAIL_FROM"):
        log("Modo nube: leyendo credenciales desde variables de entorno")
        return {
            "smtp": {
                "from":     os.environ.get("BTD_EMAIL_FROM", ""),
                "to":       os.environ.get("BTD_EMAIL_TO", ""),
                "user":     os.environ.get("BTD_EMAIL_USER", ""),
                "password": os.environ.get("BTD_EMAIL_PASSWORD", "")
            },
            "telegram": {
                "token":   os.environ.get("BTD_TELEGRAM_TOKEN", ""),
                "chat_id": os.environ.get("BTD_TELEGRAM_CHAT_ID", "")
            },
            "filtro_sma200": True,
            "tp_min": float(os.environ.get("BTD_TP_MIN", str(TP_MIN_DEFAULT)))
        }
    # Modo local: lee config.json
    if not CFG_FILE.exists():
        log("ERROR: No existe config.json. Ejecuta: python btd_scanner.py --setup")
        sys.exit(1)
    cfg = json.loads(CFG_FILE.read_text())
    cfg.setdefault("tp_min", TP_MIN_DEFAULT)
    return cfg

def sesiones_desde_calibracion():
    if not PARAMS_FILE.exists():
        return 999
    data = json.loads(PARAMS_FILE.read_text())
    fecha_str = data.get("fecha_calibracion", "2000-01-01")
    fecha = datetime.strptime(fecha_str, "%Y-%m-%d")
    dias = (datetime.today() - fecha).days
    return max(0, int(dias * 5/7))

def descargar_precios(ticker, dias=250):
    end = datetime.today()
    start = end - timedelta(days=dias)
    try:
        df = yf.download(ticker, start=start, end=end,
                         progress=False, auto_adjust=True)
        if df.empty or len(df) < 30:
            return None
        close = df['Close']
        # Fix compatibilidad yfinance MultiIndex
        if hasattr(close, 'squeeze'):
            close = close.squeeze()
        return close.dropna()
    except Exception as e:
        log(f"  ✗ {ticker}: {e}")
        return None

def backtest_xyz(prices, X, Y, Z, window=10, max_hold=20):
    p = np.array(prices, dtype=float)  # Fix: forzar float nativo
    wins, signals, returns = 0, 0, []
    for i in range(window, len(p) - max_hold):
        high_w = p[i-window:i].max()
        if high_w == 0:
            continue
        drop = (high_w - p[i]) / high_w * 100
        if drop >= X:
            entry = p[i]
            closed = False
            for j in range(1, max_hold + 1):
                gain = (p[i+j] - entry) / entry * 100
                if gain >= Y:
                    wins += 1; returns.append(float(gain)); signals += 1; closed = True; break
                elif gain <= -Z:
                    returns.append(float(gain)); signals += 1; closed = True; break
            if not closed:
                returns.append(float((p[i+max_hold] - entry) / entry * 100))
                signals += 1
    if signals < MIN_SIGNALS:
        return None
    n_perdedoras = sum(1 for r in returns if r < 0)
    wr        = float(wins) / float(signals)
    ganancias = float(sum(r for r in returns if r > 0))
    perdidas  = float(abs(sum(r for r in returns if r < 0))) or 1e-9
    total_ret = float(sum(returns))
    # PF acotado: con MIN_PERDEDORAS>=2 el denominador es real y no explota;
    # el tope es solo un cinturón de seguridad para la visualización.
    pf = round(min(ganancias / perdidas, PF_DISPLAY_CAP), 4)
    return {
        'signals': signals,
        'n_perdedoras': n_perdedoras,
        'win_rate': round(wr, 4),
        'profit_factor': pf,
        'expectancy': round(total_ret / signals, 3),   # retorno medio por operación (%)
        'total_return': round(total_ret, 2)
    }

def optimizar_xyz(prices):
    best, best_params, best_key = None, None, None
    for X, Y, Z in product(range(7,16), [3,4,5,6,7,8,9,10,12,15], range(3,11)):
        r = backtest_xyz(prices, X, Y, Z)
        if not r:
            continue
        if r['win_rate'] < 0.45:
            continue
        # Descarta combinaciones sin pérdidas reales: su PF no es fiable (sobreajuste).
        if r['n_perdedoras'] < MIN_PERDEDORAS:
            continue
        # Ranking honesto: mayor PF; a igualdad, más operaciones (más fiable) y más retorno.
        key = (r['profit_factor'], r['signals'], r['total_return'])
        if best_key is None or key > best_key:
            best_key, best, best_params = key, r, (X, Y, Z)
    return best_params, best

def calibrar_todos():
    log("═══ Iniciando calibración XYZ para todos los tickers ═══")
    resultados = {}
    for i, ticker in enumerate(TICKERS):
        log(f"  [{i+1}/{len(TICKERS)}] Calibrando {ticker}...")
        prices = descargar_precios(ticker, dias=200)
        if prices is None or len(prices) < 60:
            log(f"  ⚠ {ticker}: datos insuficientes")
            continue
        params, stats = optimizar_xyz(prices)
        if params:
            resultados[ticker] = {
                'X': params[0], 'Y': params[1], 'Z': params[2],
                'win_rate': stats['win_rate'],
                'profit_factor': stats['profit_factor'],
                'signals': stats['signals'],
                'n_perdedoras': stats['n_perdedoras'],
                'expectancy': stats['expectancy'],
                'total_return': stats['total_return']
            }
            log(f"  ✓ {ticker}: X={params[0]}% Y={params[1]}% Z={params[2]}% | "
                f"PF={stats['profit_factor']:.2f} WR={stats['win_rate']*100:.1f}%")
        else:
            log(f"  ⚠ {ticker}: sin combinación válida")
    datos = {'fecha_calibracion': datetime.today().strftime("%Y-%m-%d"), 'tickers': resultados}
    PARAMS_FILE.write_text(json.dumps(datos, indent=2))
    log(f"═══ Calibración completada: {len(resultados)}/{len(TICKERS)} tickers válidos ═══")
    return resultados

def precio_sobre_sma200(prices):
    """Filtro tendencia bajista: solo operar si precio > SMA(200).
    En mercados bajistas sostenidos, buy the dip genera múltiples stop-losses."""
    arr = np.array(prices, dtype=float)
    if len(arr) < 200:
        return True
    sma200 = float(arr[-200:].mean())
    precio_actual = float(arr[-1])
    return precio_actual > sma200

def _caida_en(arr, offset, window=10):
    """Caída % desde el máximo de 'window' sesiones, evaluada 'offset' sesiones atrás.
    offset=0 = hoy (arr[-1]), offset=1 = ayer (arr[-2]), etc. Misma definición que el escáner."""
    fin = len(arr) - offset          # el punto a evaluar es arr[fin-1]
    if fin < window + 1:
        return None
    ventana = arr[fin - window:fin]
    high = float(ventana.max())
    actual = float(arr[fin - 1])
    if high == 0:
        return None
    return (high - actual) / high * 100

def es_nueva_hoy(arr, X, sesiones_limpias=1, window=10):
    """True si HOY es señal (caída >= X) y NO lo era en las 'sesiones_limpias'
    sesiones previas. Detecta el inicio de un episodio SIN necesidad de memoria:
    si el ticker ya venía señalando (caída libre), NO se considera nueva → no se re-entra."""
    hoy = _caida_en(arr, 0, window)
    if hoy is None or hoy < X:
        return False
    for o in range(1, sesiones_limpias + 1):
        c = _caida_en(arr, o, window)
        if c is not None and c >= X:
            return False   # ya señalaba en una sesión reciente → continuación, no entrada nueva
    return True

def escanear(params_xyz, filtro_sma200=True, tp_min=0.0):
    log("═══ Escaneando señales activas hoy ═══")
    if tp_min > 0:
        log(f"Filtro TP: solo señales con objetivo Y >= {tp_min}%")
    oportunidades = []
    tickers_validos = params_xyz.get('tickers', {})
    n_filtrados_tp = 0
    for ticker, r in tickers_validos.items():
        # ── Filtro TP mínimo: descarta objetivos pequeños (bajo R/R) ──
        # El objetivo de beneficio es exactamente r['Y'] (Take_Profit = precio*(1+Y/100)).
        if tp_min > 0 and r.get('Y', 0) < tp_min:
            n_filtrados_tp += 1
            continue
        prices = descargar_precios(ticker, dias=260)
        if prices is None or len(prices) < 12:
            continue
        # Guard de frescura: la ultima barra debe ser de HOY.
        # Los cron corren en horario de mercado EE.UU. (19-21 UTC), asi que la fecha
        # UTC coincide con la de la sesion. Suprime senales en festivos de EE.UU.
        # (el cron dispara aunque el mercado este cerrado y el dato sea de ayer).
        _hoy = datetime.now(timezone.utc).date()
        if prices.index[-1].date() != _hoy:
            log(f"  x {ticker}: ultima barra {prices.index[-1].date()} != {_hoy} -> salto (dato no fresco)")
            continue
        if filtro_sma200 and not precio_sobre_sma200(prices):
            log(f"  ↓ {ticker}: precio < SMA200 → filtrado (tendencia bajista)")
            continue
        arr = np.array(prices, dtype=float)
        high_10  = float(arr[-10:].max())
        current  = float(arr[-1])
        if high_10 == 0:
            continue
        caida = (high_10 - current) / high_10 * 100
        if caida >= r['X']:
            sma200_val = float(arr[-200:].mean()) if len(arr) >= 200 else None
            nueva = es_nueva_hoy(arr, r['X'], sesiones_limpias=SESIONES_LIMPIAS)
            oportunidades.append({
                'Ticker':        ticker,
                'Caída_10d%':    round(caida, 2),
                'Entrada':       round(current, 2),
                'Take_Profit':   round(current * (1 + r['Y']/100), 2),
                'Stop_Loss':     round(current * (1 - r['Z']/100), 2),
                'X%':            r['X'], 'Y%': r['Y'], 'Z%': r['Z'],
                'R/R':           round(r['Y'] / r['Z'], 2),
                'Profit_Factor': round(r['profit_factor'], 4),
                'Win_Rate%':     round(r['win_rate'] * 100, 1),
                'SMA200':        round(sma200_val, 2) if sma200_val else None,
                'Sobre_SMA200':  '✅' if (sma200_val and current > sma200_val) else '⚠',
                'Fecha':         datetime.today().strftime("%Y-%m-%d"),
                'Nueva_hoy':     bool(nueva)
            })
            log(f"  🔔 {ticker}: caída {caida:.1f}% ≥ X={r['X']}% → SEÑAL ACTIVA")
    if tp_min > 0 and n_filtrados_tp:
        log(f"  (Filtro TP≥{tp_min}%: {n_filtrados_tp} tickers descartados por objetivo pequeño)")
    oportunidades.sort(key=lambda x: x['Profit_Factor'], reverse=True)
    log(f"═══ {len(oportunidades)} señales activas ═══")
    return oportunidades

def señales_nuevas(hoy, ayer_df):
    if ayer_df is None or ayer_df.empty:
        return hoy
    tickers_ayer = set(ayer_df['Ticker'].tolist())
    return [s for s in hoy if s['Ticker'] not in tickers_ayer]

def guardar_historico(señales):
    df_hoy = pd.DataFrame(señales)
    if HIST_FILE.exists():
        df_hist = pd.read_csv(HIST_FILE)
        df_total = pd.concat([df_hist, df_hoy], ignore_index=True)
    else:
        df_total = df_hoy
    df_total.to_csv(HIST_FILE, index=False)

def cargar_señales_ayer():
    if not HIST_FILE.exists():
        return None
    df = pd.read_csv(HIST_FILE)
    ayer = (datetime.today() - timedelta(days=1)).strftime("%Y-%m-%d")
    df_ayer = df[df['Fecha'] == ayer]
    return df_ayer if not df_ayer.empty else None

def construir_email_html(señales, nuevas, fecha, tp_min=0.0):
    filtro_txt = f" · Filtro TP≥{tp_min:g}%" if tp_min > 0 else ""
    filas = ""
    for s in señales:
        es_nueva = s['Ticker'] in [n['Ticker'] for n in nuevas]
        bg = "#1a3a2a" if es_nueva else ""
        premium = s['R/R'] >= 1.5 and s['Win_Rate%'] >= 55
        badge  = '<span style="background:#4f98a3;color:white;padding:2px 8px;border-radius:8px;font-size:11px;margin-left:6px">PREMIUM</span>' if premium else ""
        nuevo_b = '<span style="background:#fdab43;color:#1a1a1a;padding:2px 8px;border-radius:8px;font-size:11px;margin-left:6px">NUEVO</span>' if es_nueva else ""
        filas += f"""
        <tr style="background:{bg}">
          <td style="padding:10px 8px;font-weight:700">{s['Ticker']}{badge}{nuevo_b}</td>
          <td style="padding:10px 8px;color:#fd7070">{s['Caída_10d%']}%</td>
          <td style="padding:10px 8px">${s['Entrada']}</td>
          <td style="padding:10px 8px;color:#6daa45">${s['Take_Profit']}</td>
          <td style="padding:10px 8px;color:#d163a7">${s['Stop_Loss']}</td>
          <td style="padding:10px 8px">{s['R/R']}</td>
          <td style="padding:10px 8px">{s['Profit_Factor']}</td>
          <td style="padding:10px 8px">{s['Win_Rate%']}%</td>
          <td style="padding:10px 8px">{s['Sobre_SMA200']}</td>
        </tr>"""
    tabla = f"""
    <table style="width:100%;border-collapse:collapse;font-family:monospace;font-size:13px">
      <thead><tr style="background:#1c1b19;color:#797876">
        <th style="padding:10px 8px;text-align:left">Ticker</th>
        <th style="padding:10px 8px;text-align:left">Caída</th>
        <th style="padding:10px 8px;text-align:left">Entrada</th>
        <th style="padding:10px 8px;text-align:left">Take Profit</th>
        <th style="padding:10px 8px;text-align:left">Stop Loss</th>
        <th style="padding:10px 8px;text-align:left">R/R</th>
        <th style="padding:10px 8px;text-align:left">Profit Factor</th>
        <th style="padding:10px 8px;text-align:left">Win Rate</th>
        <th style="padding:10px 8px;text-align:left">SMA200</th>
      </tr></thead>
      <tbody>{filas}</tbody>
    </table>""" if señales else "<p style='color:#797876'>No hay señales activas hoy.</p>"
    n_nuevas = len(nuevas)
    resumen = f"<p style='color:#fdab43;font-weight:bold'>🆕 {n_nuevas} señal(es) nueva(s) detectada(s) hoy.</p>" if n_nuevas else "<p style='color:#797876'>No hay señales nuevas respecto a ayer.</p>"
    return f"""<!DOCTYPE html><html><head><meta charset="utf-8"></head>
<body style="background:#171614;color:#cdccca;font-family:'Helvetica Neue',Arial,sans-serif;margin:0;padding:0">
  <div style="max-width:900px;margin:0 auto;padding:32px 16px">
    <h1 style="font-size:28px;margin-bottom:4px">📊 Buy the Dip Scanner</h1>
    <p style="color:#797876;margin-bottom:24px">NASDAQ-100 · {fecha} · Filtro SMA200 activo{filtro_txt}</p>
    {resumen}
    <h2 style="font-size:18px;margin:24px 0 12px">🚨 Señales activas hoy ({len(señales)})</h2>
    <div style="background:#1c1b19;border-radius:16px;padding:16px;overflow:auto">{tabla}</div>
    <div style="margin-top:32px;background:#1c1b19;border-radius:16px;padding:20px">
      <h3 style="font-size:15px;margin-bottom:10px;color:#797876">⚠️ Advertencias</h3>
      <ul style="color:#797876;font-size:13px;line-height:1.8;padding-left:18px">
        <li>Estrategia mecánica: no considera fundamentales, noticias ni macro.</li>
        <li>Filtro SMA200 <strong style="color:#4f98a3">ACTIVO</strong>: solo señales donde precio &gt; SMA(200). En mercados bajistas sostenidos, buy the dip sin este filtro puede generar múltiples stop-losses consecutivos.</li>
        <li>Backtesting sobre 6 meses puede mostrar overfitting. Verificar en múltiples períodos antes de operar con capital real.</li>
        <li>Tamaño de posición sugerido: no superar el 2–5% del capital total por señal simultánea.</li>
        <li><strong>Análisis informativo. No constituye asesoramiento financiero.</strong></li>
      </ul>
    </div>
    <p style="color:#5a5957;font-size:11px;margin-top:24px">Generado automáticamente por btd_scanner.py</p>
  </div>
</body></html>"""

def enviar_email(html, señales, nuevas, cfg, test=False):
    if test:
        log("  [TEST] Email no enviado. Guardado en email_preview.html")
        Path("email_preview.html").write_text(html, encoding="utf-8")
        return
    smtp_cfg = cfg.get("smtp", {})
    msg = MIMEMultipart("alternative")
    n = len(señales); n_new = len(nuevas)
    asunto = f"📊 BTD Scanner · {n} señal(es)"
    if n_new: asunto += f" · 🆕 {n_new} nueva(s)"
    msg["Subject"] = asunto
    msg["From"]    = smtp_cfg["from"]
    msg["To"]      = smtp_cfg["to"]
    msg.attach(MIMEText(html, "html"))
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(smtp_cfg["user"], smtp_cfg["password"])
            server.sendmail(smtp_cfg["from"], smtp_cfg["to"], msg.as_string())
        log(f"  ✅ Email enviado a {smtp_cfg['to']}")
    except Exception as e:
        log(f"  ✗ Error enviando email: {e}")

def enviar_telegram(señales, nuevas, cfg, test=False, tp_min=0.0, solo_validacion=False):
    tg = cfg.get("telegram", {})
    if not tg.get("token") or not tg.get("chat_id"):
        log("  ⚠ Telegram no configurado. Saltando.")
        return
    n = len(señales); n_new = len(nuevas)
    fecha = datetime.today().strftime("%d/%m/%Y")
    cab = "NASDAQ-100 · Filtro SMA200 activo"
    if solo_validacion:
        cab = "VALIDACION (NO OPERAR) - senales a cierre real | " + cab
    if tp_min > 0:
        cab += f" · TP≥{tp_min:g}%"
    lineas = [f"*📊 Buy the Dip Scanner — {fecha}*", cab + "\n"]
    if n == 0:
        lineas.append("No hay señales activas hoy.")
    else:
        if n_new:
            lineas.append(f"🆕 *{n_new} señal(es) nueva(s):*")
            for s in nuevas:
                lineas.append(f"  → *{s['Ticker']}*: entrada ${s['Entrada']} | TP ${s['Take_Profit']} | SL ${s['Stop_Loss']} | PF {s['Profit_Factor']}")
            lineas.append("")
        lineas.append(f"📋 *{n} señal(es) totales:*")
        for s in señales[:8]:
            lineas.append(f"  `{s['Ticker']:5}` caída {s['Caída_10d%']:5.1f}% | R/R {s['R/R']} | PF {s['Profit_Factor']}")
        if n > 8:
            lineas.append(f"  ...y {n-8} más. Ver email para tabla completa.")
    lineas.append("\n⚠ _Solo informativo. No asesoramiento financiero._")
    texto = "\n".join(lineas)
    if test:
        log(f"  [TEST] Telegram no enviado. Mensaje:\n{texto}")
        return
    url = f"https://api.telegram.org/bot{tg['token']}/sendMessage"
    try:
        r = requests.post(url, json={"chat_id": tg["chat_id"], "text": texto, "parse_mode": "Markdown"}, timeout=10)
        log("  ✅ Telegram enviado" if r.status_code == 200 else f"  ✗ Error Telegram: {r.text}")
    except Exception as e:
        log(f"  ✗ Error Telegram: {e}")

def setup_interactivo():
    print("\n═══════════════════════════════════════════")
    print("  BUY THE DIP SCANNER — Configuración")
    print("═══════════════════════════════════════════\n")
    cfg = {}
    print("── EMAIL (Gmail recomendado) ──")
    print("  Necesitas una contraseña de aplicación Gmail:")
    print("  myaccount.google.com → Seguridad → Verificación en 2 pasos → Contraseñas de app\n")
    cfg["smtp"] = {
        "from":     input("  Tu email (remitente): ").strip(),
        "to":       input("  Email destino (puede ser el mismo): ").strip(),
        "user":     input("  Usuario Gmail: ").strip(),
        "password": input("  Contraseña de app Gmail: ").strip()
    }
    print("\n── TELEGRAM (opcional, pulsa Enter para omitir) ──")
    print("  1. Habla con @BotFather en Telegram → /newbot")
    print("  2. Habla con @userinfobot para tu chat_id\n")
    token   = input("  Token del bot (o Enter para omitir): ").strip()
    chat_id = input("  Tu chat_id (o Enter para omitir): ").strip()
    cfg["telegram"] = {"token": token, "chat_id": chat_id}
    sma200 = input("\n  ¿Activar filtro SMA200? (recomendado) [S/n]: ").strip().lower()
    cfg["filtro_sma200"] = sma200 != "n"
    tp = input(f"  Objetivo TP mínimo en %% (Enter = {TP_MIN_DEFAULT:g}, 0 = sin filtro): ").strip()
    try:
        cfg["tp_min"] = float(tp) if tp else TP_MIN_DEFAULT
    except ValueError:
        cfg["tp_min"] = TP_MIN_DEFAULT
    CFG_FILE.write_text(json.dumps(cfg, indent=2))
    print(f"\n✅ Config guardada en {CFG_FILE}")
    print("  Siguiente paso: python btd_scanner.py --calibrar\n")

def main():
    parser = argparse.ArgumentParser(description="Buy the Dip Scanner — NASDAQ-100")
    parser.add_argument("--calibrar", action="store_true")
    parser.add_argument("--test",     action="store_true")
    parser.add_argument("--setup",    action="store_true")
    parser.add_argument("--sin-sma",  action="store_true")
    parser.add_argument("--no-email", action="store_true",
                        help="No envia email; solo Telegram + CSV (run post-cierre de validacion)")
    parser.add_argument("--tp-min",   type=float, default=None,
                        help="Objetivo TP mínimo en %% (sobrescribe config/env; 0 = sin filtro)")
    args = parser.parse_args()
    if args.setup:
        setup_interactivo(); return
    cfg = cargar_config()
    usar_sma200 = cfg.get("filtro_sma200", True) and not args.sin_sma
    tp_min = args.tp_min if args.tp_min is not None else cfg.get("tp_min", TP_MIN_DEFAULT)
    log(f"Filtro SMA200: {'ACTIVO' if usar_sma200 else 'INACTIVO'}")
    log(f"Filtro TP mínimo: {tp_min:g}%" if tp_min > 0 else "Filtro TP mínimo: INACTIVO")
    sesiones = sesiones_desde_calibracion()
    if args.calibrar or sesiones >= RECALIBRAR_CADA_N_SESIONES:
        log(f"Recalibrando ({sesiones} sesiones desde última calibración)...")
        calibrar_todos()
    else:
        log(f"Parámetros vigentes ({sesiones} sesiones). OK.")
    if not PARAMS_FILE.exists():
        log("ERROR: Ejecuta primero: python btd_scanner.py --calibrar")
        sys.exit(1)
    params_xyz = json.loads(PARAMS_FILE.read_text())
    fecha = datetime.today().strftime("%Y-%m-%d")
    señales_hoy  = escanear(params_xyz, filtro_sma200=usar_sma200, tp_min=tp_min)
    # "Nuevas" SIN memoria: una señal es nueva si NO lo era en las sesiones previas
    # (calculado desde los precios). Esto evita marcar como nuevas las re-entradas
    # en caída libre y no depende de que Render conserve el histórico.
    nuevas = [s for s in señales_hoy if s.get('Nueva_hoy')]
    if señales_hoy:
        guardar_historico(señales_hoy)
        pd.DataFrame(señales_hoy).to_csv(f"señales_{fecha}.csv", index=False)
        log(f"CSV guardado: señales_{fecha}.csv")
    if args.no_email:
        log("  [--no-email] Run de validacion: sin email, solo Telegram + CSV")
        enviar_telegram(señales_hoy, nuevas, cfg, test=args.test, tp_min=tp_min, solo_validacion=True)
    else:
        html = construir_email_html(señales_hoy, nuevas, fecha, tp_min=tp_min)
        enviar_email(html, señales_hoy, nuevas, cfg, test=args.test)
        enviar_telegram(señales_hoy, nuevas, cfg, test=args.test, tp_min=tp_min)
    log("═══ Ejecución completada ═══\n")

if __name__ == "__main__":
    main()
