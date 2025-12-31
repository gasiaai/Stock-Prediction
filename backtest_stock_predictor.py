"""
ตัวทดลองการวิเคราะห์หุ้นจากอดีตเพื่อคาดการณ์อนาคต
Backtesting stock predictor using historical data
"""

import yfinance as yf
import pandas as pd
import numpy as np
from stock_analysis_system import StockAnalysisSystem
import os
from datetime import datetime, timedelta

def load_symbols_from_csv(csv_path):
    """
    โหลดรายชื่อหุ้นจาก CSV ไฟล์
    """
    try:
        df = pd.read_csv(csv_path)
        symbols = df['Symbol'].dropna().tolist()
        print(f"✓ โหลดได้ {len(symbols)} หุ้นจาก {csv_path}")
        return symbols
    except Exception as e:
        print(f"❌ ไม่สามารถโหลด CSV: {e}")
        return []

def get_historical_data(symbols, start_date, end_date):
    """
    ดึงข้อมูลราคาปิดในช่วงเวลาที่กำหนด
    """
    print(f"กำลังดึงข้อมูลหุ้น {len(symbols)} ตัว จาก {start_date} ถึง {end_date}...")

    # ดึงข้อมูลเป็นชุดเล็กๆ เพื่อหลีกเลี่ยงข้อจำกัด API
    chunk_size = 50
    all_data = []

    for i in range(0, len(symbols), chunk_size):
        chunk_symbols = symbols[i:i+chunk_size]
        print(f"  ดึงข้อมูลกลุ่มที่ {i//chunk_size + 1}: {len(chunk_symbols)} หุ้น")

        try:
            data = yf.download(chunk_symbols, start=start_date, end=end_date)['Close']
            all_data.append(data)
        except Exception as e:
            print(f"  ❌ ข้อผิดพลาดในการดึงข้อมูลกลุ่มนี้: {e}")
            continue

    if not all_data:
        print("❌ ไม่มีข้อมูลใดๆ ที่ดึงได้")
        return pd.DataFrame()

    # รวมข้อมูลทั้งหมด
    combined_data = pd.concat(all_data, axis=1)

    # ลบคอลัมน์ที่ไม่มีข้อมูล
    combined_data = combined_data.dropna(axis=1, how='all')

    # ลบแถวที่ไม่มีข้อมูลมากเกินไป
    combined_data = combined_data.dropna(thresh=len(combined_data.columns) * 0.5)

    print(f"✓ ได้รับข้อมูล {len(combined_data)} วัน สำหรับ {len(combined_data.columns)} หุ้น")
    print(f"ช่วงเวลา: {combined_data.index[0].date()} ถึง {combined_data.index[-1].date()}")

    return combined_data

def predict_top_stocks(train_data, fundamentals, top_n=10):
    """
    วิเคราะห์ข้อมูลฝึกอบรมและเลือกหุ้นที่ดีที่สุด
    """
    print(f"\nกำลังวิเคราะห์ข้อมูลฝึกอบรมเพื่อเลือกหุ้นที่ดีที่สุด {top_n} อันดับ...")

    # สร้างระบบวิเคราะห์
    system = StockAnalysisSystem()

    # วิเคราะห์หุ้น
    results = system.analyze_stock(
        train_data,
        fundamentals.get('market_cap', pd.Series()),
        fundamentals.get('book_to_market', pd.Series())
    )

    # คำนวณผลตอบแทนและความเสี่ยง
    log_returns = results['preprocessed_data']['log_returns']
    annual_returns = log_returns.mean() * 252 * 100
    annual_vols = log_returns.std() * np.sqrt(252) * 100
    sharpe_ratios = annual_returns / annual_vols.replace(0, np.nan)

    # เลือกหุ้นที่ดีที่สุดตาม Sharpe Ratio
    top_stocks = sharpe_ratios.nlargest(top_n).index.tolist()

    print(f"✓ เลือกหุ้นที่ดีที่สุด: {', '.join(top_stocks)}")

    return top_stocks, annual_returns, sharpe_ratios

def evaluate_predictions(top_stocks, test_data):
    """
    ประเมินผลการคาดการณ์ด้วยข้อมูลจริงในอนาคต
    """
    print(f"\nกำลังประเมินผลการคาดการณ์ด้วยข้อมูลจริง...")

    if test_data.empty:
        print("❌ ไม่มีข้อมูลทดสอบ")
        return {}

    # คำนวณผลตอบแทนจริงในช่วงทดสอบ
    test_returns = test_data.pct_change().dropna()

    # คำนวณผลตอบแทนรวมในช่วงทดสอบ
    cumulative_returns = (1 + test_returns).cumprod() - 1
    final_returns = cumulative_returns.iloc[-1] * 100  # เป็นเปอร์เซ็นต์

    # ผลการคาดการณ์
    results = {}
    predicted_correct = 0

    for stock in top_stocks:
        if stock in final_returns.index:
            actual_return = final_returns[stock]
            results[stock] = actual_return
            if actual_return > 0:  # ถ้าขึ้นจริง
                predicted_correct += 1
        else:
            results[stock] = np.nan
            print(f"  ⚠️  ไม่มีข้อมูลทดสอบสำหรับ {stock}")

    accuracy = predicted_correct / len(top_stocks) * 100 if top_stocks else 0

    print(f"✓ ผลการคาดการณ์: {predicted_correct}/{len(top_stocks)} ({accuracy:.1f}%) ถูกต้อง")

    return results, accuracy

def evaluate_all_predictions(train_data, test_data, train_sharpe):
    """
    ประเมินผลการคาดการณ์สำหรับทุกหุ้น
    """
    print(f"\nกำลังประเมินผลการคาดการณ์สำหรับทุกหุ้น ({len(train_data.columns)} ตัว)...")

    if test_data.empty:
        print("❌ ไม่มีข้อมูลทดสอบ")
        return pd.DataFrame(), []

    # คำนวณผลตอบแทนจริงในช่วงทดสอบ
    test_returns = test_data.pct_change().dropna()
    cumulative_returns = (1 + test_returns).cumprod() - 1
    final_returns = cumulative_returns.iloc[-1] * 100

    # รวมข้อมูล Sharpe Ratio และผลตอบแทนจริง
    results = []
    for stock in train_data.columns:
        if stock in final_returns.index and stock in train_sharpe.index:
            sharpe = train_sharpe[stock]
            actual_return = final_returns[stock]
            actually_up = actual_return > 0
            results.append({
                'stock': stock,
                'sharpe': sharpe,
                'actual_return': actual_return,
                'actually_up': actually_up
            })

    results_df = pd.DataFrame(results).sort_values('sharpe', ascending=False)

    # แบ่งเป็นเดซิล (10 กลุ่ม)
    results_df['decile'] = pd.qcut(results_df['sharpe'], 10, labels=False, duplicates='drop') + 1

    # คำนวณความแม่นยำในแต่ละเดซิล
    decile_stats = []
    for decile in range(1, 11):
        decile_data = results_df[results_df['decile'] == decile]
        if len(decile_data) > 0:
            accuracy = decile_data['actually_up'].mean() * 100
            avg_return = decile_data['actual_return'].mean()
            count = len(decile_data)
            decile_stats.append({
                'decile': decile,
                'count': count,
                'accuracy': accuracy,
                'avg_return': avg_return
            })

    return results_df, decile_stats

def get_fundamentals(symbols):
    """
    ดึงข้อมูลพื้นฐานของหุ้น
    """
    fundamentals = {}

    print(f"กำลังดึงข้อมูลพื้นฐานสำหรับ {len(symbols)} หุ้น...")

    for i, symbol in enumerate(symbols):
        if (i + 1) % 20 == 0:
            print(f"  ดึงข้อมูลพื้นฐาน... {i+1}/{len(symbols)}")

        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info

            market_cap = info.get('marketCap', np.nan)
            book_value = info.get('bookValue', np.nan)
            total_revenue = info.get('totalRevenue', np.nan)

            if book_value and total_revenue and total_revenue > 0:
                book_to_market = book_value / total_revenue
            else:
                book_to_market = np.nan

            fundamentals[symbol] = {
                'market_cap': market_cap,
                'book_to_market': book_to_market
            }

        except Exception as e:
            fundamentals[symbol] = {
                'market_cap': np.nan,
                'book_to_market': np.nan
            }

    return pd.DataFrame(fundamentals).T

def main():
    """
    ตัวทดลองการคาดการณ์หุ้นจากอดีต
    """
    print("=" * 90)
    print("*** ตัวทดลองการคาดการณ์หุ้นจากอดีต (ทุกหุ้น + CSV Export) ***")
    print("=" * 90)

    # เส้นทางไปยังไฟล์ CSV
    csv_path = r"G:\download\code practice\Research\MIT2\nasdaqmedtomeg.csv"

    if not os.path.exists(csv_path):
        print(f"❌ ไม่พบไฟล์ CSV: {csv_path}")
        return

    # ขั้นตอนที่ 1: โหลดรายชื่อหุ้นจาก CSV
    print("\n[1/8] กำลังโหลดรายชื่อหุ้นจาก CSV...")
    symbols = load_symbols_from_csv(csv_path)

    if not symbols:
        return

    # กำหนดช่วงเวลา
    train_end_date = '2025-10-12'
    test_start_date = '2025-10-12'
    test_end_date = '2025-12-29'

    print(f"\nช่วงเวลาฝึกอบรม: 2020-01-01 ถึง {train_end_date}")
    print(f"ช่วงเวลาทดสอบ: {test_start_date} ถึง {test_end_date}")

    # ขั้นตอนที่ 2: ดึงข้อมูลฝึกอบรม
    print("\n[2/8] กำลังดึงข้อมูลฝึกอบรม...")
    train_data = get_historical_data(symbols, '2020-01-01', train_end_date)

    if train_data.empty:
        return

    # ขั้นตอนที่ 3: ดึงข้อมูลพื้นฐาน
    print("\n[3/8] กำลังดึงข้อมูลพื้นฐาน...")
    fundamentals = get_fundamentals(train_data.columns.tolist())

    # ขั้นตอนที่ 4: วิเคราะห์และเลือกหุ้นที่ดีที่สุด
    print("\n[4/8] กำลังวิเคราะห์และเลือกหุ้นที่ดีที่สุด...")
    top_stocks, train_returns, train_sharpe = predict_top_stocks(train_data, fundamentals, top_n=10)

    # ขั้นตอนที่ 5: ดึงข้อมูลทดสอบสำหรับหุ้นที่ดีที่สุด
    print("\n[5/8] กำลังดึงข้อมูลทดสอบสำหรับหุ้นที่ดีที่สุด...")
    test_data_top = get_historical_data(top_stocks, test_start_date, test_end_date)

    # ขั้นตอนที่ 6: ประเมินผลการคาดการณ์สำหรับหุ้นที่ดีที่สุด
    print("\n[6/8] กำลังประเมินผลการคาดการณ์สำหรับหุ้นที่ดีที่สุด...")
    prediction_results, accuracy = evaluate_predictions(top_stocks, test_data_top)

    # ขั้นตอนที่ 7: ดึงข้อมูลทดสอบสำหรับทุกหุ้น
    print("\n[7/8] กำลังดึงข้อมูลทดสอบสำหรับทุกหุ้น...")
    test_data_all = get_historical_data(train_data.columns.tolist(), test_start_date, test_end_date)

    # ขั้นตอนที่ 8: ประเมินผลการคาดการณ์สำหรับทุกหุ้น
    print("\n[8/9] กำลังประเมินผลการคาดการณ์สำหรับทุกหุ้น...")
    all_results_df, decile_stats = evaluate_all_predictions(train_data, test_data_all, train_sharpe)

    # ขั้นตอนที่ 9: บันทึกผลการวิเคราะห์
    print("\n[9/9] กำลังบันทึกผลการวิเคราะห์เป็นไฟล์ CSV...")

    # แสดงผลลัพธ์
    print("\n" + "=" * 90)
    print("*** ผลการทดลองการคาดการณ์หุ้น ***")
    print("=" * 90)

    print(f"\n🎯 เลือกหุ้นที่ดีที่สุด 10 อันดับ จากการวิเคราะห์อดีต:")
    for i, stock in enumerate(top_stocks, 1):
        train_sharpe_val = train_sharpe[stock] if stock in train_sharpe.index else np.nan
        print(f"  {i}. {stock} (Sharpe: {train_sharpe_val:.3f})")

    print(f"\n📈 ผลตอบแทนจริงในช่วงทดสอบ ({test_start_date} ถึง {test_end_date}):")
    positive_count = 0
    for stock, actual_return in prediction_results.items():
        if not np.isnan(actual_return):
            status = "📈 ขึ้น" if actual_return > 0 else "📉 ลง"
            print(f"  {stock}: {actual_return:+.2f}% {status}")
            if actual_return > 0:
                positive_count += 1
        else:
            print(f"  {stock}: ไม่มีข้อมูล")

    print(f"\n✅ ความแม่นยำสำหรับหุ้นที่ดีที่สุด: {positive_count}/{len(prediction_results)} ({accuracy:.1f}%)")

    # แสดงผลการประเมินสำหรับทุกหุ้น
    print(f"\n" + "=" * 90)
    print("*** การวิเคราะห์ความแม่นยำสำหรับทุกหุ้น (แบ่งตาม Sharpe Ratio) ***")
    print("=" * 90)

    print(f"\n📈 สรุปการคาดการณ์สำหรับทุกหุ้น ({len(all_results_df)} ตัว):")
    print(f"{'Decile':<8} {'Range':<8} {'Stocks':>6} {'Accuracy':>8} {'Avg Return':>10}")
    print("-" * 70)

    for stat in decile_stats:
        decile = stat['decile']
        count = stat['count']
        accuracy = stat['accuracy']
        avg_return = stat['avg_return']
        sharpe_range = ""
        if decile == 1:
            sharpe_range = "สูงสุด"
        elif decile == 10:
            sharpe_range = "ต่ำสุด"

        print(f"{decile:>2d} {sharpe_range:<8} {count:>6d} {accuracy:>8.1f}% {avg_return:>10.2f}%")

    overall_accuracy = all_results_df['actually_up'].mean() * 100
    overall_avg_return = all_results_df['actual_return'].mean()

    print("-" * 70)
    print(f"Overall     {len(all_results_df):>6d} {overall_accuracy:>8.1f}% {overall_avg_return:>10.2f}%")

    # บันทึกผลการวิเคราะห์เป็น CSV
    print("\n[9/9] กำลังบันทึกผลการวิเคราะห์เป็นไฟล์ CSV...")

    # บันทึกผลการคาดการณ์สำหรับทุกหุ้น
    all_results_df.to_csv('all_stocks_predictions.csv', index=False)
    print("✓ บันทึก all_stocks_predictions.csv สำเร็จ")

    # บันทึกผลการวิเคราะห์เดซิล
    decile_df = pd.DataFrame(decile_stats)
    decile_df.to_csv('decile_analysis.csv', index=False)
    print("✓ บันทึก decile_analysis.csv สำเร็จ")

    # บันทึกผลการคาดการณ์หุ้นที่ดีที่สุด
    top_results_df = pd.DataFrame({
        'stock': top_stocks,
        'sharpe': [train_sharpe.get(stock, np.nan) for stock in top_stocks],
        'actual_return': [prediction_results.get(stock, np.nan) for stock in top_stocks],
        'actually_up': [prediction_results.get(stock, np.nan) > 0 if not np.isnan(prediction_results.get(stock, np.nan)) else np.nan for stock in top_stocks]
    })
    top_results_df.to_csv('top10_predictions.csv', index=False)
    print("✓ บันทึก top10_predictions.csv สำเร็จ")

if __name__ == "__main__":
    main()