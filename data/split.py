import pandas as pd

# 讀取原始 CSV
df = pd.read_csv('retail_trend_data.csv')  # 假設你已儲存為這檔案名

# 取得所有唯一的 SKU
skus = df['SKU'].unique()

for sku in skus:
    sub = df[df['SKU'] == sku].copy()
    # 選擇欄位（可按需修改）
    sub = sub[['Date','SKU','Category','Price','Sales','Stock','Promotion']]
    # 儲存為單一檔案
    out_filename = f'{sku}.csv'
    sub.to_csv(out_filename, index=False)
    print(f'Saved {out_filename} with {len(sub)} rows')
