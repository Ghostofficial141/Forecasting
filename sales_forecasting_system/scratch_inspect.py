import pandas as pd
try:
    df = pd.read_excel(r'c:\Users\Asus\Desktop\New folder\Forecasting Case- Study (1).xlsx')
    print("Columns:", df.columns.tolist())
    print("\nHead:\n", df.head())
    print("\nStates:", df.iloc[:, 0].unique() if len(df.columns) > 0 else "None")
except Exception as e:
    print("Error:", e)
