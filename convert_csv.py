import pandas as pd
import os

print("="*60)
print("🔄 CONVERTING SPAM CSV TO CORRECT FORMAT")
print("="*60)

# Read your original CSV (assuming tab-separated)
try:
    # Try reading as tab-separated (your format)
    df = pd.read_csv('data/spam.csv', sep='\t', header=None, encoding='utf-8')
    print("✅ Loaded as tab-separated format")
except:
    try:
        # Try reading as normal CSV
        df = pd.read_csv('data/spam.csv', encoding='utf-8')
        print("✅ Loaded as CSV format")
    except:
        print("❌ Could not read file. Make sure data/spam.csv exists")
        exit()

print(f"\n📊 Original shape: {df.shape}")

# Check column names
if len(df.columns) == 2:
    # Rename columns
    df.columns = ['label', 'text']
    
    # Convert labels: ham->0, spam->1
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})
    
    # Remove any rows with NaN
    df = df.dropna()
    
    print(f"✅ Converted {len(df)} messages")
    print(f"   Ham (0): {sum(df['label']==0)}")
    print(f"   Spam (1): {sum(df['label']==1)}")
    
    # Save in correct format
    df.to_csv('data/spam.csv', index=False, encoding='utf-8')
    print("\n✅ Saved to data/spam.csv in correct format!")
    
    print("\n📝 First 5 rows after conversion:")
    print(df.head())
    
else:
    print(f"⚠️ Unexpected format. Columns: {df.columns}")
    print("\n📝 First 3 rows:")
    print(df.head(3))

print("\n" + "="*60)
print("🎯 DONE! Now run: python main.py")
print("="*60)