import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('logs/log_1m.txt')

df = df[8:]
df.columns = ['loss']
df = df.dropna()

df['loss'] = df['loss'].apply(lambda x: x[10:])
df['loss'] = df['loss'].astype(float)
df = df.reset_index(drop=True)

plt.plot(range(len(df)), df['loss'])
plt.xlabel('Steps / 500')
plt.ylabel('Loss')
plt.savefig('figs/fig_1m.png')
