import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
import matplotlib
matplotlib.use('Qt5Agg')

# Günlük ya?mur verisini yükleyin
df = pd.read_csv('datasets/processed_rain/300236.csv', sep=";", encoding='windows-1252')

# Tarih sütununu datetime format?na çevirin
df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d %H:%M:%S', errors='coerce')

# Tarih sütununu indeks olarak ayarlay?n
df.set_index('Date', inplace=True)
df.head()
# Ayl?k toplam ya?mur miktar?n? hesaplay?n
monthly_rain = df.resample('ME').mean()
monthly_rain.head(10)

# Görselle?tirme
plt.figure(figsize=(19, 10))

# Ayl?k ya?mur verisini çiz
plt.plot(monthly_rain.index, monthly_rain['rain'], color='blue', linestyle='-')

# X ekseninde y?llar? belirli aral?klarla göster
plt.gca().xaxis.set_major_locator(mdates.YearLocator(5))  # Her 5 y?lda bir
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

plt.title('Ayl?k ortalama Ya?mur Miktar?')
plt.xlabel('Tarih')
plt.ylabel('Ort Ya?mur (mm)')
plt.xticks(rotation=45)
plt.grid(True)

plt.tight_layout()
plt.show()


# kar için örnek:

dff = pd.read_csv('datasets/processed_snow/300665.csv', sep=";", encoding='windows-1252')
dff['Date'] = pd.to_datetime(dff['Date'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
dff.set_index('Date', inplace=True)
dff.head()
monthly_snow = dff.resample('ME').mean()
monthly_snow.head(10)

plt.figure(figsize=(19, 10))
plt.plot(monthly_snow.index, monthly_snow['snow'], color='blue', linestyle='-')
plt.gca().xaxis.set_major_locator(mdates.YearLocator(5))  # Her 5 y?lda bir
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.title('Ayl?k ortalama kar Miktar?')
plt.xlabel('Tarih')
plt.ylabel('Ort kar (cm)')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

# temp için örnek:

df_ = pd.read_csv('datasets/processed_temp/300236.csv', sep=";", encoding='windows-1252')
df_['Date'] = pd.to_datetime(df_['Date'], format='%Y-%m-%d', errors='coerce')
df_.set_index('Date', inplace=True)
df_.head(15)


plt.figure(figsize=(19, 10))
plt.plot(df_.index, df_['temp'], color='blue', linestyle='-')
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.title('Ayl?k ortalama yer alt? suyu s?cakl???')
plt.xlabel('Tarih')
plt.ylabel('Ort su s?cakl??? (Celcius)')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()


