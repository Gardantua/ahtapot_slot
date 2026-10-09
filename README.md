# Epsilon-greedy ve çok kollu bandit deneyi

Keşfetme ile eldeki en iyi seçeneği kullanma arasındaki dengeyi anlamak için yaptığım ödev. Sabit ve değişen epsilon stratejilerini NumPy ile uygulayıp ortalama ödül, optimal eylem oranı ve kümülatif regret üzerinden karşılaştırdım.

Hazır pekiştirmeli öğrenme kütüphanesi kullanmadım. Deney kodu [epsilon_greedy_odev.py](epsilon_greedy_odev.py), parametre seçimlerim ve yorumlarım [rapor.md](rapor.md) içinde. PNG dosyaları bu deneyin kayıtlı grafikleridir.

## Çalıştırma

```bash
python -m venv .venv
python -m pip install numpy matplotlib
python epsilon_greedy_odev.py
```

Sanal ortamı etkinleştirerek yükle. Ana deney çok sayıda tekrar içerdiğinden çalışması zaman alabilir ve grafik dosyalarını yeniden üretir. Repo adı eski; içerik bir slot oyunu değil, çok kollu bandit simülasyonudur.
