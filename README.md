# ParkVision AI

ParkVision AI, üstten kamera görüntüsüyle park alanı doluluk durumunu takip eden ve plaka okumadan otomatik park oturumu/ücretlendirme yapan akıllı otopark prototipidir.

Bu sürüm, mevcut maske ve eğitilmiş model ile her park yerini boş/dolu olarak sınıflandırır. Bir park yeri belirli süre dolu kalırsa sistem geçici bir araç ID'si ve oturum ID'si oluşturur. Park yeri boşaldığında oturum kapanır ve ücret hesaplanır.

## Senaryo

Sistem araçlara plaka gibi kalıcı kimlik vermez. Bunun yerine kamera görüntüsü içinde geçerli olan geçici takip/oturum kimlikleri kullanır:

```text
Park yeri: A07
Araç ID: V-0004
Oturum ID: S-00004
Süre: 18 dk
Ücret: 60 TL
```

Bu yaklaşım üstten çekimde plaka okunamadığı durumlar için daha gerçekçidir. Proje anlatımı şu fikre dayanır:

> Plaka, bariyer veya fiziksel sensör kullanmadan; yalnızca kamera görüntüsüyle park alanı kullanımını izleyen ve her park oturumu için otomatik ücret hesaplayan sistem.

## Özellikler

- Maske tabanlı park alanı tanımlama
- Boş/dolu park yeri tespiti
- Geçici araç ID'si üretimi
- Park oturumu başlatma/kapatma
- Saatlik ücret hesaplama
- Canlı video üstüne kutu ve etiket çizimi
- Streamlit dashboard prototipi
- Aktif oturum ve geçmiş oturum tabloları

## Dosyalar

```text
main.py        OpenCV penceresinde canlı demo
dashboard.py   Streamlit dashboard
detector.py    Park yeri tespit ve çizim mantığı
billing.py     Oturum ve ücretlendirme mantığı
util.py        Model tahmini ve maske bileşenleri
```

## Kurulum

Python 3.10 önerilir. Mevcut model ve bağımlılık sürümleri eski olduğu için en sorunsuz kurulum Python 3.10 ile yapılır.

```bash
pip install -r requirements.txt
```

## Çalıştırma

OpenCV pencereli demo:

```bash
python main.py
```

Dashboard:

```bash
streamlit run dashboard.py
```

## Geliştirme Fikri

Bu prototip doluluk geçişinden geçici araç ID'si üretir. Daha gelişmiş sürümde YOLO + ByteTrack/SORT eklenerek araçlar park alanına girmeden önce de hareket boyunca takip edilebilir. Mevcut mimaride bu katman `detector.py` tarafına eklenebilir; `billing.py` aynı kalır.
