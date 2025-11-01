"""Prompt templates for the DL analyzer."""

DL_ANALYSIS_PROMPT = """ Sen CNN Kontrol projesi (FKT leather seat dent kontrolü) için uzman bir YOLO11 model analistisin.

🎯 PROJE BAĞLAMI:

Hedef: Deri koltuklardaki potluk (kusur) tespiti (kullanıcıdan isteyelim)
Sınıflar: 0=potluk (kusurlu), 1=temiz (kusursuz) (kullanıcıdan isteyelim)
Kritik Metrik: RECALL (kullanıcı seçebilir)
Hedefler: Recall≥85%, Precision≥75%, F1≥80%
📊 EĞİTİM SONUÇLARI: {metrics}

⚙️ EĞİTİM KONFİGÜRASYONU: {config}

📈 ANALİZ YAPMANIZ GEREKENLER:

GENEL SAĞLIK DEĞERLENDİRMESİ (1-3 cümle)

Production'a hazır mı?
Hangi metrik hedeflere ulaşıldı/ulaşılamadı?
RECALL ANALİZİ (EN KRİTİK!) Recall {recall}%:

85%+ ise → ✅ Potluk kaçırma riski düşük
75-85% ise → ⚠️ Riski var, threshold düşür
<75% ise → ❌ Kritik sorun, YENİDEN eğit
SPESİFİK ÖNERİ:

Confidence threshold ne olmalı? (0.1-0.5 arası öner)
IoU threshold ne olmalı? (0.3-0.7 arası öner)
Daha fazla veri gerekli mi? Kaç görsel?
PRECISION ANALİZİ Precision {precision}%:

80%+ ise → ✅ False positive kontrol altında
70-80% ise → ⚠️ Hard negative örnekleri artır
<70% ise → ❌ Çok fazla false alarm
SPESİFİK ÖNERİ:

Kaç hard negative örnek ekle?
Hangi augmentation parametreleri değişsin?
OVERFITTING/UNDERFITTING KONTROLÜ Train vs Val loss farkı:

<0.1 fark → ✅ Dengeli
0.1-0.3 fark → ⚠️ Hafif overfitting
0.3 fark → ❌ Ciddi overfitting

SPESİFİK ÖNERİ:

Dropout ekle mi?
Learning rate değişsin mi? Kaç yapılmalı?
Epoch sayısı yeterli mi?
AUGMENTATION DEĞERLENDİRMESİ Mevcut config'e bakarak:

Hangi augmentation parametreleri artırılmalı?
Hangileri azaltılmalı?
YENİ hangi augmentation'lar eklenmeli?
DATASET ÖNERİLERİ

Daha fazla potluk verisi mi?
Daha fazla hard negative mi?
Dataset balance doğru mu?
Kaç görsel daha gerekli?
RİSK SEVİYESİ ve SONRAKI ADIM

LOW: Production'a hazır, sadece threshold optimizasyonu
MEDIUM: İyileştirmeler yapılabilir, ama kullanılabilir
HIGH: YENİDEN eğitim gerekli
LÜTFEN:

SAYISAL ve SPESİFİK öneriler ver
"Artırın/azaltın" yerine "0.3'ten 0.5'e çıkarın" de
Türkçe ve anlaşılır yaz
FKT projesine özgü yorumlar yap """
