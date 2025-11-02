"""Prompt template for post-report question answering."""

QA_PROMPT = """Derin öğrenme değerlendirme danışmanısın. Aşağıdaki rapor özetine ve eğitim artefaktlarına dayanarak kullanıcının sorusunu Türkçe yanıtla.

ÖNCEKİ RAPOR ÖZETİ:
- Summary: {summary}
- Güçlü Yönler: {strengths}
- Zayıf Yönler: {weaknesses}
- Risk: {risk}
- Yayın Profili: {deploy_profile}
- Aksiyonlar: {actions}

ÇEKİRDEK METRİKLER:
{metrics}

EĞİTİM KONFİGÜRASYONU:
{config}

VERİ SETİ ÖZETİ:
{dataset}

EPOCH GEÇMİŞİ:
{history}

EĞİTİM KODU:
{training_code}

ARTEFAKT DURUMU:
{artefacts}

KULLANICI SORUSU:
{question}

YANIT FORMATIN:
{{
  "answer": "Soruya verilen kapsamlı yanıt. Somut sayılar kullan ve varsayımlarını belirt.",
  "references": ["Hangi artefakt satırları veya dosyalarına referans verdiğini açıkça yaz."],
  "follow_up_questions": ["Gerekirse sonraki adım için önerdiğin sorular."],
  "notes": "Opsiyonel ek açıklamalar."
}}

Sadece geçerli JSON döndür. Tüm açıklamalar Türkçe olmalı."""
