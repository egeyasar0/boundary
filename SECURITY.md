# Güvenlik Politikası (Security Policy)

**boundary.** projesi, ajan izlerinin (*traces*) ve araç çağrılarının güvenliğini değişmez bir mimari ilke (**R7 — Sandbox & Secret Hygiene**) olarak ele alır.

## Mimari Güvenlik Garantileri (R7)

1. **Salt-Okunur (Read-Only) Araç Kutusu:** `list_files`, `read_file` ve `grep` araçları yalnızca okuma yapar; kabuk (*shell*) yürütme veya dosya yazma yetkisi yoktur.
2. **Kök Dizin Kilidi (Sandbox Hardening):** Tüm dosya yolu erişimleri hedef fixture/repo kök dizinine kilitlidir. `..` (path traversal), mutlak yollar ve sembolik bağ (*symlink*) kaçışları çalışma zamanında engellenir ve birim testleriyle doğrulanır.
3. **Sıfır Gizli Bilgi Sızıntısı (Secret Redaction):**
   - API anahtarları (`BOUNDARY_API_KEY`) yalnızca ortam değişkenlerinden okunur ve `config_hash` hesaplamasına veya JSONL izlerine (`traces.jsonl`) **asla** dahil edilmez.
   - Hata mesajlarında veya istisnalarda (`LLMError`, `Trace.error`) geçebilecek `sk-...`, `nvapi-...`, `Bearer ...` desenleri otomatik olarak `[REDACTED]` ile maskelenir.

## Güvenlik Açığı Bildirimi

Bir güvenlik açığı (ör. sandbox kaçışı veya iz dosyalarına secret sızıntısı) tespit ederseniz lütfen halka açık bir GitHub Issue açmak yerine doğrudan depo sahibiyle ([`@egeyasar0`](https://github.com/egeyasar0)) iletişime geçin veya GitHub **Private Vulnerability Reporting** özelliğini kullanın.
