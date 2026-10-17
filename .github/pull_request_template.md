## Özet

<!-- Bu PR'ın çözdüğü problemi veya eklediği özelliği kısaca açıklayın. -->

## Mimari İlkeler ve Güvenlik Kontrol Listesi

- [ ] `uv run ruff check .` sıfır hata ile geçti.
- [ ] `uv run mypy src tests` (strict mode) sıfır hata ile geçti.
- [ ] `uv run pytest -v` ağ bağlantısı ve API anahtarı olmadan tamamen yeşil geçti.
- [ ] İz dosyalarında veya test verilerinde (`nvapi-`, `sk-`, `Bearer`) hiçbir gizli bilgi bulunmadığı doğrulandı (R7).
- [ ] Yeni davranışlar için `FakeLLM` veya sentetik izlerle deterministik birim testler eklendi.
