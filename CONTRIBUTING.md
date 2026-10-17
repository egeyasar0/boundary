# Katkıda Bulunma Rehberi (Contributing to boundary.)

**boundary.** projesine katkıda bulunmak istediğiniz için teşekkürler! Deneysel tekrar üretilebilirliği ve kod kalitesini korumak için aşağıdaki mühendislik kurallarına uyulması zorunludur.

## Değişmez Kurallar

1. **Ağsız ve Deterministik Testler:** Hiçbir birim veya entegrasyon testi gerçek bir LLM API'sine ağ çağrısı yapmamalı ve `BOUNDARY_API_KEY` gerektirmemelidir. Tüm testlerde `FakeLLM` protokolü kullanılmalıdır.
2. **Katı Tip Güvenliği:** Tüm fonksiyon ve metotlarda Python 3.12+ tip ipuçları (*type hints*) zorunludur (`mypy --strict`).
3. **Eşit Bütçe ve Occam İlkeleri (R1–R15):** Yeni bir profil sinyali veya alt ajan sentez kuralı eklenirken formüller docstring içinde belgelenmeli ve sentetik izlerle test edilmelidir.

## Geliştirme Ortamı Kurulumu

```bash
uv sync --all-extras
```

## Pull Request Öncesi Kalite Kontrolleri

Gönderilen her PR, GitHub Actions CI üzerinde aşağıdaki üç kontrolü **sıfır hata** ile geçmelidir:

```bash
# 1. Kod stili ve lint kontrolü
uv run ruff check .

# 2. Katı statik tip kontrolü
uv run mypy src tests

# 3. Birim ve entegrasyon testleri
uv run pytest -v
```

## Commit Mesajı Standartları

Proje **Conventional Commits** standardını kullanır:
- `feat(profile): ...`
- `fix(sandbox): ...`
- `test(subagent): ...`
- `chore(ci,git): ...`
- `docs(readme): ...`
