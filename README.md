# OCR Fé Representação (Coolify)

Serviço FastAPI com Tesseract + OpenCV para ler crop de produto e etiqueta de preço no fluxo **Fazer Pesquisa** (app web_fe).

Repositório standalone: o **Dockerfile fica na raiz** — no Coolify deixe **Base Directory** vazio.

## Endpoints

| Método | Path | Auth | Descrição |
|--------|------|------|-----------|
| `GET` | `/health` | — | Healthcheck |
| `POST` | `/ocr/pesquisa` | `X-OCR-Secret` | OCR + match fuzzy em `codigos` (tipo `interna`) |

### `POST /ocr/pesquisa` (multipart)

| Campo | Tipo | Obrigatório |
|-------|------|-------------|
| `produto_crop` | file (JPEG/PNG) | sim |
| `preco_crop` | file (JPEG/PNG) | sim |
| `tipo` | `interna` \| `externa` | sim |
| `industria` | string | sim |

Resposta JSON (resumo): `texto_ocr`, `preco`, `descricao`, `sugerido`, `candidatos` (top 3 se interna).

## Variáveis de ambiente (Coolify)

| Variável | Obrigatória | Descrição |
|----------|-------------|-----------|
| `OCR_SHARED_SECRET` | sim | Valor do header `X-OCR-Secret` (igual a `VITE_PESQUISA_OCR_SECRET` no front) |
| `SUPABASE_URL` | sim* | URL do projeto Supabase (`https://….supabase.co`) |
| `SUPABASE_SERVICE_ROLE_KEY` | sim* | Service role — **somente leitura** de `public.codigos` |
| `CORS_ORIGINS` | não | Default `*` (browser Vite/prod). Alternativa: lista CSV de origins |
| `HOST` | não | Default `0.0.0.0` |
| `PORT` | não | Default `8000` (Coolify mapeia a porta do container) |

\*Obrigatórias para `tipo=interna`. Em `externa` o serviço devolve só o texto OCR + preço.

## Deploy Coolify

1. Novo app a partir deste repo; Build Pack: **Dockerfile**.
2. **Base Directory**: vazio (Dockerfile na raiz do repo).
3. Porta: `8000`.
4. Healthcheck path: `/health`.
5. Configure as env vars acima (runtime).
6. Domínio público (ex.: `https://ocr-pesquisa.seudominio.com`) — usar como `VITE_PESQUISA_OCR_URL` no build do front.

## Rodar local

Pré-requisito: [Tesseract](https://github.com/tesseract-ocr/tesseract) com idiomas `por` + `eng` no PATH.

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate
# Linux/macOS
# source .venv/bin/activate

pip install -r requirements.txt

set OCR_SHARED_SECRET=dev-secret
set SUPABASE_URL=https://SEU_PROJETO.supabase.co
set SUPABASE_SERVICE_ROLE_KEY=eyJ...

uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Health:

```bash
curl http://127.0.0.1:8000/health
```

Exemplo OCR:

```bash
curl -X POST http://127.0.0.1:8000/ocr/pesquisa ^
  -H "X-OCR-Secret: dev-secret" ^
  -F produto_crop=@produto.jpg ^
  -F preco_crop=@preco.jpg ^
  -F tipo=interna ^
  -F industria=PREDILECTA
```

## Docker local

```bash
docker build -t pesquisa-ocr .
docker run --rm -p 8000:8000 ^
  -e OCR_SHARED_SECRET=dev-secret ^
  -e SUPABASE_URL=https://SEU_PROJETO.supabase.co ^
  -e SUPABASE_SERVICE_ROLE_KEY=eyJ... ^
  pesquisa-ocr
```
