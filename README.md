# `lmlab` --- minimal local-first labs for comp-ling

`lmlab` is a web application acting as a playground for labs in a university teaching setting. Currently there's four main features:

1. POS-tagging using `stanza`.
2. Visualize different tokenizers.
3. Analyze and visualize processed sequences in LLMs.
4. Visualize next-token predictions in LLMs.

It's primary use is to be used in teaching non-CS students about aspects of computational linguistics. It is designed with flexibility and extensibility in mind, relying only on standard Python ML libraries (`transformers`/`torch`).

Existing solutions either require coding, aren't extensible enough, or require complicated local installation. This is unfeasible in many university settings, where computer labs don't offer that kind of flexibility in what's allowed to be installed, or where users might want to access the tool from home.

There's a built-in async queue, allowing deployment on less powerful servers. For reference, with a small 126M model, 4 tokenizers, and a stanza model, `lmlab` takes up about ~2GB RAM while actively loaded in memory. This is dynamically unloaded after a period of idleness.


# Data setup --- Swedish \& Sparv

Data setup for Swedish using Sparv POS tagging is a bit tricky, since it depends on `stanza` where models can't just be shared willy-nilly. First prepare a data directory by downloading the default tokenizer, e.g:

```
python3 -c "import stanza; stanza.download(lang='sv', processors='tokenize', model_dir='sparv')"
```

Then download the Sparv stanza models, e.g:

```
cd sparv

urls=(
  "https://svn.spraakbanken.gu.se/sb-arkiv/pub/stanza/stanza_pretrain.zip"
  "https://svn.spraakbanken.gu.se/sb-arkiv/pub/stanza/morph_stanza_full2.zip"
)

for u in "${urls[@]}"; do
  f=$(basename "$u")
  curl -L "$u" -o "$f"
  unzip -j "$f" "*.pt"
  rm "$f"
done
```

This should create a `./sparv` folder which contains all the
necessary files.

This can be extended to other languages by changing the POS/tokenizer etc using the environment variables (see below).

# Environment variables

Configuration is done mostly using environment variables since it's developed with docker in mind. Set these in `docker-compose.yml` (see the sample configuration) or in local environment if deploying outside of docker.

| Variable | Default | Description |
| --- | --- | --- |
| LMLAB_LM | AI-Sweden-Models/gpt-sw3-126m | Language model identifier. |
| LMLAB_TOKENIZERS | ["AI-Sweden-Models/gpt-sw3-126m", "gpt2", "facebook/xglm-564M", "google/byt5-small"] | List of tokenizer identifiers (JSON or comma-separated). |
| LMLAB_MAX_TOKENS | 50 | Maximum tokens for generation condition. |
| LMLAB_MAX_LENGTH_LM | 250 | Max sequence length for the language model analysis. |
| LMLAB_MAX_LENGTH_TOK | 90 | Max sequence length for tokenizers API. |
| LMLAB_TOP_K | 10 | How many candidates to include in lists. |
| LMLAB_IDLE_TIMEOUT | 15 | Idle timeout (minutes) before unloading models from memory. |
| LMLAB_POS_LANG | sv | Language for POS tagging. |
| LMLAB_POS_MODEL_DIR | ./sparv | Directory for POS tagging models. |
| LMLAB_POS_PRETRAIN | sv_talbanken.pretrain.pt | Pretrained POS model filename. |
| LMLAB_POS_MODEL | sv_talbanken_tagger.pt | POS tagger model filename. |

# Building

```
docker build -f docker/Dockerfile -t lmlab .
```

# Development

Run these commands while developing to do auto-reload (from
within ./lmlab directory), after setting up the data according to the steps
previously mentioned:

```
cd lmlab
fastapi dev app.py
npm run watch
```

# Code Structure

If you want to use this you probably want to customize it. `lmlab` is boilerplate code, and should be easily extensible if you dare look inside the code. The general idea is that there's a FastAPI backend which sets up API endpoints which the frontend interacts with. Since FastAPI includes support for templating HTML and static serving, it is also used for hosting the actual web server. When a user interacts with the website and submits an API request, it is sent to FastAPI which includes async lock (so any model is only loaded once), enabling scaling up at least a little bit.

Here's the general code structure:

```
lmlab/app.py
    contains the FastAPI web app and API for the different
    features. also includes lazy loading logic.

lmlab/src
    contains the typescript files for interacting with the
    backend.

lmlab/templates
    contains the html

lmlab/static
    contains resources which aren't built or
    dynamically generated; css, images, etc.
```
