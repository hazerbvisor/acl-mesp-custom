# XTool Mobile swift-transformers bootstrap

This directory vendors the production `Hub` and `Tokenizers` sources from
Hugging Face `swift-transformers` **0.1.24**, matching this repository's
`Package.resolved`.

The vendored `Hub.swift` uses `Bundle.main` instead of SwiftPM-generated
`Bundle.module`, and the fallback tokenizer JSON files are copied into the app
bundle by `xtool-mobile.json`.

For the first on-device MeSP milestone, training uses the bundled pre-tokenized
`wikitext2_base.jsonl`. A small module named `Jinja` preserves the API surface
needed to compile upstream Tokenizers but intentionally throws if chat-template
rendering is invoked. Replace it with the real Jinja dependency before using
chat-format/instruction training.

The upstream swift-transformers license is preserved in `LICENSE`.
