# Vendored swift-numerics

This directory contains the production source code from Apple Swift Numerics **1.1.1**, rearranged only for the XTool Mobile build graph.

Upstream: https://github.com/apple/swift-numerics

The original license is preserved in `LICENSE.txt`.

Why vendored: XTool Mobile's portable `xtool-mobile.json` build path does not resolve arbitrary SwiftPM dependencies on-device. These sources are therefore compiled locally with the same Swift compiler as the app, avoiding a prebuilt Swift module compatibility problem.
