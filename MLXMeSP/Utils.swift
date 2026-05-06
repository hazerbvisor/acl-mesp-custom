//
//  For licensing see accompanying LICENSE file.
//  Copyright (C) 2025 Apple Inc. All Rights Reserved.
//
//  Utils.swift
//  mlx-mesp
//
//  Created by Congzheng Song on 9/18/25.
//

import MLX
import Tokenizers
import Hub

public typealias Messages = [[String: String]]

public enum SpecialTokenIds: Int {
    case pad = 0
    case null = -100
}

public extension LanguageModelConfigurationFromHub {
    func updateTokenizerConfig() async throws -> Config? {
        guard let tokenizerConfig = try await tokenizerConfig else {
            return nil
        }
        // Workaround: replacement tokenizers for unhandled values in swift-transformers]
        let replacementTokenizers = [
            "InternLM2Tokenizer": "PreTrainedTokenizer",
            "Qwen2Tokenizer": "PreTrainedTokenizer",
            "Qwen3Tokenizer": "PreTrainedTokenizer",
            "CohereTokenizer": "PreTrainedTokenizer",
        ]
        if let tokenizerClass = tokenizerConfig.tokenizerClass?.string(),
            let replacement = replacementTokenizers[tokenizerClass]
        {
            if var dictionary = tokenizerConfig.dictionary() {
                dictionary["tokenizer_class"] = .init(replacement)
                return Config(dictionary)
            }
        }
        return tokenizerConfig
    }
}

public func loadTokenizer(config: LanguageModelConfigurationFromHub, hub: HubApi = .shared) async throws -> PreTrainedTokenizer
{
    guard let tokenizerConfig = try await config.updateTokenizerConfig() else {
        throw MLXError.caught("Failed to load updated tokenizer config for \(config).")
    }
    let tokenizerData = try await config.tokenizerData
    return try PreTrainedTokenizer(tokenizerConfig: tokenizerConfig, tokenizerData: tokenizerData)
}

public func processCompletionData(
    _ data: [Messages],
    tokenizer: PreTrainedTokenizer,
    inputIdsKey: String = "input_ids",
    labelIdsKey: String = "label_ids",
    maxLength: Int = 256,
    padToMaxLength: Bool = true,
) -> [[String: MLXArray]] {
    func process(messages: Messages) -> [String: MLXArray]? {
        do {
            guard messages.count == 2 else { return nil }
            var tokens = try tokenizer.applyChatTemplate(messages: messages, addGenerationPrompt: false)
            let offset = try tokenizer.applyChatTemplate(messages: [messages.first!], addGenerationPrompt: false).count
            guard offset < maxLength, offset > 0 else { return nil }
            // Truncate
            if tokens.count > maxLength  { tokens = Array(tokens[..<maxLength]) }
            // Target labels shifted by 1 and tokens up to offset is masked.
            var labels = Array(repeating: SpecialTokenIds.null.rawValue, count: offset - 1) + Array(tokens[offset...]) + Array(repeating: SpecialTokenIds.pad.rawValue, count:1)
            // Pad
            if padToMaxLength, tokens.count < maxLength {
                tokens.append(contentsOf: Array(repeating: SpecialTokenIds.pad.rawValue, count: maxLength - tokens.count))
                labels.append(contentsOf: Array(repeating: SpecialTokenIds.null.rawValue, count: maxLength - labels.count))
            }
            return [
                inputIdsKey: MLXArray(tokens, [1, maxLength]),
                labelIdsKey: MLXArray(labels, [1, maxLength])
            ]
        } catch {
            return nil
        }
    }

    return data.compactMap(process)
}

/// Pre-tokenized data format for base model language modeling
public struct PreTokenizedSample: Codable {
    public let tokens: [Int]
}

/// Process pre-tokenized data for base model language modeling (no chat template)
/// This format is for pure LM tasks where tokens are already prepared.
public func processPreTokenizedData(
    _ samples: [PreTokenizedSample],
    inputIdsKey: String = "input_ids",
    labelIdsKey: String = "label_ids",
    maxLength: Int = 256
) -> [[String: MLXArray]] {
    return samples.compactMap { sample in
        var tokens = sample.tokens

        // Ensure proper length
        guard tokens.count >= 2 else { return nil }

        // Truncate if needed
        if tokens.count > maxLength {
            tokens = Array(tokens.prefix(maxLength))
        }

        // Pad if needed
        let actualLength = tokens.count
        if tokens.count < maxLength {
            tokens.append(contentsOf: Array(repeating: SpecialTokenIds.pad.rawValue, count: maxLength - tokens.count))
        }

        // Labels: shifted by 1 for next-token prediction
        // First token has no label, last token's label is padding
        var labels = Array(tokens.dropFirst()) + [SpecialTokenIds.null.rawValue]

        // Mask padding tokens in labels
        for i in actualLength..<labels.count {
            labels[i] = SpecialTokenIds.null.rawValue
        }

        return [
            inputIdsKey: MLXArray(tokens, [1, maxLength]),
            labelIdsKey: MLXArray(labels, [1, maxLength])
        ]
    }
}
