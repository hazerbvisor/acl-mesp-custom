//
//  For licensing see accompanying LICENSE file.
//  Copyright (C) 2025 Apple Inc. All Rights Reserved.
//
//  Model.swift
//  MLXMeSPExample
//
//  Created by Congzheng Song on 9/18/25.
//

import Foundation
import Hub
import MLX
import MLXMeSP
import Tokenizers

/// Training mode detected from training_config.json
enum TrainingMode {
    case mebp           // Full backprop (baseline)
    case loraStructured // Swift-side structured backward (MeSP, ours)
}

/// Protocol for unified runner interface
protocol TrainingRunner {
    var trainableParams: [String: MLXArray] { get set }
    var checkpoints: MLXArrayDictionary { get set }
}

extension MeBPRunner: TrainingRunner {}
extension LoRAStructuredRunner: TrainingRunner {}

/// Data format type for training
enum TrainingDataFormat {
    case chatFormat           // Chat-style data with user/assistant messages
    case preTokenized         // Pre-tokenized tokens for base model LM
}

struct Model {
    var data: [Messages]
    var preTokenizedData: [PreTokenizedSample]
    var dataFormat: TrainingDataFormat
    let tokenizer: PreTrainedTokenizer
    var mebpRunner: MeBPRunner?
    var structuredRunner: LoRAStructuredRunner?
    let trainingMode: TrainingMode
    private let modelDir: URL
    private let functionPaths: [String]
    private let paramsLoaders: [() throws -> [String: MLXArray]]
    private let configs: [RunFunctionConfig]
    private let numLayers: Int

    // Attention config for Q/K/V backward
    private let nHeads: Int
    private let nKvHeads: Int
    private let headDim: Int

    // Exported model's sequence length and batch size (must match training input)
    let exportedSeqLength: Int
    let exportedBatchSize: Int

    init(modelDir: URL, useInMemoryStorage: Bool = true) async throws {
        self.modelDir = modelDir
        let config = LanguageModelConfigurationFromHub(modelFolder: modelDir)
        tokenizer = try await loadTokenizer(config: config)
        let modelConfig = try await config.modelConfig
        let numLayers = modelConfig.numHiddenLayers.integer()!
        self.numLayers = numLayers

        // Read attention config
        let hiddenSize = modelConfig.hiddenSize.integer()!
        let nHeads = modelConfig.numAttentionHeads.integer()!
        let nKvHeads = modelConfig.numKeyValueHeads?.integer() ?? nHeads  // Default to nHeads if not specified (MHA)
        let headDim = hiddenSize / nHeads
        self.nHeads = nHeads
        self.nKvHeads = nKvHeads
        self.headDim = headDim
        print("[Model] Attention config: nHeads=\(nHeads), nKvHeads=\(nKvHeads), headDim=\(headDim)")

        // Load training config to determine mode
        var mode: TrainingMode = .mebp
        var seqLength = 256  // Default
        var batchSize = 1    // Default

        let trainingConfigURL = modelDir.appendingPathComponent("training_config.json")
        if let trainingConfigData = try? Data(contentsOf: trainingConfigURL),
           let trainingConfig = try? JSONDecoder().decode(TrainingConfig.self, from: trainingConfigData) {
            seqLength = trainingConfig.seqLength
            batchSize = trainingConfig.batchSize
            mode = trainingConfig.swiftStructuredBackward ? .loraStructured : .mebp
            print("[Model] Training mode: \(trainingConfig.mode), seq_length: \(seqLength), batch_size: \(batchSize)")
        } else {
            print("[Model] No training_config.json found, using default MeBP mode with seq_length=256, batch_size=1")
        }

        self.exportedSeqLength = seqLength
        self.exportedBatchSize = batchSize
        self.trainingMode = mode

        // Build function paths and params loaders
        var functionPaths: [String] = [modelDir.appendingPathComponent("embedding.mlxfn").path()]
        var paramsLoaders: [() throws -> [String: MLXArray]] = [{ try loadArrays(url: modelDir.appendingPathComponent("embedding.safetensors")) }]

        // Swift structured backward uses layer{N}_forward_swift.mlxfn
        let isLoraStructured: Bool
        if case .loraStructured = mode {
            isLoraStructured = true
        } else {
            isLoraStructured = false
        }

        for i in 0..<numLayers {
            if isLoraStructured {
                // Swift structured: layer-specific forward functions with intermediates
                functionPaths.append(modelDir.appendingPathComponent("layer\(i)_forward_swift.mlxfn").path())
            } else {
                // Standard: shared forward function
                functionPaths.append(modelDir.appendingPathComponent("transformer_block_forward.mlxfn").path())
            }
            paramsLoaders.append({ try loadArrays(url: modelDir.appendingPathComponent("layer\(i)_transformer_block.safetensors")) })
        }

        functionPaths.append(modelDir.appendingPathComponent("loss.mlxfn").path())
        paramsLoaders.append({
            let embeddingParams = try loadArrays(url: modelDir.appendingPathComponent("embedding.safetensors"))
            let lossParams = try loadArrays(url: modelDir.appendingPathComponent("loss.safetensors"))
            return embeddingParams.merging(lossParams) { (_, new) in new }
        })

        // Add backward paths based on mode
        switch mode {
        case .mebp:
            // Full backprop — all layers
            for i in stride(from: numLayers - 1, to: -1, by: -1) {
                functionPaths.append(modelDir.appendingPathComponent("transformer_block_backward.mlxfn").path())
                paramsLoaders.append({ try loadArrays(url: modelDir.appendingPathComponent("layer\(i)_transformer_block.safetensors")) })
            }
        case .loraStructured:
            // No exported backward — Swift computes structured backward in-process
            break
        }

        let configData = try Data(contentsOf: modelDir.appendingPathComponent("run_configs.json"))
        let configs = try JSONDecoder().decode([RunFunctionConfig].self, from: configData)

        // Store configuration for recreation
        self.functionPaths = functionPaths
        self.paramsLoaders = paramsLoaders
        self.configs = configs

        // Create appropriate runner based on mode
        switch mode {
        case .mebp:
            let storageType: BaseRunner<MeBPContext>.StorageType = useInMemoryStorage ? .inMemory : .mmap()
            mebpRunner = try MeBPRunner(
                functionPaths: functionPaths,
                trainableParamsLoader: { try loadArrays(url: modelDir.appendingPathComponent("trainable_params.safetensors"))},
                paramsLoaders: paramsLoaders,
                configs: configs,
                storageType: storageType
            )
            print("Init MeBPRunner created successfully with \(useInMemoryStorage ? "in-memory" : "memory-mapped") storage")

        case .loraStructured:
            let storageType: BaseRunner<LoRAStructuredContext>.StorageType = useInMemoryStorage ? .inMemory : .mmap()
            mebpRunner = nil
            structuredRunner = try LoRAStructuredRunner(
                functionPaths: functionPaths,
                trainableParamsLoader: { try loadArrays(url: modelDir.appendingPathComponent("trainable_params.safetensors"))},
                paramsLoaders: paramsLoaders,
                configs: configs,
                storageType: storageType,
                numLayers: numLayers,
                nHeads: nHeads,
                nKvHeads: nKvHeads,
                headDim: headDim
            )
            structuredRunner?.enableSwiftBackward()
            print("Init LoRAStructuredRunner created successfully with Swift-side structured backward")
        }

        // Try to load training data (priority order: pre-tokenized > chat format)
        var data: [Messages] = []
        var preTokenizedData: [PreTokenizedSample] = []
        var dataFormat: TrainingDataFormat = .chatFormat

        // Priority: pre-tokenized base model data first
        let preTokenizedFiles = ["wikitext2_base", "wikitext2-base"]  // For base model LM
        let chatFormatFiles = ["wikitext2-demo", "wikitext2", "wiki-text-demo"]  // Chat format

        // Try pre-tokenized format first (for base models)
        for fileName in preTokenizedFiles {
            if let fileURL = Bundle.main.url(forResource: fileName, withExtension: "jsonl") {
                do {
                    let content = try String(contentsOf: fileURL, encoding: .utf8)
                    let decoder = JSONDecoder()
                    for line in content.split(separator: "\n") {
                        if let jsonData = line.data(using: .utf8) {
                            if let sample = try? decoder.decode(PreTokenizedSample.self, from: jsonData) {
                                preTokenizedData.append(sample)
                            }
                        }
                    }
                    if !preTokenizedData.isEmpty {
                        dataFormat = .preTokenized
                        print("Loaded \(preTokenizedData.count) pre-tokenized samples from \(fileName).jsonl (base model LM format)")
                        break
                    }
                } catch {
                    print("Failed to load \(fileName).jsonl: \(error)")
                }
            }
        }

        // Fallback to chat format if no pre-tokenized data found
        if preTokenizedData.isEmpty {
            for fileName in chatFormatFiles {
                if let fileURL = Bundle.main.url(forResource: fileName, withExtension: "jsonl") {
                    do {
                        let content = try String(contentsOf: fileURL, encoding: .utf8)
                        let decoder = JSONDecoder()
                        for line in content.split(separator: "\n") {
                            if let jsonData = line.data(using: .utf8) {
                                data.append(try decoder.decode([[String: String]].self, from: jsonData))
                            }
                        }
                        dataFormat = .chatFormat
                        print("Loaded \(data.count) chat-format samples from \(fileName).jsonl")
                        break
                    } catch {
                        print("Failed to load \(fileName).jsonl: \(error)")
                    }
                }
            }
        }

        if data.isEmpty && preTokenizedData.isEmpty {
            print("Warning: No training data found. Please add wikitext2_base.jsonl or wikitext2-demo.jsonl to the app bundle.")
        }

        self.data = data
        self.preTokenizedData = preTokenizedData
        self.dataFormat = dataFormat
    }
    
    /// Create a fresh runner to reset all training state
    mutating func recreateRunner(useInMemoryStorage: Bool = true) throws {
        print("Creating fresh runner to reset training state...")
        cleanup()

        // Capture modelDir to avoid escaping closure issue
        let trainableParamsURL = modelDir.appendingPathComponent("trainable_params.safetensors")

        switch trainingMode {
        case .mebp:
            let storageType: BaseRunner<MeBPContext>.StorageType = useInMemoryStorage ? .inMemory : .mmap()
            mebpRunner = try MeBPRunner(
                functionPaths: functionPaths,
                trainableParamsLoader: { try loadArrays(url: trainableParamsURL) },
                paramsLoaders: paramsLoaders,
                configs: configs,
                storageType: storageType
            )
            print("Fresh MeBPRunner created successfully")

        case .loraStructured:
            let storageType: BaseRunner<LoRAStructuredContext>.StorageType = useInMemoryStorage ? .inMemory : .mmap()
            structuredRunner = try LoRAStructuredRunner(
                functionPaths: functionPaths,
                trainableParamsLoader: { try loadArrays(url: trainableParamsURL) },
                paramsLoaders: paramsLoaders,
                configs: configs,
                storageType: storageType,
                numLayers: numLayers,
                nHeads: nHeads,
                nKvHeads: nKvHeads,
                headDim: headDim
            )
            structuredRunner?.enableSwiftBackward()
            print("Fresh LoRAStructuredRunner created successfully")
        }
    }

    func runWithProgress(
        learningRate: Float = 0.005,
        numSteps: Int = 20,
        epsilon: Float = 1e-3,  // For Hybrid ZO-FO
        sequenceLength: Int = 256,  // Must match export context_length
        debugMemory: Bool = false,  // Enable detailed memory logging
        onIterationStart: IterationStartHook? = nil,
        onIterationEnd: IterationEndHook? = nil
    ) async throws {
        // Prepare inputs based on data format
        let inputs: [[String: MLXArray]]

        switch dataFormat {
        case .preTokenized:
            guard !preTokenizedData.isEmpty else {
                throw MLXError.caught("No pre-tokenized data provided")
            }
            inputs = processPreTokenizedData(preTokenizedData, maxLength: sequenceLength)
            print("Using pre-tokenized data format (\(inputs.count) samples, seq_length=\(sequenceLength))")

        case .chatFormat:
            guard !data.isEmpty else {
                throw MLXError.caught("No chat-format data provided")
            }
            inputs = processCompletionData(data, tokenizer: tokenizer, maxLength: sequenceLength)
            print("Using chat-format data (\(inputs.count) samples, seq_length=\(sequenceLength))")
        }

        guard !inputs.isEmpty else {
            throw MLXError.caught("No valid training samples after processing")
        }

        let iterationStartHook: IterationStartHook = { step in
            try Task.checkCancellation()
            try onIterationStart?(step)
        }
        let iterationEndHook: IterationEndHook = { step, metrics in
            try Task.checkCancellation()
            try onIterationEnd?(step, metrics)
        }

        switch trainingMode {
        case .mebp:
            guard let runner = mebpRunner else {
                throw MLXError.caught("MeBPRunner not initialized")
            }
            _ = try await runner.run(
                batchedInputs: inputs,
                metricNames: ["loss"],
                learningRate: learningRate,
                numSteps: numSteps,
                numAccumulationSteps: 1,
                onIterationStart: iterationStartHook,
                onIterationEnd: iterationEndHook,
                verbose: false,
                debugMemory: debugMemory
            )

        case .loraStructured:
            guard let runner = structuredRunner else {
                throw MLXError.caught("LoRAStructuredRunner not initialized")
            }
            _ = try await runner.run(
                batchedInputs: inputs,
                metricNames: ["loss"],
                learningRate: learningRate,
                numSteps: numSteps,
                numAccumulationSteps: 1,
                onIterationStart: iterationStartHook,
                onIterationEnd: iterationEndHook,
                verbose: false,
                debugMemory: debugMemory
            )
        }
    }
    
    /// Get the loaded training data (chat format)
    func getData() -> [Messages] {
        return data
    }

    /// Get the number of training samples
    func getDataCount() -> Int {
        switch dataFormat {
        case .preTokenized:
            return preTokenizedData.count
        case .chatFormat:
            return data.count
        }
    }

    /// Get data format info string
    func getDataFormatInfo() -> String {
        switch dataFormat {
        case .preTokenized:
            return "Pre-tokenized (Base Model LM)"
        case .chatFormat:
            return "Chat Format (Instruction Model)"
        }
    }
    
    /// Cleanup method to release all resources explicitly
    func cleanup() {
        GPU.clearCache()
        if let runner = mebpRunner {
            runner.trainableParams.removeAll(keepingCapacity: false)
            runner.checkpoints.removeAll(keepingCapacity: false)
        }
        if let runner = structuredRunner {
            runner.trainableParams.removeAll(keepingCapacity: false)
            runner.checkpoints.removeAll(keepingCapacity: false)
        }
    }

    /// Get current trainable params for saving
    func getTrainableParams() -> [String: MLXArray]? {
        if let runner = mebpRunner {
            return runner.trainableParams
        }
        if let runner = structuredRunner {
            return runner.trainableParams
        }
        return nil
    }
}
