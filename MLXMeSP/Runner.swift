//
//  For licensing see accompanying LICENSE file.
//  Copyright (C) 2025 Apple Inc. All Rights Reserved.
//
//  Runner.swift
//  mlx-mesp
//
//  Created by Congzheng Song on 9/18/25.
//

import Foundation
import MLX

#if XTOOL_MOBILE
internal typealias MeSPImportedFunction = XToolImportedFunction
#else
internal typealias MeSPImportedFunction = ImportedFunction
#endif

/// Protocol for checkpoint storage to allow swapping between in-memory and mmap implementations
public protocol MLXArrayDictionary {
    subscript(key: String) -> MLXArray? { get set }
    mutating func merge(_ other: [String: MLXArray], uniquingKeysWith combine: (MLXArray, MLXArray) throws -> MLXArray) rethrows
    var count: Int { get }
    var isEmpty: Bool { get }
    mutating func removeAll(keepingCapacity keepCapacity: Bool)
    mutating func removeValue(forKey: String) -> MLXArray?
}

extension MLXArrayDictionary {
    // Convenience method with default parameter
    mutating func removeAll() {
        removeAll(keepingCapacity: false)
    }
}

extension Dictionary: MLXArrayDictionary where Key == String, Value == MLXArray {
    // Dictionary's built-in removeAll(keepingCapacity:) satisfies the protocol requirement
}

extension MMapCheckpointStorage: MLXArrayDictionary {
    public func removeAll(keepingCapacity _: Bool = false) { cleanup() }

    public func removeValue(forKey: String) -> MLXArray? {
        removeArray(for: forKey)
        return nil
    }
}

public struct RunFunctionConfig: Codable {
    public let functionName: String
    public let inputNames: [String]
    public let outputNames: [String]
    public var checkpointsToRemove: [String]
    
    enum CodingKeys: String, CodingKey {
        case functionName = "function_name"
        case inputNames = "input_names"
        case outputNames = "output_names"
        case checkpointsToRemove = "checkpoints_to_remove"
    }
    
    public init(functionName: String, inputNames: [String], outputNames: [String], checkpointsToRemove: [String] = []) {
        self.functionName = functionName
        self.inputNames = inputNames
        self.outputNames = outputNames
        self.checkpointsToRemove = checkpointsToRemove
    }
}

// MARK: - MeBP Runner Implementation

public typealias IterationStartHook = (Int) throws -> Void
public typealias IterationEndHook = (Int, [String: MLXArray]) throws -> Void

public class BaseRunner<Context> {
    internal var functions: [String: MeSPImportedFunction] = [:]
    internal let paramsPaths: [String?]
    internal let configs: [RunFunctionConfig]
    public var checkpoints: MLXArrayDictionary

    public let trainableParamsLoader: () throws -> [String: MLXArray]
    public var trainableParams: [String: MLXArray]
    internal let paramsLoaders: [() throws -> [String: MLXArray]]
    
    public enum StorageType {
        case inMemory
        case mmap(baseDirectory: URL? = nil)
    }

    public init(
        functionPaths: [String],
        trainableParamsLoader: @escaping () throws -> [String: MLXArray],
        paramsLoaders: [() throws -> [String: MLXArray]],
        configs: [RunFunctionConfig],
        storageType: StorageType,
    ) throws {
        self.trainableParamsLoader = trainableParamsLoader
        self.trainableParams = try trainableParamsLoader()
        self.paramsLoaders = paramsLoaders
        self.configs = configs
        self.paramsPaths = Array(repeating: nil, count: configs.count)
        
        for (path, config) in zip(functionPaths, configs) {
            if functions[config.functionName] == nil {
                functions[config.functionName] = try MeSPImportedFunction(url: URL(fileURLWithPath: path))
            }
        }
        // Initialize checkpoint storage based on type
        switch storageType {
        case .inMemory:
            checkpoints = [:]
        case .mmap(let baseDirectory):
            checkpoints = try MMapCheckpointStorage(baseDirectory: baseDirectory)
        }
    }
    
    internal func runFunction(
        _ f: MeSPImportedFunction,
        inputs: MLXArrayDictionary,
        params: MLXArrayDictionary,
        config: RunFunctionConfig,
        eval: Bool = true,
        clearCache: Bool = true
    ) throws -> [String: MLXArray] {
        var args: [MLXArray] = []

        for inputName in config.inputNames {
            if let input = inputs[inputName] {
                args.append(input)
            } else if let param = params[inputName] {
                args.append(param)
            } else {
                throw MLXError.caught("Missing input \(inputName) for function \(config.functionName)")
            }
        }

        var outputs = try f.call(args: args, kwargs: [:])

        guard outputs.count == config.outputNames.count else {
            throw MLXError.caught("Output count mismatch: expected \(config.outputNames.count), got \(outputs.count)")
        }

        // Workaround: Convert embedding output to match model dtype if needed
        if config.functionName == "embedding" {
            var expectedDtype: DType? = nil
            for inputName in config.inputNames {
                if let param = params[inputName], param.dtype == .bfloat16 || param.dtype == .float16 {
                    expectedDtype = param.dtype
                    break
                }
            }

            if let targetDtype = expectedDtype {
                outputs = outputs.map { output in
                    let elementCount = output.shape.reduce(1, *)
                    let dataSize = output.asData().data.count
                    let bytesPerElement = dataSize / elementCount
                    let needsConversion = output.dtype != targetDtype || bytesPerElement != 2

                    if needsConversion {
                        let converted = output.asType(targetDtype)
                        converted.eval()
                        return converted
                    }
                    return output
                }
            }
        }

        // Workaround: Convert transformer_block outputs to bfloat16 if needed
        if config.functionName == "transformer_block_forward" || config.functionName == "transformer_block_backward" {
            var expectedDtype: DType? = nil
            for inputName in config.inputNames {
                if let param = params[inputName], param.dtype == .bfloat16 || param.dtype == .float16 {
                    expectedDtype = param.dtype
                    break
                }
            }

            if let targetDtype = expectedDtype {
                outputs = outputs.map { output in
                    output.eval()
                    let elementCount = output.shape.reduce(1, *)
                    let dataSize = output.asData().data.count
                    let bytesPerElement = elementCount > 0 ? dataSize / elementCount : 0

                    if bytesPerElement == 4 && (targetDtype == .bfloat16 || targetDtype == .float16) {
                        let converted = output.asType(targetDtype)
                        converted.eval()
                        return converted
                    }
                    return output
                }
            }
        }

        // Eval the function outputs each time to avoid loading all unread arrays in memory
        if eval {
            outputs.forEach { $0.eval() }
        }
        // NOTE: clearCache moved to end of step() for accurate peak memory measurement
        // if clearCache { GPU.clearCache() }

        return Dictionary(uniqueKeysWithValues: zip(config.outputNames, outputs))
    }
    
    internal func step(inputs: [String: MLXArray], eval: Bool = true, debugMemory: Bool = false) async throws {
        checkpoints.removeAll()
        inputs.forEach { checkpoints[$0.key] = $0.value }

        // Memory debug helper
        func getMemoryMB() -> Double {
            return Double(GPU.activeMemory) / (1024 * 1024)
        }

        if debugMemory {
            GPU.resetPeakMemory()
            print("\n=== STEP START (checkpoints: \(checkpoints.count), mem: \(String(format: "%.0f", getMemoryMB()))MB) ===")
        }

        for (i, config) in configs.enumerated() {
            try Task.checkCancellation()
            let paramsLoader = paramsLoaders[i]
            guard let f = functions[config.functionName] else {
                throw MLXError.caught("Function runner not found for \(config.functionName)")
            }

            var params = try paramsLoader()
            params.merge(trainableParams) { (_, new) in new }

            let outputs = try runFunction(f, inputs: checkpoints, params: params, config: config, eval: eval)
            checkpoints.merge(outputs) { _, new in new }

            // Remove checkpoints as specified in config
            let removedCount = config.checkpointsToRemove.count
            for checkpoint in config.checkpointsToRemove {
                _ = checkpoints.removeValue(forKey: checkpoint)
            }

            if debugMemory {
                let phase = config.functionName.contains("backward") ? "BWD" : "FWD"
                let removal = removedCount > 0 ? "removed:\(removedCount)" : "kept"
                print("[\(String(format: "%02d", i))] \(phase) \(removal) → ckpts:\(checkpoints.count) mem:\(String(format: "%.0f", getMemoryMB()))MB")
            }
        }

        if debugMemory {
            let peakMB = Double(GPU.peakMemory) / (1024 * 1024)
            print("=== STEP END (peak: \(String(format: "%.0f", peakMB))MB) ===\n")
        }
    }
    
    internal func accumulate(accumulatedGrads: inout [String: MLXArray], grads: [String: MLXArray]) {
        for (name, grad) in grads {
            if let currentGrad = accumulatedGrads[name] {
                accumulatedGrads[name] = currentGrad + grad
            } else {
                accumulatedGrads[name] = grad
            }
        }
    }
    
    internal func sgdStep(accumulatedGrads: [String: MLXArray], learningRate: Float, numAccumulationSteps: Int) {
        let lr = MLXArray(learningRate / Float(numAccumulationSteps))
        
        for (name, grad) in accumulatedGrads {
            if let param = trainableParams[name] {
                trainableParams[name] = param - lr.asType(grad.dtype) * grad
            }
        }
    }
    
    internal func getGradName(_ name: String) -> String {
        return "\(name).grad"
    }
    
    internal func getMetrics(metricsName: [String]) -> [String: MLXArray] {
        var metrics = [String: MLXArray]()
        for name in metricsName {
            if let metric = checkpoints[name] {
                metrics[name] = metric
            }
        }
        return metrics
    }
    
    public func gradients(
        inputs: [String: MLXArray],
        metricNames: [String],
        context: Context,
    ) async throws -> (grads: [String: MLXArray], metrics: [String: MLXArray]) {
        throw MLXError.caught("Not implemented.")
    }

    public func run(
        batchedInputs: [[String: MLXArray]],
        metricNames: [String],
        learningRate: Float,
        numSteps: Int,
        numAccumulationSteps: Int,
        gradientsContext: Context,
        onIterationStart: IterationStartHook? = nil,
        onIterationEnd: IterationEndHook? = nil,
        verbose: Bool = false,
    ) async throws -> [[String: MLXArray]] {
        var accumulatedGrads: [String: MLXArray] = [:]
        var metricsList: [[String: MLXArray]] = []
        
        for i in 0..<numSteps {
            // Check for task cancellation and throw to properly exit
            try Task.checkCancellation()
            try onIterationStart?(i)
            let inputIdx = i % batchedInputs.count
            let (grads, metrics) = try await gradients(inputs: batchedInputs[inputIdx], metricNames: metricNames, context: gradientsContext)
            metricsList.append(metrics)
            // Accumulate gradients
            accumulate(accumulatedGrads: &accumulatedGrads, grads: grads)
            if (i + 1) % numAccumulationSteps == 0 {
                // Optimize
                sgdStep(accumulatedGrads: accumulatedGrads, learningRate: learningRate, numAccumulationSteps: numAccumulationSteps)
                // Reset accumulated gradients
                accumulatedGrads = [:]
            }
            try onIterationEnd?(i, metrics)
        }
        return metricsList
    }

}

public struct MeBPContext {
    public var debugMemory: Bool = false

    public init(debugMemory: Bool = false) {
        self.debugMemory = debugMemory
    }
}

public class MeBPRunner: BaseRunner<MeBPContext> {
    override public func gradients(
        inputs: [String: MLXArray],
        metricNames: [String],
        context: MeBPContext
    ) async throws -> (grads: [String: MLXArray], metrics: [String: MLXArray]){
        try await step(inputs: inputs, debugMemory: context.debugMemory)
        let metrics = getMetrics(metricsName: metricNames)
        var grads: [String: MLXArray] = [:]
        for name in trainableParams.keys {
            if let grad = checkpoints[getGradName(name)] {
                grads[name] = grad
            }
        }
        checkpoints.removeAll()
        return (grads: grads, metrics: metrics)
    }

    public func run(
        batchedInputs: [[String: MLXArray]],
        metricNames: [String],
        learningRate: Float,
        numSteps: Int,
        numAccumulationSteps: Int,
        onIterationStart: IterationStartHook? = nil,
        onIterationEnd: IterationEndHook? = nil,
        verbose: Bool = false,
        debugMemory: Bool = true,
    ) async throws -> [[String: MLXArray]] {
        try await run(
            batchedInputs: batchedInputs,
            metricNames: metricNames,
            learningRate: learningRate,
            numSteps: numSteps,
            numAccumulationSteps: numAccumulationSteps,
            gradientsContext: MeBPContext(debugMemory: debugMemory),
            onIterationStart: onIterationStart,
            onIterationEnd: onIterationEnd,
            verbose: verbose
        )
    }
}

/// Training metadata loaded from `training_config.json` produced by the
/// Python export pipeline. Optional fields keep older exports loadable.
public struct TrainingConfig: Codable {
    public let numLayers: Int
    public let mode: String
    public let swiftStructuredBackward: Bool
    public let loraRank: Int
    public let seqLength: Int
    public let batchSize: Int

    enum CodingKeys: String, CodingKey {
        case numLayers = "num_layers"
        case mode
        case swiftStructuredBackward = "swift_structured_backward"
        case loraRank = "lora_rank"
        case seqLength = "seq_length"
        case batchSize = "batch_size"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.numLayers = try c.decode(Int.self, forKey: .numLayers)
        self.mode = (try? c.decode(String.self, forKey: .mode)) ?? "mebp"
        self.swiftStructuredBackward =
            (try? c.decode(Bool.self, forKey: .swiftStructuredBackward)) ?? false
        self.loraRank = (try? c.decode(Int.self, forKey: .loraRank)) ?? 8
        self.seqLength = (try? c.decode(Int.self, forKey: .seqLength)) ?? 256
        self.batchSize = (try? c.decode(Int.self, forKey: .batchSize)) ?? 1
    }
}
