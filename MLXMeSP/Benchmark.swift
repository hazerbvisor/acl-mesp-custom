//
//  For licensing see accompanying LICENSE file.
//  Copyright (C) 2025 Apple Inc. All Rights Reserved.
//
//  Benchmark.swift
//  mlx-mesp
//
//  Benchmark utilities for measuring training metrics as reported in the MeBP paper.
//  Metrics: Peak Memory, Training Loss, Latency, Throughput
//

import Foundation
import MLX
#if os(iOS)
import UIKit
#endif

// MARK: - Benchmark Result Data Structures

/// Single step benchmark measurement
public struct StepBenchmark: Codable {
    public let step: Int
    public let loss: Float
    public let latencyMs: Double
    public let physFootprintMB: Double      // Paper's metric: task_vm_info_data_t.phys_footprint
    public let mlxPeakMemoryMB: Double      // MLX GPU peak memory
    public let tokensPerSecond: Double
    public let timestamp: Date

    public init(step: Int, loss: Float, latencyMs: Double, physFootprintMB: Double, mlxPeakMemoryMB: Double, tokensPerSecond: Double) {
        self.step = step
        self.loss = loss
        self.latencyMs = latencyMs
        self.physFootprintMB = physFootprintMB
        self.mlxPeakMemoryMB = mlxPeakMemoryMB
        self.tokensPerSecond = tokensPerSecond
        self.timestamp = Date()
    }
}

/// Aggregated benchmark result from multiple runs (paper reports average of 10 runs)
public struct AggregatedBenchmark: Codable {
    public let numRuns: Int
    public let averageLatencyMs: Double
    public let stddevLatencyMs: Double
    public let averagePeakPhysFootprintMB: Double
    public let stddevPeakPhysFootprintMB: Double
    public let averageThroughput: Double

    public init(runs: [BenchmarkResult]) {
        self.numRuns = runs.count

        let latencies = runs.map { $0.averageLatencyMs }
        self.averageLatencyMs = latencies.reduce(0, +) / Double(runs.count)
        self.stddevLatencyMs = Self.stddev(latencies)

        let memories = runs.map { $0.peakMemoryMB }
        self.averagePeakPhysFootprintMB = memories.reduce(0, +) / Double(runs.count)
        self.stddevPeakPhysFootprintMB = Self.stddev(memories)

        let throughputs = runs.map { $0.averageThroughput }
        self.averageThroughput = throughputs.reduce(0, +) / Double(runs.count)
    }

    private static func stddev(_ values: [Double]) -> Double {
        let mean = values.reduce(0, +) / Double(values.count)
        let squaredDiffs = values.map { ($0 - mean) * ($0 - mean) }
        return sqrt(squaredDiffs.reduce(0, +) / Double(values.count))
    }
}

/// Complete benchmark run configuration and results
public struct BenchmarkResult: Codable {
    // Configuration
    public let modelName: String
    public let deviceName: String
    public let batchSize: Int
    public let sequenceLength: Int
    public let numSteps: Int
    public let learningRate: Float
    public let numAccumulationSteps: Int
    public let useStructuredBackward: Bool

    // Aggregated metrics
    public let totalTrainingTimeMs: Double
    public let averageLatencyMs: Double
    public let averageThroughput: Double
    public let peakMemoryMB: Double             // Peak phys_footprint (paper's metric)
    public let peakMLXMemoryMB: Double          // Peak MLX GPU memory
    public let finalLoss: Float
    public let initialLoss: Float

    // Per-step data
    public let steps: [StepBenchmark]

    // Metadata
    public let startTime: Date
    public let endTime: Date
    public let osVersion: String
    public let mlxVersion: String

    public init(
        modelName: String,
        deviceName: String,
        batchSize: Int,
        sequenceLength: Int,
        numSteps: Int,
        learningRate: Float,
        numAccumulationSteps: Int,
        useStructuredBackward: Bool,
        steps: [StepBenchmark],
        startTime: Date,
        endTime: Date
    ) {
        self.modelName = modelName
        self.deviceName = deviceName
        self.batchSize = batchSize
        self.sequenceLength = sequenceLength
        self.numSteps = numSteps
        self.learningRate = learningRate
        self.numAccumulationSteps = numAccumulationSteps
        self.useStructuredBackward = useStructuredBackward
        self.steps = steps
        self.startTime = startTime
        self.endTime = endTime

        // Calculate aggregated metrics
        self.totalTrainingTimeMs = endTime.timeIntervalSince(startTime) * 1000
        self.averageLatencyMs = steps.isEmpty ? 0 : steps.map { $0.latencyMs }.reduce(0, +) / Double(steps.count)
        self.averageThroughput = steps.isEmpty ? 0 : steps.map { $0.tokensPerSecond }.reduce(0, +) / Double(steps.count)
        // Paper's metric: phys_footprint
        self.peakMemoryMB = steps.isEmpty ? 0 : steps.map { $0.physFootprintMB }.max() ?? 0
        self.peakMLXMemoryMB = steps.isEmpty ? 0 : steps.map { $0.mlxPeakMemoryMB }.max() ?? 0
        self.finalLoss = steps.last?.loss ?? 0
        self.initialLoss = steps.first?.loss ?? 0

        // System info
        self.osVersion = ProcessInfo.processInfo.operatingSystemVersionString
        self.mlxVersion = "0.26.3"  // Update as needed
    }
}

// MARK: - Memory Measurement

public class MemoryMonitor {
    private var peakPhysFootprint: UInt64 = 0

    public init() {}

    /// Get physical footprint memory usage in bytes
    /// This is the metric used by iOS Jetsam and reported in the MeBP paper
    /// Uses task_vm_info_data_t.phys_footprint as specified in the paper
    public func getPhysicalFootprint() -> UInt64 {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)

        let result = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }

        if result == KERN_SUCCESS {
            let footprint = info.phys_footprint
            peakPhysFootprint = max(peakPhysFootprint, footprint)
            return footprint
        }
        return 0
    }

    /// Get physical footprint in MB (paper's metric)
    public func getPhysicalFootprintMB() -> Double {
        return Double(getPhysicalFootprint()) / 1024.0 / 1024.0
    }

    /// Get peak physical footprint in MB
    public func getPeakPhysicalFootprintMB() -> Double {
        return Double(peakPhysFootprint) / 1024.0 / 1024.0
    }

    /// Reset peak memory tracking
    public func resetPeak() {
        peakPhysFootprint = getPhysicalFootprint()
    }

    /// Get resident size (alternative metric)
    public func getResidentSize() -> UInt64 {
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size) / 4

        let result = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count)
            }
        }

        return result == KERN_SUCCESS ? info.resident_size : 0
    }

    /// Get resident size in MB
    public func getResidentSizeMB() -> Double {
        return Double(getResidentSize()) / 1024.0 / 1024.0
    }

    /// Get GPU memory usage (MLX specific)
    public func getGPUMemoryMB() -> Double {
        let activeMemory = GPU.activeMemory
        let cacheMemory = GPU.cacheMemory
        return Double(activeMemory + cacheMemory) / 1024.0 / 1024.0
    }

    /// Get MLX active memory only
    public func getMLXActiveMemoryMB() -> Double {
        return Double(GPU.activeMemory) / 1024.0 / 1024.0
    }

    /// Get MLX peak memory
    public func getMLXPeakMemoryMB() -> Double {
        return Double(GPU.peakMemory) / 1024.0 / 1024.0
    }
}

// MARK: - Benchmark Runner

public class BenchmarkRunner {
    private let memoryMonitor = MemoryMonitor()
    public private(set) var stepResults: [StepBenchmark] = []
    private var startTime: Date?
    private var stepStartTime: Date?

    // Configuration
    public let modelName: String
    public let batchSize: Int
    public let sequenceLength: Int
    public let useStructuredBackward: Bool

    public init(
        modelName: String,
        batchSize: Int,
        sequenceLength: Int,
        useStructuredBackward: Bool = true
    ) {
        self.modelName = modelName
        self.batchSize = batchSize
        self.sequenceLength = sequenceLength
        self.useStructuredBackward = useStructuredBackward
    }

    /// Call before starting training
    public func startBenchmark() {
        startTime = Date()
        stepResults.removeAll()
        memoryMonitor.resetPeak()
        GPU.resetPeakMemory()
    }

    /// Call before each training step
    public func startStep() {
        stepStartTime = Date()
        // Reset peak memory tracking for this step
        GPU.resetPeakMemory()
        memoryMonitor.resetPeak()
    }

    /// Call after each training step (after forward + backward + optimizer)
    /// Measures using paper's methodology:
    /// - Latency: wall-clock time for the step
    /// - Memory: task_vm_info_data_t.phys_footprint (paper's metric)
    public func endStep(step: Int, loss: Float) {
        guard let stepStart = stepStartTime else { return }

        // Ensure all GPU operations complete before measuring
        // MLX automatically synchronizes when reading values, no explicit sync needed

        let latencyMs = Date().timeIntervalSince(stepStart) * 1000

        // Paper's metric: phys_footprint from task_vm_info_data_t
        let physFootprintMB = memoryMonitor.getPeakPhysicalFootprintMB()

        // Also record MLX GPU peak memory for comparison
        let mlxPeakMemoryMB = Double(GPU.peakMemory) / 1024.0 / 1024.0

        let tokensProcessed = batchSize * sequenceLength
        let tokensPerSecond = Double(tokensProcessed) / (latencyMs / 1000.0)

        let benchmark = StepBenchmark(
            step: step,
            loss: loss,
            latencyMs: latencyMs,
            physFootprintMB: physFootprintMB,
            mlxPeakMemoryMB: mlxPeakMemoryMB,
            tokensPerSecond: tokensPerSecond
        )
        stepResults.append(benchmark)
    }

    /// Generate final benchmark result
    public func finishBenchmark(
        numSteps: Int,
        learningRate: Float,
        numAccumulationSteps: Int
    ) -> BenchmarkResult {
        let endTime = Date()

        return BenchmarkResult(
            modelName: modelName,
            deviceName: getDeviceName(),
            batchSize: batchSize,
            sequenceLength: sequenceLength,
            numSteps: numSteps,
            learningRate: learningRate,
            numAccumulationSteps: numAccumulationSteps,
            useStructuredBackward: useStructuredBackward,
            steps: stepResults,
            startTime: startTime ?? endTime,
            endTime: endTime
        )
    }

    /// Get device name
    private func getDeviceName() -> String {
        #if os(iOS)
        return UIDevice.current.model + " " + UIDevice.current.systemVersion
        #elseif os(macOS)
        var size = 0
        sysctlbyname("hw.model", nil, &size, nil, 0)
        var model = [CChar](repeating: 0, count: size)
        sysctlbyname("hw.model", &model, &size, nil, 0)
        return String(cString: model)
        #else
        return "Unknown Device"
        #endif
    }
}

// MARK: - Result Export

public class BenchmarkExporter {

    /// Save benchmark result as JSON
    public static func saveAsJSON(_ result: BenchmarkResult, to url: URL) throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .iso8601

        let data = try encoder.encode(result)
        try data.write(to: url)
    }

    /// Save benchmark result as CSV
    public static func saveAsCSV(_ result: BenchmarkResult, to url: URL) throws {
        var csv = "step,loss,latency_ms,memory_mb,tokens_per_second,timestamp\n"

        let dateFormatter = ISO8601DateFormatter()
        for step in result.steps {
            csv += "\(step.step),\(step.loss),\(step.latencyMs),\(step.physFootprintMB),\(step.tokensPerSecond),\(dateFormatter.string(from: step.timestamp))\n"
        }

        try csv.write(to: url, atomically: true, encoding: .utf8)
    }

    /// Save summary as CSV (one row per benchmark run)
    public static func saveSummaryAsCSV(_ results: [BenchmarkResult], to url: URL) throws {
        var csv = "model_name,device_name,batch_size,sequence_length,num_steps,learning_rate,"
        csv += "use_structured_backward,total_time_ms,avg_latency_ms,avg_throughput,peak_memory_mb,"
        csv += "initial_loss,final_loss,os_version,start_time,end_time\n"

        let dateFormatter = ISO8601DateFormatter()
        for result in results {
            csv += "\(result.modelName),\(result.deviceName),\(result.batchSize),\(result.sequenceLength),"
            csv += "\(result.numSteps),\(result.learningRate),\(result.useStructuredBackward),"
            csv += "\(result.totalTrainingTimeMs),\(result.averageLatencyMs),\(result.averageThroughput),"
            csv += "\(result.peakMemoryMB),\(result.initialLoss),\(result.finalLoss),"
            csv += "\(result.osVersion),\(dateFormatter.string(from: result.startTime)),"
            csv += "\(dateFormatter.string(from: result.endTime))\n"
        }

        try csv.write(to: url, atomically: true, encoding: .utf8)
    }

    /// Load benchmark result from JSON
    public static func loadFromJSON(url: URL) throws -> BenchmarkResult {
        let data = try Data(contentsOf: url)
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return try decoder.decode(BenchmarkResult.self, from: data)
    }
}

// MARK: - Benchmark-enabled Runner Extensions

public extension MeBPRunner {

    /// Run training with benchmark measurement
    func runWithBenchmark(
        batchedInputs: [[String: MLXArray]],
        metricNames: [String],
        learningRate: Float,
        numSteps: Int,
        numAccumulationSteps: Int,
        benchmarkRunner: BenchmarkRunner,
        onIterationStart: IterationStartHook? = nil,
        onIterationEnd: IterationEndHook? = nil,
        verbose: Bool = false
    ) async throws -> (metrics: [[String: MLXArray]], benchmark: BenchmarkResult) {

        benchmarkRunner.startBenchmark()

        var accumulatedGrads: [String: MLXArray] = [:]
        var metricsList: [[String: MLXArray]] = []

        for i in 0..<numSteps {
            try Task.checkCancellation()
            try onIterationStart?(i)

            benchmarkRunner.startStep()

            let inputIdx = i % batchedInputs.count
            let (grads, metrics) = try await gradients(
                inputs: batchedInputs[inputIdx],
                metricNames: metricNames,
                context: .init()
            )

            // Extract loss for benchmark
            let loss = metrics["loss"]?.item(Float.self) ?? 0
            benchmarkRunner.endStep(step: i, loss: loss)

            metricsList.append(metrics)
            accumulate(accumulatedGrads: &accumulatedGrads, grads: grads)

            if (i + 1) % numAccumulationSteps == 0 {
                sgdStep(accumulatedGrads: accumulatedGrads, learningRate: learningRate, numAccumulationSteps: numAccumulationSteps)
                accumulatedGrads = [:]
            }

            try onIterationEnd?(i, metrics)

            if verbose {
                print("Step \(i): loss=\(loss), latency=\(benchmarkRunner.stepResults.last?.latencyMs ?? 0)ms")
            }
        }

        let benchmarkResult = benchmarkRunner.finishBenchmark(
            numSteps: numSteps,
            learningRate: learningRate,
            numAccumulationSteps: numAccumulationSteps
        )

        return (metrics: metricsList, benchmark: benchmarkResult)
    }

    /// Run benchmark multiple times and report average (paper methodology: 10 runs)
    /// This matches the paper: "We repeat the training process 10 times and report the average runtime and peak memory usage"
    func runBenchmarkWithRepetitions(
        batchedInputs: [[String: MLXArray]],
        metricNames: [String],
        learningRate: Float,
        numSteps: Int,
        numAccumulationSteps: Int,
        benchmarkRunner: BenchmarkRunner,
        numRepetitions: Int = 10,
        verbose: Bool = false
    ) async throws -> (runs: [BenchmarkResult], aggregated: AggregatedBenchmark) {

        var allRuns: [BenchmarkResult] = []

        for run in 0..<numRepetitions {
            if verbose {
                print("=== Benchmark Run \(run + 1)/\(numRepetitions) ===")
            }

            // Reset trainable params to initial state for fair comparison
            trainableParams = try trainableParamsLoader()

            let (_, benchmark) = try await runWithBenchmark(
                batchedInputs: batchedInputs,
                metricNames: metricNames,
                learningRate: learningRate,
                numSteps: numSteps,
                numAccumulationSteps: numAccumulationSteps,
                benchmarkRunner: benchmarkRunner,
                verbose: verbose
            )

            allRuns.append(benchmark)

            if verbose {
                print("Run \(run + 1): latency=\(benchmark.averageLatencyMs)ms, peak_memory=\(benchmark.peakMemoryMB)MB")
            }

            // Clear GPU cache between runs
            GPU.clearCache()
        }

        let aggregated = AggregatedBenchmark(runs: allRuns)

        if verbose {
            print("\n=== Aggregated Results (\(numRepetitions) runs) ===")
            print("Average Latency: \(aggregated.averageLatencyMs) ± \(aggregated.stddevLatencyMs) ms")
            print("Average Peak Memory (phys_footprint): \(aggregated.averagePeakPhysFootprintMB) ± \(aggregated.stddevPeakPhysFootprintMB) MB")
            print("Average Throughput: \(aggregated.averageThroughput) tokens/sec")
        }

        return (runs: allRuns, aggregated: aggregated)
    }
}

// MARK: - Paper-compliant Benchmark Configuration

/// Default benchmark configuration matching the paper's methodology
public struct PaperBenchmarkConfig {
    public static let batchSize = 1
    public static let sequenceLength = 256
    public static let numRepetitions = 10

    /// Create a benchmark runner with paper's configuration
    public static func createRunner(
        modelName: String,
        useStructuredBackward: Bool = true
    ) -> BenchmarkRunner {
        return BenchmarkRunner(
            modelName: modelName,
            batchSize: batchSize,
            sequenceLength: sequenceLength,
            useStructuredBackward: useStructuredBackward
        )
    }
}

