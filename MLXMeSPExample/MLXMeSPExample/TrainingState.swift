//
//  For licensing see accompanying LICENSE file.
//  Copyright (C) 2025 Apple Inc. All Rights Reserved.
//
//  TrainingState.swift
//  MLXMeSPExample
//
//  Created by Congzheng Song on 9/30/25.
//

internal import Combine
import Darwin
import Foundation
import MLX
import UIKit

struct MemoryUsage {
    let usedMemoryMB: Double
    let totalMemoryMB: Double
    let percentageUsed: Double
}

// MARK: - Thermal State Monitoring

struct ThermalInfo {
    let state: ProcessInfo.ThermalState
    let stateString: String
    let stateLevel: Int  // 0=nominal, 1=fair, 2=serious, 3=critical

    static func current() -> ThermalInfo {
        let state = ProcessInfo.processInfo.thermalState
        let (stateString, level) = switch state {
        case .nominal: ("Nominal", 0)
        case .fair: ("Fair", 1)
        case .serious: ("Serious", 2)
        case .critical: ("Critical", 3)
        @unknown default: ("Unknown", -1)
        }
        return ThermalInfo(state: state, stateString: stateString, stateLevel: level)
    }
}

// MARK: - Battery Monitoring

struct BatteryInfo {
    let level: Float  // 0.0 to 1.0 (-1 if unknown)
    let levelPercent: Int
    let state: UIDevice.BatteryState
    let stateString: String
    let isMonitoringEnabled: Bool

    static func current() -> BatteryInfo {
        let device = UIDevice.current
        let wasEnabled = device.isBatteryMonitoringEnabled

        // Enable monitoring if not already
        if !wasEnabled {
            device.isBatteryMonitoringEnabled = true
        }

        let level = device.batteryLevel
        let state = device.batteryState
        let stateString = switch state {
        case .unknown: "Unknown"
        case .unplugged: "Unplugged"
        case .charging: "Charging"
        case .full: "Full"
        @unknown default: "Unknown"
        }

        return BatteryInfo(
            level: level,
            levelPercent: level >= 0 ? Int(level * 100) : -1,
            state: state,
            stateString: stateString,
            isMonitoringEnabled: true
        )
    }

    static func enableMonitoring() {
        UIDevice.current.isBatteryMonitoringEnabled = true
    }
}

extension MemoryUsage {
    static func current() -> MemoryUsage {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout.size(ofValue: info) / MemoryLayout<Int32>.size)

        let kerr = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }

        if kerr == KERN_SUCCESS {
            let usedMemoryBytes = Double(info.phys_footprint)
            let usedMemoryMB = usedMemoryBytes / (1024 * 1024)

            // Get total system memory
            let totalMemoryBytes = Double(ProcessInfo.processInfo.physicalMemory)
            let totalMemoryMB = totalMemoryBytes / (1024 * 1024)

            let percentage = (usedMemoryMB / totalMemoryMB) * 100

            return MemoryUsage(usedMemoryMB: usedMemoryMB, totalMemoryMB: totalMemoryMB, percentageUsed: percentage)
        }

        return MemoryUsage(usedMemoryMB: 0, totalMemoryMB: 0, percentageUsed: 0)
    }

    /// Get current physical footprint in MB (snapshot at call time)
    static func currentFootprintMB() -> Double {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout.size(ofValue: info) / MemoryLayout<Int32>.size)

        let kerr = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }

        if kerr == KERN_SUCCESS {
            return Double(info.phys_footprint) / (1024 * 1024)
        }
        return 0
    }
}

/// Metrics for a single gradient step (for Performance Comparison experiment)
struct GradientStepMetrics: Codable {
    let step: Int
    let loss: Float
    let perplexity: Float          // exp(loss)
    let computeTimeSec: Double     // Per gradient step compute time in seconds
    let startMemoryMB: Double      // Memory at start of this step (phys_footprint snapshot)
    let endMemoryMB: Double        // Memory at end of this step (phys_footprint snapshot)
    let peakMemoryMB: Double       // Peak memory during this step (phys_footprint sampled at 0.5s intervals)
    let thermalState: String       // Thermal state: Nominal/Fair/Serious/Critical
    let thermalLevel: Int          // 0=nominal, 1=fair, 2=serious, 3=critical
    let batteryLevel: Int          // Battery percentage (0-100, -1 if unknown)
    let batteryState: String       // Battery state: Unplugged/Charging/Full/Unknown
    let timestamp: String
}

/// Legacy iteration stats for backward compatibility
struct IterationStats {
    let step: Int
    let loss: Float
    let perplexity: Float          // exp(loss)
    let iterationTime: Double
    let startMemoryMB: Double      // Memory at start of this step
    let endMemoryMB: Double        // Memory at end of this step
    let peakMemoryMB: Double       // Peak memory for this step
    let thermalState: String       // Thermal state
    let thermalLevel: Int          // Thermal level (0-3)
    let batteryLevel: Int          // Battery percentage
}

/// Experiment configuration matching paper settings
struct ExperimentConfig: Codable {
    var sequenceLength: Int = 256
    var numSamples: Int = 2048
    var batchSize: Int = 1
    var loraRank: Int = 8
    var learningRate: Float = 0.0001
    var numSteps: Int = 10

    static let paperDefaults = ExperimentConfig()
}

/// Dynamic model configuration detected from Documents directory
struct ModelConfiguration: Identifiable, Hashable {
    let directoryName: String
    let displayName: String
    let mode: String  // "mebp" (baseline) or "mesp" (ours)

    var id: String { directoryName }

    /// Parse model configuration from directory name
    /// Format: {model_name}-{dtype}[-rank{N}]-{mode}
    /// Example: Qwen2.5-0.5B-4bit-bfloat16-mebp, Qwen2.5-0.5B-4bit-bfloat16-rank16-mesp
    init?(directoryName: String) {
        self.directoryName = directoryName

        // Extract mode from the end of directory name
        if directoryName.hasSuffix("-mebp") {
            self.mode = "mebp"
            self.displayName = directoryName.replacingOccurrences(of: "-mebp", with: "") + " (MeBP)"
        } else if directoryName.hasSuffix("-mesp") {
            self.mode = "mesp"
            self.displayName = directoryName.replacingOccurrences(of: "-mesp", with: "") + " (MeSP)"
        } else {
            // Unknown format, still accept it
            self.mode = "unknown"
            self.displayName = directoryName
        }
    }

    /// Scan Documents directory for available model configurations
    static func scanAvailableModels() -> [ModelConfiguration] {
        let fm = FileManager.default
        guard let docs = fm.urls(for: .documentDirectory, in: .userDomainMask).first else {
            return []
        }

        var models: [ModelConfiguration] = []

        do {
            let contents = try fm.contentsOfDirectory(atPath: docs.path)
            for item in contents.sorted() {
                let itemPath = docs.appendingPathComponent(item).path
                var isDir: ObjCBool = false
                if fm.fileExists(atPath: itemPath, isDirectory: &isDir), isDir.boolValue {
                    // Check if it's a valid model directory (has run_configs.json)
                    let configPath = docs.appendingPathComponent(item).appendingPathComponent("run_configs.json").path
                    if fm.fileExists(atPath: configPath) {
                        if let config = ModelConfiguration(directoryName: item) {
                            models.append(config)
                        }
                    }
                }
            }
        } catch {
            print("Error scanning models: \(error)")
        }

        return models
    }
}

/// Complete experiment result for export
struct ExperimentResult: Codable {
    let experimentType: String     // "performance_comparison" or "utility_comparison"
    let method: String             // "mebp_structured" or "mebp_standard"
    let modelName: String
    let deviceName: String
    let config: ExperimentConfig

    // Per-step metrics
    let gradientSteps: [GradientStepMetrics]

    // Aggregated statistics
    let avgComputeTimeSec: Double
    let stdComputeTimeSec: Double
    let minComputeTimeSec: Double
    let maxComputeTimeSec: Double

    // Memory statistics (phys_footprint sampled at 0.5s intervals during each step)
    let avgStartMemoryMB: Double   // Average memory at step start
    let avgEndMemoryMB: Double     // Average memory at step end
    let avgPeakMemoryMB: Double    // Average peak memory across steps
    let maxPeakMemoryMB: Double    // Maximum peak memory across all steps
    let minPeakMemoryMB: Double    // Minimum peak memory across steps

    let initialLoss: Float
    let finalLoss: Float

    // Thermal statistics
    let maxThermalLevel: Int       // Maximum thermal level reached (0-3)
    let thermalThrottlingOccurred: Bool  // Did thermal reach serious/critical?
    let avgThermalLevel: Double

    // Battery statistics
    let startBatteryLevel: Int     // Battery at start (%)
    let endBatteryLevel: Int       // Battery at end (%)
    let batteryConsumed: Int       // Battery consumed during training (%)

    // Metadata
    let startTime: String
    let endTime: String
    let platform: String
    let memoryMeasurementMethod: String  // Description of memory measurement approach
}

/*
 Memory Measurement Method:

 Using phys_footprint from task_vm_info_data_t:
 - Samples physical memory (phys_footprint) every 0.5 seconds during gradient step
 - Tracks maximum value seen during step as peakMemoryMB
 - Also records startMemoryMB and endMemoryMB at step boundaries

 This follows the paper's described approach of using task_vm_info_data_t
 to measure peak memory footprint via phys_footprint sampling.
 */

class TrainingState: ObservableObject {
    static let shared = TrainingState()

    @Published var currentStep: Int = 0
    @Published var totalSteps: Int = 0
    @Published var iterationStats: [IterationStats] = []
    @Published var isTraining: Bool = false
    @Published var trainingCompleted: Bool = false
    @Published var canStartTraining: Bool = false
    @Published var trainingCancelled: Bool = false
    @Published var currentMemoryUsage: MemoryUsage = MemoryUsage.current()
    @Published var isModelInitialized: Bool = false

    // User configurable settings
    @Published var useInMemoryStorage: Bool = true
    @Published var learningRate: Float = 0.000001  // Low LR for stable benchmarking
    @Published var numberOfSteps: Int = 100
    @Published var showDataVisualization: Bool = false
    @Published var debugMemory: Bool = false  // Enable detailed memory logging (console)

    // Model selection for benchmarking (dynamic)
    @Published var availableModels: [ModelConfiguration] = []
    @Published var selectedModelConfiguration: ModelConfiguration? = nil

    // Experiment settings (fixed for paper)
    @Published var useExperimentMode: Bool = true  // Use paper settings
    @Published var experimentConfig: ExperimentConfig = .paperDefaults

    // Model's exported sequence length and batch size (from training_config.json)
    @Published var modelSeqLength: Int? = nil  // nil = not yet loaded
    @Published var modelBatchSize: Int? = nil  // nil = not yet loaded

    // Peak memory tracking for current step
    @Published var currentStepPeakMemoryMB: Double = 0
    @Published var overallPeakMemoryMB: Double = 0

    // Thermal and battery monitoring
    @Published var currentThermalInfo: ThermalInfo = ThermalInfo.current()
    @Published var currentBatteryInfo: BatteryInfo = BatteryInfo.current()
    @Published var maxThermalLevel: Int = 0

    // Export state
    @Published var lastExportedFilePath: String? = nil

    // Battery tracking for experiment
    private var startBatteryLevel: Int = -1

    var trainingTask: Task<Void, Error>?
    private var stepStartTime: DispatchTime?
    private var stepStartMemoryMB: Double = 0
    private var stepPeakMemoryMB: Double = 0
    private var memoryTimer: Timer?
    private var peakMemoryTimer: Timer?

    // For experiment result export
    private var gradientStepMetrics: [GradientStepMetrics] = []
    private var experimentStartTime: Date?

    private init() {
        BatteryInfo.enableMonitoring()
        startSystemMonitoring()
        scanForModels()
    }

    /// Scan Documents directory for available models
    func scanForModels() {
        let models = ModelConfiguration.scanAvailableModels()
        DispatchQueue.main.async {
            self.availableModels = models
            // Auto-select first model if none selected
            if self.selectedModelConfiguration == nil && !models.isEmpty {
                self.selectedModelConfiguration = models.first
            }
            print("Found \(models.count) model(s): \(models.map { $0.directoryName })")
        }
    }

    private func startSystemMonitoring() {
        // Monitor memory, thermal, and battery every second
        memoryTimer = Timer.scheduledTimer(withTimeInterval: 1.0, repeats: true) { _ in
            DispatchQueue.main.async {
                self.currentMemoryUsage = MemoryUsage.current()
                self.currentThermalInfo = ThermalInfo.current()
                self.currentBatteryInfo = BatteryInfo.current()

                // Track max thermal level during training
                if self.isTraining && self.currentThermalInfo.stateLevel > self.maxThermalLevel {
                    self.maxThermalLevel = self.currentThermalInfo.stateLevel
                }
            }
        }
    }

    deinit {
        memoryTimer?.invalidate()
        peakMemoryTimer?.invalidate()
    }

    /// Get system memory (phys_footprint) in MB
    private func systemMemoryMB() -> Double {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout.size(ofValue: info) / MemoryLayout<Int32>.size)

        let kerr = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }

        if kerr == KERN_SUCCESS {
            return Double(info.phys_footprint) / (1024 * 1024)
        }
        return 0
    }

    /// Start peak memory tracking for a gradient step (uses phys_footprint sampling)
    private func startPeakMemoryTracking() {
        // Use system memory (phys_footprint) for measurement
        stepStartMemoryMB = systemMemoryMB()
        stepPeakMemoryMB = stepStartMemoryMB

        // Sample phys_footprint every 0.5 seconds during gradient step
        peakMemoryTimer = Timer.scheduledTimer(withTimeInterval: 0.5, repeats: true) { [weak self] _ in
            guard let self = self else { return }
            let currentMemoryMB = self.systemMemoryMB()
            if currentMemoryMB > self.stepPeakMemoryMB {
                self.stepPeakMemoryMB = currentMemoryMB
            }
            if currentMemoryMB > self.overallPeakMemoryMB {
                DispatchQueue.main.async {
                    self.overallPeakMemoryMB = currentMemoryMB
                }
            }
        }
    }

    /// Stop peak memory tracking and return (startMemory, endMemory, peakMemory)
    private func stopPeakMemoryTracking() -> (start: Double, end: Double, peak: Double) {
        peakMemoryTimer?.invalidate()
        peakMemoryTimer = nil

        // Final measurement
        let endMemoryMB = systemMemoryMB()
        if endMemoryMB > stepPeakMemoryMB {
            stepPeakMemoryMB = endMemoryMB
        }

        // Update UI
        DispatchQueue.main.async {
            self.currentStepPeakMemoryMB = self.stepPeakMemoryMB
            if self.stepPeakMemoryMB > self.overallPeakMemoryMB {
                self.overallPeakMemoryMB = self.stepPeakMemoryMB
            }
        }

        return (start: stepStartMemoryMB, end: endMemoryMB, peak: stepPeakMemoryMB)
    }

    func onExperimentStart() {
        experimentStartTime = Date()
        gradientStepMetrics = []

        // Record starting battery level and reset max thermal
        let battery = BatteryInfo.current()
        startBatteryLevel = battery.levelPercent

        DispatchQueue.main.async {
            self.overallPeakMemoryMB = 0
            self.maxThermalLevel = 0
        }
    }

    func onIterationStart(step: Int) {
        DispatchQueue.main.async {
            self.currentStep = step
            self.isTraining = true
            self.trainingCancelled = false
        }
        stepStartTime = DispatchTime.now()
        startPeakMemoryTracking()
    }

    func onIterationEnd(step: Int, metrics: [String: MLXArray]) {
        guard let startTime = stepStartTime else { return }
        let elapsed = DispatchTime.now().uptimeNanoseconds - startTime.uptimeNanoseconds
        let elapsedMs = Double(elapsed) / 1_000_000  // Convert to milliseconds
        let elapsedSeconds = elapsedMs / 1000

        let memoryStats = stopPeakMemoryTracking()
        let loss = metrics["loss"]?.item(Float.self) ?? 0.0
        let perplexity = exp(loss)  // Perplexity = exp(cross-entropy loss)

        // Get current thermal and battery state
        let thermal = ThermalInfo.current()
        let battery = BatteryInfo.current()

        // Track max thermal level
        if thermal.stateLevel > maxThermalLevel {
            DispatchQueue.main.async {
                self.maxThermalLevel = thermal.stateLevel
            }
        }

        // Create legacy stats for UI
        let stats = IterationStats(
            step: step,
            loss: loss,
            perplexity: perplexity,
            iterationTime: elapsedSeconds,
            startMemoryMB: memoryStats.start,
            endMemoryMB: memoryStats.end,
            peakMemoryMB: memoryStats.peak,
            thermalState: thermal.stateString,
            thermalLevel: thermal.stateLevel,
            batteryLevel: battery.levelPercent
        )

        // Create detailed metrics for export
        let gradientMetrics = GradientStepMetrics(
            step: step,
            loss: loss,
            perplexity: perplexity,
            computeTimeSec: elapsedSeconds,
            startMemoryMB: memoryStats.start,
            endMemoryMB: memoryStats.end,
            peakMemoryMB: memoryStats.peak,
            thermalState: thermal.stateString,
            thermalLevel: thermal.stateLevel,
            batteryLevel: battery.levelPercent,
            batteryState: battery.stateString,
            timestamp: ISO8601DateFormatter().string(from: Date())
        )
        gradientStepMetrics.append(gradientMetrics)

        print("Step \(step) - Loss: \(String(format: "%.4f", loss)), PPL: \(String(format: "%.2f", perplexity)), Time: \(String(format: "%.2f", elapsedSeconds))s, Mem: \(String(format: "%.1f", memoryStats.start))->\(String(format: "%.1f", memoryStats.end))MB (Peak: \(String(format: "%.1f", memoryStats.peak))MB), Thermal: \(thermal.stateString), Battery: \(battery.levelPercent)%")

        DispatchQueue.main.async {
            self.iterationStats.append(stats)
            self.objectWillChange.send()

            if step == self.totalSteps - 1 {
                self.isTraining = false
                self.trainingCompleted = true
            }
        }
    }

    func cancelTraining() {
        trainingTask?.cancel()
        peakMemoryTimer?.invalidate()
        print("Task cancelled")
        DispatchQueue.main.async {
            self.isTraining = false
            self.trainingCancelled = true
        }
    }

    func reset() {
        trainingTask?.cancel()
        peakMemoryTimer?.invalidate()
        DispatchQueue.main.async {
            self.currentStep = 0
            self.iterationStats = []
            self.isTraining = false
            self.trainingCompleted = false
            self.trainingCancelled = false
            self.trainingTask = nil
            self.currentStepPeakMemoryMB = 0
            self.overallPeakMemoryMB = 0
            self.maxThermalLevel = 0
            self.lastExportedFilePath = nil
            // Note: modelSeqLength is NOT reset here - it stays fixed after model init
        }
        gradientStepMetrics = []
        experimentStartTime = nil
        startBatteryLevel = -1

        // Reset the MeBPRunner to fresh state
        Task {
            await MLXMeSPExampleApp.resetRunner()
        }
    }

    // MARK: - Export Functions

    /// Export experiment results to JSON file
    func exportResults(modelName: String = "Unknown", method: String = "mebp_structured") -> URL? {
        guard !gradientStepMetrics.isEmpty else {
            print("No metrics to export")
            return nil
        }

        let times = gradientStepMetrics.map { $0.computeTimeSec }
        let startMemories = gradientStepMetrics.map { $0.startMemoryMB }
        let endMemories = gradientStepMetrics.map { $0.endMemoryMB }
        let peakMemories = gradientStepMetrics.map { $0.peakMemoryMB }
        let thermalLevels = gradientStepMetrics.map { Double($0.thermalLevel) }

        let count = Double(gradientStepMetrics.count)
        let avgTime = times.reduce(0, +) / count
        let stdTime = sqrt(times.map { pow($0 - avgTime, 2) }.reduce(0, +) / count)

        // Memory statistics
        let avgStartMemory = startMemories.reduce(0, +) / count
        let avgEndMemory = endMemories.reduce(0, +) / count
        let avgPeakMemory = peakMemories.reduce(0, +) / count

        // Get current battery level for end state
        let endBattery = BatteryInfo.current()
        let batteryConsumed = startBatteryLevel >= 0 && endBattery.levelPercent >= 0
            ? startBatteryLevel - endBattery.levelPercent
            : 0

        let result = ExperimentResult(
            experimentType: "performance_comparison",
            method: method,
            modelName: modelName,
            deviceName: getDeviceName(),
            config: experimentConfig,
            gradientSteps: gradientStepMetrics,
            avgComputeTimeSec: avgTime,
            stdComputeTimeSec: stdTime,
            minComputeTimeSec: times.min() ?? 0,
            maxComputeTimeSec: times.max() ?? 0,
            avgStartMemoryMB: avgStartMemory,
            avgEndMemoryMB: avgEndMemory,
            avgPeakMemoryMB: avgPeakMemory,
            maxPeakMemoryMB: peakMemories.max() ?? 0,
            minPeakMemoryMB: peakMemories.min() ?? 0,
            initialLoss: gradientStepMetrics.first?.loss ?? 0,
            finalLoss: gradientStepMetrics.last?.loss ?? 0,
            maxThermalLevel: gradientStepMetrics.map { $0.thermalLevel }.max() ?? 0,
            thermalThrottlingOccurred: gradientStepMetrics.contains { $0.thermalLevel >= 2 },
            avgThermalLevel: thermalLevels.reduce(0, +) / count,
            startBatteryLevel: startBatteryLevel,
            endBatteryLevel: endBattery.levelPercent,
            batteryConsumed: batteryConsumed,
            startTime: experimentStartTime.map { ISO8601DateFormatter().string(from: $0) } ?? "",
            endTime: ISO8601DateFormatter().string(from: Date()),
            platform: "\(UIDevice.current.systemName) \(UIDevice.current.systemVersion)",
            memoryMeasurementMethod: "phys_footprint sampled at 0.5s intervals via task_vm_info_data_t"
        )

        do {
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            let jsonData = try encoder.encode(result)

            // Save to Documents directory
            let documentsPath = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            let timestamp = DateFormatter.localizedString(from: Date(), dateStyle: .short, timeStyle: .short)
                .replacingOccurrences(of: "/", with: "-")
                .replacingOccurrences(of: ":", with: "-")
                .replacingOccurrences(of: " ", with: "_")
            let fileName = "performance_\(method)_\(timestamp).json"
            let fileURL = documentsPath.appendingPathComponent(fileName)

            try jsonData.write(to: fileURL)

            DispatchQueue.main.async {
                self.lastExportedFilePath = fileURL.path
            }

            print("Exported results to: \(fileURL.path)")
            return fileURL
        } catch {
            print("Failed to export results: \(error)")
            return nil
        }
    }

    /// Export results as CSV
    func exportResultsCSV(modelName: String = "Unknown", method: String = "mebp_structured") -> URL? {
        guard !gradientStepMetrics.isEmpty else {
            print("No metrics to export")
            return nil
        }

        var csvContent = "step,loss,perplexity,compute_time_sec,start_memory_mb,end_memory_mb,peak_memory_mb,thermal_state,thermal_level,battery_level,battery_state,timestamp\n"
        for metrics in gradientStepMetrics {
            csvContent += "\(metrics.step),\(metrics.loss),\(metrics.perplexity),\(metrics.computeTimeSec),\(metrics.startMemoryMB),\(metrics.endMemoryMB),\(metrics.peakMemoryMB),\(metrics.thermalState),\(metrics.thermalLevel),\(metrics.batteryLevel),\(metrics.batteryState),\(metrics.timestamp)\n"
        }

        do {
            let documentsPath = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            let timestamp = DateFormatter.localizedString(from: Date(), dateStyle: .short, timeStyle: .short)
                .replacingOccurrences(of: "/", with: "-")
                .replacingOccurrences(of: ":", with: "-")
                .replacingOccurrences(of: " ", with: "_")
            let fileName = "performance_\(method)_\(timestamp).csv"
            let fileURL = documentsPath.appendingPathComponent(fileName)

            try csvContent.write(to: fileURL, atomically: true, encoding: .utf8)

            print("Exported CSV to: \(fileURL.path)")
            return fileURL
        } catch {
            print("Failed to export CSV: \(error)")
            return nil
        }
    }

    private func getDeviceName() -> String {
        var systemInfo = utsname()
        uname(&systemInfo)
        let machineMirror = Mirror(reflecting: systemInfo.machine)
        let identifier = machineMirror.children.reduce("") { identifier, element in
            guard let value = element.value as? Int8, value != 0 else { return identifier }
            return identifier + String(UnicodeScalar(UInt8(value)))
        }
        return identifier
    }
}
