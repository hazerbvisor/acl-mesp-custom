//
//  For licensing see accompanying LICENSE file.
//  Copyright (C) 2025 Apple Inc. All Rights Reserved.
//
//  ContentView.swift
//  MLXMeSPExample
//
//  Created by Congzheng Song on 9/18/25.
//

import SwiftUI
import MLX

struct ContentView: View {
    @ObservedObject private var trainingState = TrainingState.shared
    @State private var showExportSheet = false
    @State private var exportedFileURL: URL? = nil

    var body: some View {
        ScrollView {
            VStack(spacing: 20) {
                Text("MLX MeBP Training")
                    .font(.largeTitle)
                    .fontWeight(.bold)

                // Live Memory Usage Display
                memoryUsageSection

                // Thermal and Battery Status
                thermalBatterySection

                // Experiment Settings
                experimentSettingsSection

                // Training Settings
                trainingSettingsSection

                // Model Initialization or Training Section
                trainingControlSection
            }
            .padding()
        }
        .sheet(isPresented: $showExportSheet) {
            if let url = exportedFileURL {
                ShareSheet(activityItems: [url])
            }
        }
    }

    // MARK: - Memory Usage Section

    private var memoryUsageSection: some View {
        VStack(spacing: 8) {
            HStack {
                Text("Memory Usage")
                    .font(.headline)
                    .fontWeight(.semibold)
                Spacer()
                Text("\(String(format: "%.1f", trainingState.currentMemoryUsage.usedMemoryMB)) MB")
                    .font(.system(.body, design: .monospaced))
                    .fontWeight(.medium)
            }

            // Memory usage progress bar
            ProgressView(value: trainingState.currentMemoryUsage.percentageUsed, total: 100.0)
                .progressViewStyle(LinearProgressViewStyle(tint: memoryBarColor))

            HStack {
                Text("\(String(format: "%.1f", trainingState.currentMemoryUsage.percentageUsed))% of system memory")
                    .font(.caption)
                    .foregroundColor(.secondary)
                Spacer()
                Text("Total: \(String(format: "%.0f", trainingState.currentMemoryUsage.totalMemoryMB)) MB")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }

            // Peak memory display
            if trainingState.overallPeakMemoryMB > 0 {
                HStack {
                    Text("Peak Memory:")
                        .font(.caption)
                        .foregroundColor(.orange)
                    Spacer()
                    Text("\(String(format: "%.1f", trainingState.overallPeakMemoryMB)) MB")
                        .font(.system(.caption, design: .monospaced))
                        .fontWeight(.semibold)
                        .foregroundColor(.orange)
                }
            }
        }
        .padding()
        .background(Color(.systemGray6))
        .cornerRadius(10)
    }

    // MARK: - Thermal and Battery Section

    private var thermalBatterySection: some View {
        VStack(spacing: 8) {
            HStack {
                Text("Device Status")
                    .font(.headline)
                    .fontWeight(.semibold)
                Spacer()
            }

            HStack(spacing: 20) {
                // Thermal State
                VStack(alignment: .leading, spacing: 4) {
                    HStack {
                        Image(systemName: "thermometer.medium")
                            .foregroundColor(thermalColor)
                        Text("Thermal")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                    Text(trainingState.currentThermalInfo.stateString)
                        .font(.system(.body, design: .monospaced))
                        .fontWeight(.semibold)
                        .foregroundColor(thermalColor)
                }

                Spacer()

                // Battery Level
                VStack(alignment: .trailing, spacing: 4) {
                    HStack {
                        Text("Battery")
                            .font(.caption)
                            .foregroundColor(.secondary)
                        Image(systemName: batteryIcon)
                            .foregroundColor(batteryColor)
                    }
                    HStack(spacing: 4) {
                        Text("\(trainingState.currentBatteryInfo.levelPercent)%")
                            .font(.system(.body, design: .monospaced))
                            .fontWeight(.semibold)
                            .foregroundColor(batteryColor)
                        Text("(\(trainingState.currentBatteryInfo.stateString))")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                }
            }

            // Max thermal during training
            if trainingState.isTraining || trainingState.trainingCompleted {
                HStack {
                    Text("Max Thermal Level:")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    Spacer()
                    Text(thermalLevelString(trainingState.maxThermalLevel))
                        .font(.system(.caption, design: .monospaced))
                        .fontWeight(.semibold)
                        .foregroundColor(thermalColorForLevel(trainingState.maxThermalLevel))
                }
            }
        }
        .padding()
        .background(Color(.systemGray6))
        .cornerRadius(10)
    }

    // MARK: - Experiment Settings Section

    private var experimentSettingsSection: some View {
        VStack(spacing: 12) {
            HStack {
                Text("Experiment Settings")
                    .font(.headline)
                    .fontWeight(.semibold)
                Spacer()
            }

            VStack(spacing: 10) {
                // Sequence Length (read-only after model init, editable before)
                HStack {
                    Text("Sequence Length:")
                        .fontWeight(.medium)
                    Spacer()
                    if trainingState.isModelInitialized, let modelSeq = trainingState.modelSeqLength {
                        // Show model's fixed seq_length after initialization
                        Text("\(modelSeq)")
                            .font(.system(.body, design: .monospaced))
                            .foregroundColor(.secondary)
                            .padding(.horizontal, 12)
                            .padding(.vertical, 6)
                            .background(Color(.systemGray5))
                            .cornerRadius(6)
                        Text("(fixed by model)")
                            .font(.caption2)
                            .foregroundColor(.orange)
                    } else {
                        // Editable before model init
                        Picker("Sequence Length", selection: $trainingState.experimentConfig.sequenceLength) {
                            Text("64").tag(64)
                            Text("128").tag(128)
                            Text("256").tag(256)
                            Text("512").tag(512)
                            Text("1024").tag(1024)
                        }
                        .pickerStyle(MenuPickerStyle())
                        .frame(width: 100)
                        .disabled(trainingState.isTraining)
                    }
                }

                // Batch Size (read-only after model init, editable before)
                HStack {
                    Text("Batch Size:")
                        .fontWeight(.medium)
                    Spacer()
                    if trainingState.isModelInitialized, let modelBatch = trainingState.modelBatchSize {
                        // Show model's fixed batch_size after initialization
                        Text("\(modelBatch)")
                            .font(.system(.body, design: .monospaced))
                            .foregroundColor(.secondary)
                            .padding(.horizontal, 12)
                            .padding(.vertical, 6)
                            .background(Color(.systemGray5))
                            .cornerRadius(6)
                        Text("(fixed by model)")
                            .font(.caption2)
                            .foregroundColor(.orange)
                    } else {
                        // Editable before model init
                        Stepper(value: $trainingState.experimentConfig.batchSize, in: 1...8) {
                            Text("\(trainingState.experimentConfig.batchSize)")
                                .font(.system(.body, design: .monospaced))
                                .frame(width: 30, alignment: .trailing)
                        }
                        .disabled(trainingState.isTraining)
                    }
                }

                // Info display (read-only)
                HStack {
                    Text("Samples Available:")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    Spacer()
                    Text("\(trainingState.experimentConfig.numSamples)")
                        .font(.system(.caption, design: .monospaced))
                        .foregroundColor(.secondary)
                }

                HStack {
                    Text("LoRA Rank:")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    Spacer()
                    Text("\(trainingState.experimentConfig.loraRank)")
                        .font(.system(.caption, design: .monospaced))
                        .foregroundColor(.secondary)
                }
            }
        }
        .padding()
        .background(Color(.systemGray6))
        .cornerRadius(10)
    }

    // MARK: - Training Settings Section

    private var trainingSettingsSection: some View {
        VStack(spacing: 12) {
            Text("Training Settings")
                .font(.headline)
                .fontWeight(.semibold)

            VStack(spacing: 10) {
                // Model Selection (Dynamic)
                HStack {
                    Text("Model:")
                        .fontWeight(.medium)
                    Spacer()
                    if trainingState.availableModels.isEmpty {
                        Text("No models found")
                            .font(.caption)
                            .foregroundColor(.red)
                    } else {
                        Picker("Model", selection: $trainingState.selectedModelConfiguration) {
                            ForEach(trainingState.availableModels) { config in
                                Text(config.displayName).tag(Optional(config))
                            }
                        }
                        .pickerStyle(MenuPickerStyle())
                        .disabled(trainingState.isModelInitialized)
                    }
                }

                // Refresh models button
                if !trainingState.isModelInitialized {
                    Button(action: {
                        trainingState.scanForModels()
                    }) {
                        HStack {
                            Image(systemName: "arrow.clockwise")
                            Text("Refresh Models")
                        }
                        .font(.caption)
                        .foregroundColor(.blue)
                    }
                }

                // Storage Mode Toggle
                HStack {
                    Text("Storage Mode:")
                        .fontWeight(.medium)
                    Spacer()
                    Picker("Storage Mode", selection: $trainingState.useInMemoryStorage) {
                        Text("In-Memory").tag(true)
                        Text("Mmap").tag(false)
                    }
                    .pickerStyle(SegmentedPickerStyle())
                    .frame(width: 180)
                    .disabled(trainingState.isTraining)
                }

                // Learning Rate Input
                HStack {
                    Text("Learning Rate:")
                        .fontWeight(.medium)
                    Spacer()
                    TextField("0.0001", value: $trainingState.learningRate, format: .number.precision(.fractionLength(5)))
                        .textFieldStyle(RoundedBorderTextFieldStyle())
                        .frame(width: 100)
                        .keyboardType(.decimalPad)
                        .disabled(trainingState.isTraining)
                }

                // Number of Steps Input
                HStack {
                    Text("Number of Steps:")
                        .fontWeight(.medium)
                    Spacer()
                    TextField("50", value: $trainingState.numberOfSteps, format: .number)
                        .textFieldStyle(RoundedBorderTextFieldStyle())
                        .frame(width: 80)
                        .keyboardType(.numberPad)
                        .disabled(trainingState.isTraining)
                }

                // Debug Memory Toggle
                HStack {
                    Text("Debug Memory:")
                        .fontWeight(.medium)
                    Spacer()
                    Toggle("", isOn: $trainingState.debugMemory)
                        .labelsHidden()
                        .disabled(trainingState.isTraining)
                }
                if trainingState.debugMemory {
                    Text("Detailed memory logs will be printed to Xcode console")
                        .font(.caption2)
                        .foregroundColor(.orange)
                }
            }
        }
        .padding()
        .background(Color(.systemGray6))
        .cornerRadius(10)
    }

    // MARK: - Training Control Section

    @ViewBuilder
    private var trainingControlSection: some View {
        if !trainingState.isModelInitialized {
            // Show Initialize Model button
            VStack(spacing: 8) {
                Button(action: {
                    Task {
                        await MLXMeSPExampleApp.initializeModel()
                    }
                }) {
                    HStack {
                        Image(systemName: "gearshape.fill")
                        Text("Initialize Model")
                    }
                    .font(.title2)
                    .fontWeight(.semibold)
                    .foregroundColor(.white)
                    .padding(.horizontal, 30)
                    .padding(.vertical, 15)
                    .background(trainingState.selectedModelConfiguration != nil ? Color.green : Color.gray)
                    .cornerRadius(10)
                }
                .disabled(trainingState.selectedModelConfiguration == nil)

                if trainingState.selectedModelConfiguration == nil {
                    Text("Please select a model first")
                        .font(.caption)
                        .foregroundColor(.orange)
                }
            }
        } else if !trainingState.canStartTraining {
            VStack {
                ProgressView()
                    .scaleEffect(1.2)
                Text("Initializing model...")
                    .foregroundColor(.secondary)
                    .padding(.top, 8)
            }
        } else {
            // Model is initialized - show data visualization button and training controls
            VStack(spacing: 12) {
                // Data Visualization Button
                Button(action: {
                    trainingState.showDataVisualization = true
                }) {
                    HStack {
                        Image(systemName: "doc.text.magnifyingglass")
                        Text("View Training Data")
                    }
                    .font(.title3)
                    .fontWeight(.medium)
                    .foregroundColor(.blue)
                    .padding(.horizontal, 20)
                    .padding(.vertical, 10)
                    .background(Color.blue.opacity(0.1))
                    .cornerRadius(8)
                }
                .sheet(isPresented: $trainingState.showDataVisualization) {
                    DataVisualizationView()
                }

                // Training Controls
                if !trainingState.isTraining && !trainingState.trainingCompleted && trainingState.iterationStats.isEmpty {
                    // Start button
                    Button(action: {
                        Task {
                            await MLXMeSPExampleApp.startTraining()
                        }
                    }) {
                        HStack {
                            Image(systemName: "play.fill")
                            Text("Start Training")
                        }
                        .font(.title2)
                        .fontWeight(.semibold)
                        .foregroundColor(.white)
                        .padding(.horizontal, 30)
                        .padding(.vertical, 15)
                        .background(Color.blue)
                        .cornerRadius(10)
                    }
                } else {
                    trainingProgressSection
                }
            }
        }
    }

    // MARK: - Training Progress Section

    private var trainingProgressSection: some View {
        VStack(spacing: 15) {
            // Progress Bar
            ProgressView(value: Double(trainingState.currentStep + 1), total: Double(trainingState.totalSteps))
                .progressViewStyle(LinearProgressViewStyle())

            Text("Step \(trainingState.currentStep + 1) of \(trainingState.totalSteps)")
                .font(.headline)

            // Current Status and Stop Button
            HStack {
                Text(trainingState.isTraining ? "Training..." :
                     trainingState.trainingCancelled ? "Training Cancelled" : "Training Completed")
                    .foregroundColor(trainingState.isTraining ? .blue :
                                   trainingState.trainingCancelled ? .orange : .green)
                    .fontWeight(.medium)

                Spacer()

                if trainingState.isTraining {
                    Button(action: {
                        trainingState.cancelTraining()
                    }) {
                        HStack {
                            Image(systemName: "stop.fill")
                            Text("Stop")
                        }
                        .font(.caption)
                        .fontWeight(.semibold)
                        .foregroundColor(.white)
                        .padding(.horizontal, 12)
                        .padding(.vertical, 6)
                        .background(Color.red)
                        .cornerRadius(6)
                    }
                }
            }

            // Per-iteration statistics
            if !trainingState.iterationStats.isEmpty {
                Text("Iteration Results")
                    .font(.headline)
                    .fontWeight(.semibold)

                ScrollViewReader { proxy in
                    ScrollView {
                        LazyVStack(spacing: 8) {
                            ForEach(Array(trainingState.iterationStats.enumerated()), id: \.offset) { index, stats in
                                HStack {
                                    Text("Step \(stats.step + 1)")
                                        .fontWeight(.semibold)
                                        .frame(width: 60, alignment: .leading)

                                    Spacer()

                                    VStack(alignment: .trailing, spacing: 2) {
                                        Text("Loss: \(String(format: "%.4f", stats.loss))")
                                            .font(.system(.caption, design: .monospaced))
                                        Text("Time: \(String(format: "%.1f", stats.iterationTime * 1000))ms")
                                            .font(.system(.caption, design: .monospaced))
                                            .foregroundColor(.secondary)
                                        Text("Mem: \(String(format: "%.0f", stats.startMemoryMB))->\(String(format: "%.0f", stats.endMemoryMB))MB")
                                            .font(.system(.caption, design: .monospaced))
                                            .foregroundColor(.blue)
                                        Text("Peak: \(String(format: "%.0f", stats.peakMemoryMB))MB")
                                            .font(.system(.caption, design: .monospaced))
                                            .foregroundColor(.orange)
                                    }
                                }
                                .padding(.horizontal, 12)
                                .padding(.vertical, 8)
                                .background(index == trainingState.iterationStats.count - 1 ?
                                          Color.green.opacity(0.2) : Color(.systemGray6))
                                .cornerRadius(8)
                                .id(index)
                            }
                        }
                    }
                    .frame(maxHeight: 300)
                    .onChange(of: trainingState.iterationStats.count) { newCount in
                        guard newCount > 0 else { return }
                        withAnimation(.easeInOut(duration: 0.5)) {
                            proxy.scrollTo(newCount - 1, anchor: .bottom)
                        }
                    }
                }
            } else if trainingState.isTraining {
                Text("Waiting for first iteration...")
                    .foregroundColor(.secondary)
                    .font(.caption)
            }

            // Summary statistics
            if !trainingState.iterationStats.isEmpty {
                summaryStatisticsSection
            }

            // Action buttons when training is completed or cancelled
            if trainingState.trainingCompleted || trainingState.trainingCancelled {
                actionButtonsSection
            }
        }
        .padding()
        .background(Color(.systemGray6))
        .cornerRadius(10)
    }

    // MARK: - Summary Statistics Section

    private var summaryStatisticsSection: some View {
        VStack(spacing: 8) {
            Divider()

            // Time and Memory Row 1
            HStack {
                VStack(alignment: .center, spacing: 4) {
                    Text("Avg Time:")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    Text(String(format: "%.1f ms", averageTime * 1000))
                        .font(.system(.body, design: .monospaced))
                        .fontWeight(.semibold)
                }

                Spacer()

                VStack(alignment: .center, spacing: 4) {
                    Text("Avg Mem:")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    Text(String(format: "%.0f->%.0f MB", averageStartMemory, averageEndMemory))
                        .font(.system(.caption, design: .monospaced))
                        .fontWeight(.semibold)
                        .foregroundColor(.blue)
                }

                Spacer()

                VStack(alignment: .center, spacing: 4) {
                    Text("Max Peak:")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    Text(String(format: "%.0f MB", maxPeakMemory))
                        .font(.system(.body, design: .monospaced))
                        .fontWeight(.semibold)
                        .foregroundColor(.orange)
                }
            }

            // Thermal and Battery Summary (during/after training)
            if trainingState.trainingCompleted || trainingState.trainingCancelled {
                HStack {
                    VStack(alignment: .center, spacing: 4) {
                        Text("Max Thermal:")
                            .font(.caption)
                            .foregroundColor(.secondary)
                        Text(thermalLevelString(maxThermalLevel))
                            .font(.system(.caption, design: .monospaced))
                            .fontWeight(.semibold)
                            .foregroundColor(thermalColorForLevel(maxThermalLevel))
                    }

                    Spacer()

                    VStack(alignment: .center, spacing: 4) {
                        Text("Battery Used:")
                            .font(.caption)
                            .foregroundColor(.secondary)
                        Text("\(batteryConsumed)%")
                            .font(.system(.caption, design: .monospaced))
                            .fontWeight(.semibold)
                            .foregroundColor(batteryConsumed > 5 ? .orange : .green)
                    }
                }
            }
        }
    }

    // MARK: - Action Buttons Section

    private var actionButtonsSection: some View {
        VStack(spacing: 10) {
            // Export buttons
            HStack(spacing: 12) {
                Button(action: {
                    let modelName = trainingState.selectedModelConfiguration?.displayName ?? "Unknown"
                    let method = trainingState.selectedModelConfiguration?.directoryName ?? "unknown"
                    if let url = trainingState.exportResults(modelName: modelName, method: method) {
                        exportedFileURL = url
                        showExportSheet = true
                    }
                }) {
                    HStack {
                        Image(systemName: "square.and.arrow.up")
                        Text("Export JSON")
                    }
                    .font(.caption)
                    .fontWeight(.medium)
                    .foregroundColor(.white)
                    .padding(.horizontal, 16)
                    .padding(.vertical, 8)
                    .background(Color.purple)
                    .cornerRadius(8)
                }

                Button(action: {
                    let modelName = trainingState.selectedModelConfiguration?.displayName ?? "Unknown"
                    let method = trainingState.selectedModelConfiguration?.directoryName ?? "unknown"
                    if let url = trainingState.exportResultsCSV(modelName: modelName, method: method) {
                        exportedFileURL = url
                        showExportSheet = true
                    }
                }) {
                    HStack {
                        Image(systemName: "tablecells")
                        Text("Export CSV")
                    }
                    .font(.caption)
                    .fontWeight(.medium)
                    .foregroundColor(.white)
                    .padding(.horizontal, 16)
                    .padding(.vertical, 8)
                    .background(Color.green)
                    .cornerRadius(8)
                }
            }

            // Train Again button
            Button(action: {
                trainingState.reset()
            }) {
                HStack {
                    Image(systemName: "arrow.clockwise")
                    Text("Train Again")
                }
                .font(.title3)
                .fontWeight(.medium)
                .foregroundColor(.blue)
                .padding(.horizontal, 20)
                .padding(.vertical, 10)
                .background(Color.blue.opacity(0.1))
                .cornerRadius(8)
            }
        }
        .padding(.top, 10)
    }

    // MARK: - Computed Properties

    private var memoryBarColor: Color {
        let percentage = trainingState.currentMemoryUsage.percentageUsed
        if percentage < 50 {
            return .green
        } else if percentage < 75 {
            return .orange
        } else {
            return .red
        }
    }

    private var averageTime: Double {
        guard !trainingState.iterationStats.isEmpty else { return 0.0 }
        let totalTime = trainingState.iterationStats.reduce(0.0) { $0 + $1.iterationTime }
        return totalTime / Double(trainingState.iterationStats.count)
    }

    private var averageStartMemory: Double {
        guard !trainingState.iterationStats.isEmpty else { return 0.0 }
        let total = trainingState.iterationStats.reduce(0.0) { $0 + $1.startMemoryMB }
        return total / Double(trainingState.iterationStats.count)
    }

    private var averageEndMemory: Double {
        guard !trainingState.iterationStats.isEmpty else { return 0.0 }
        let total = trainingState.iterationStats.reduce(0.0) { $0 + $1.endMemoryMB }
        return total / Double(trainingState.iterationStats.count)
    }

    private var averagePeakMemory: Double {
        guard !trainingState.iterationStats.isEmpty else { return 0.0 }
        let totalMemory = trainingState.iterationStats.reduce(0.0) { $0 + $1.peakMemoryMB }
        return totalMemory / Double(trainingState.iterationStats.count)
    }

    private var maxPeakMemory: Double {
        guard !trainingState.iterationStats.isEmpty else { return 0.0 }
        return trainingState.iterationStats.map { $0.peakMemoryMB }.max() ?? 0.0
    }

    private var maxThermalLevel: Int {
        guard !trainingState.iterationStats.isEmpty else { return 0 }
        return trainingState.iterationStats.map { $0.thermalLevel }.max() ?? 0
    }

    private var batteryConsumed: Int {
        guard !trainingState.iterationStats.isEmpty else { return 0 }
        guard let firstBattery = trainingState.iterationStats.first?.batteryLevel,
              let lastBattery = trainingState.iterationStats.last?.batteryLevel,
              firstBattery >= 0, lastBattery >= 0 else { return 0 }
        return max(0, firstBattery - lastBattery)
    }

    // MARK: - Thermal/Battery Helpers

    private var thermalColor: Color {
        thermalColorForLevel(trainingState.currentThermalInfo.stateLevel)
    }

    private func thermalColorForLevel(_ level: Int) -> Color {
        switch level {
        case 0: return .green   // Nominal
        case 1: return .yellow  // Fair
        case 2: return .orange  // Serious
        case 3: return .red     // Critical
        default: return .gray
        }
    }

    private func thermalLevelString(_ level: Int) -> String {
        switch level {
        case 0: return "Nominal"
        case 1: return "Fair"
        case 2: return "Serious"
        case 3: return "Critical"
        default: return "Unknown"
        }
    }

    private var batteryColor: Color {
        let level = trainingState.currentBatteryInfo.levelPercent
        if level < 0 {
            return .gray
        } else if level <= 20 {
            return .red
        } else if level <= 50 {
            return .orange
        } else {
            return .green
        }
    }

    private var batteryIcon: String {
        let level = trainingState.currentBatteryInfo.levelPercent
        let isCharging = trainingState.currentBatteryInfo.stateString == "Charging"

        if isCharging {
            return "battery.100.bolt"
        }

        if level < 0 {
            return "battery.0"
        } else if level <= 25 {
            return "battery.25"
        } else if level <= 50 {
            return "battery.50"
        } else if level <= 75 {
            return "battery.75"
        } else {
            return "battery.100"
        }
    }
}

// MARK: - Share Sheet for iOS

struct ShareSheet: UIViewControllerRepresentable {
    let activityItems: [Any]

    func makeUIViewController(context: Context) -> UIActivityViewController {
        UIActivityViewController(activityItems: activityItems, applicationActivities: nil)
    }

    func updateUIViewController(_ uiViewController: UIActivityViewController, context: Context) {}
}
