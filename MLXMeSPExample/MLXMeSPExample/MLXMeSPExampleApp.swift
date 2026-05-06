//
//  For licensing see accompanying LICENSE file.
//  Copyright (C) 2025 Apple Inc. All Rights Reserved.
//
//  MLXMeSPExampleApp.swift
//  MLXMeSPExample
//
//  Created by Congzheng Song on 9/18/25.
//

import SwiftUI
import MLXMeSP

@main
struct MLXMeSPExampleApp: App {
    static private var model: Model?
    
    var body: some Scene {
        WindowGroup {
            ContentView()
        }
    }

    static func initializeModel() async {
        let state = TrainingState.shared

        // Check if model is selected
        guard let selectedConfig = state.selectedModelConfiguration else {
            print("No model selected")
            return
        }

        // Mark as initializing
        await MainActor.run {
            state.isModelInitialized = true
            state.canStartTraining = false
        }

        do {
            // Setup model directory using selected configuration
            let fm = FileManager.default
            let docs = fm.urls(for: .documentDirectory, in: .userDomainMask).first!
            let modelDir = docs.appendingPathComponent(selectedConfig.directoryName, isDirectory: true)

            // List all files and folders in Documents directory
            print("=== Documents Directory Contents ===")
            print("Documents path: \(docs.path)")
            if let contents = try? fm.contentsOfDirectory(atPath: docs.path) {
                if contents.isEmpty {
                    print("  (empty)")
                } else {
                    for item in contents.sorted() {
                        var isDir: ObjCBool = false
                        let itemPath = docs.appendingPathComponent(item).path
                        fm.fileExists(atPath: itemPath, isDirectory: &isDir)
                        let typeIndicator = isDir.boolValue ? "[DIR]" : "[FILE]"
                        print("  \(typeIndicator) \(item)")
                    }
                }
            } else {
                print("  (failed to read)")
            }
            print("====================================")

            // Check if model directory exists
            guard fm.fileExists(atPath: modelDir.path) else {
                print("Model directory not found at: \(modelDir.path)")
                print("Available models should be placed in: \(docs.path)")
                print("Expected directory: \(selectedConfig.directoryName)")
                await MainActor.run {
                    state.isModelInitialized = false
                }
                return
            }

            print("Using model configuration: \(selectedConfig.displayName)")

            // Initialize Qwen3Runner with user's storage preference
            let useInMemoryStorage = state.useInMemoryStorage
            print("Initializing model with \(useInMemoryStorage ? "in-memory" : "memory-mapped") storage")
            model = try await Model(modelDir: modelDir, useInMemoryStorage: useInMemoryStorage)
            
            // Update experiment config with actual data count, model's seq_length and batch_size
            if let m = model {
                let dataCount = m.getDataCount()
                let formatInfo = m.getDataFormatInfo()
                let modelSeqLength = m.exportedSeqLength
                let modelBatchSize = m.exportedBatchSize
                await MainActor.run {
                    state.canStartTraining = true
                    state.experimentConfig.numSamples = dataCount
                    state.modelSeqLength = modelSeqLength
                    state.modelBatchSize = modelBatchSize
                    state.experimentConfig.sequenceLength = modelSeqLength  // Match model's seq_length
                    state.experimentConfig.batchSize = modelBatchSize        // Match model's batch_size
                }
                print("Data format: \(formatInfo), Samples: \(dataCount), Model seq_length: \(modelSeqLength), batch_size: \(modelBatchSize)")
            } else {
                await MainActor.run {
                    state.canStartTraining = true
                }
            }

            print("Model initialized successfully")
        } catch {
            print("Model initialization error: \(error)")
            await MainActor.run {
                state.isModelInitialized = false
            }
        }
    }

    static func startTraining() async {
        guard let model else {
            print("Model not initialized")
            return
        }

        let state = TrainingState.shared

        // Use user-configured settings
        let numSteps = state.numberOfSteps
        let learningRate = state.learningRate
        let sequenceLength = state.experimentConfig.sequenceLength
        let debugMemory = state.debugMemory

        // Set total steps and start experiment tracking
        await MainActor.run {
            state.totalSteps = numSteps
            state.trainingCompleted = false
            state.trainingCancelled = false
        }

        // Initialize experiment tracking for export
        state.onExperimentStart()

        // Create and store the training task for cancellation
        let task: Task<Void, Error> = Task {
            try await model.runWithProgress(
                learningRate: learningRate,
                numSteps: numSteps,
                sequenceLength: sequenceLength,
                debugMemory: debugMemory,
                onIterationStart: { step in
                    print("Starting iteration \(step)")
                    state.onIterationStart(step: step)
                },
                onIterationEnd: { step, metrics in
                    print("Completed iteration \(step)")
                    state.onIterationEnd(step: step, metrics: metrics)
                }
            )
        }
        
        // Store training task directly
        state.trainingTask = task
        
        // Wait for completion or cancellation
        do {
            try await task.value
        } catch is CancellationError {
            print("Training was cancelled - resetting MeBPRunner")
            await resetRunner()
        } catch {
            print("Training error: \(error)")
        }
    }

    static func getTrainingData() -> [Messages]? {
        return model?.getData()
    }

    static func getPreTokenizedData() -> [PreTokenizedSample]? {
        return model?.preTokenizedData
    }

    static func getDataFormat() -> TrainingDataFormat? {
        return model?.dataFormat
    }

    static func getDataFormatInfo() -> String? {
        return model?.getDataFormatInfo()
    }

    static func getDataCount() -> Int {
        return model?.getDataCount() ?? 0
    }
    
    static func resetRunner() async {
        guard model != nil else {
            print("No runner to reset")
            return
        }
        let state = TrainingState.shared
        do {
            try self.model?.recreateRunner(useInMemoryStorage: state.useInMemoryStorage)
            print("Runner reset completed")
        } catch {
            print("Failed to reset runner: \(error)")
        }
    }
}
