//
//  HazeCoderNanoView.swift
//  MLXMeSPExample
//

import SwiftUI
import MLXMeSP

struct HazeCoderNanoView: View {
    @State private var result: HazeCoderSelfTestResult?
    @State private var isRunning = false
    @State private var trainingResult: HazeCoderTrainingProofResult?
    @State private var isTraining = false
    @State private var codePipelineResult: HazeCoderCodePipelineResult?
    @State private var isCodeTraining = false
    @State private var generationPrompt = HazeCoderTinyCodeCorpus.defaultPrompt
    @State private var generationResult: HazeCoderGenerationResult?
    @State private var isGenerating = false

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 18) {
                    header
                    architectureCard
                    runCard

                    if let result {
                        resultCard(result)
                    }

                    trainingCard

                    if let trainingResult {
                        trainingResultCard(trainingResult)
                    }

                    codePipelineCard

                    if let codePipelineResult {
                        codePipelineResultCard(codePipelineResult)
                    }

                    generationCard

                    if let generationResult {
                        generationResultCard(generationResult)
                    }
                }
                .padding()
            }
            .navigationTitle("HazeCoder")
        }
    }

    private var header: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("HazeCoder-Nano")
                .font(.largeTitle)
                .fontWeight(.bold)

            Text("Phase 3 — real code text, tokenizer, checkpoints and generation")
                .font(.subheadline)
                .foregroundColor(.secondary)
        }
    }

    private var architectureCard: some View {
        let config = HazeCoderConfig.nano

        return VStack(alignment: .leading, spacing: 10) {
            Label("From-scratch model", systemImage: "cpu")
                .font(.headline)

            metricRow("Vocabulary", "\(config.vocabSize)")
            metricRow("Context", "\(config.maxSequenceLength)")
            metricRow("Layers", "\(config.numLayers)")
            metricRow("Hidden", "\(config.hiddenSize)")
            metricRow("GQA heads", "\(config.numQueryHeads) Q / \(config.numKVHeads) KV")
            metricRow("Head dimension", "\(config.headDimension)")
            metricRow("SwiGLU FFN", "\(config.intermediateSize)")
            metricRow("Weight dtype", config.useBFloat16 ? "BF16" : "FP32")
            metricRow("Expected params", formatted(config.estimatedParameterCount))

            Text("No pretrained weights are used. Phase 3 adds a reversible byte-level code tokenizer, a tiny authored code corpus, safetensors checkpoints and checkpoint-backed generation.")
                .font(.caption)
                .foregroundColor(.secondary)
                .padding(.top, 4)
        }
        .padding()
        .background(Color(.secondarySystemBackground))
        .clipShape(RoundedRectangle(cornerRadius: 14))
    }

    private var runCard: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("On-device forward test")
                .font(.headline)

            Text("Creates a new randomly initialized Nano model, generates dummy token IDs, runs all 5 Transformer blocks, and verifies output shape plus NaN/Inf counts.")
                .font(.caption)
                .foregroundColor(.secondary)

            Button {
                runTest()
            } label: {
                HStack {
                    Image(systemName: "play.fill")
                    Text(isRunning ? "Running…" : "Run Nano Forward Test")
                        .fontWeight(.semibold)
                }
                .frame(maxWidth: .infinity)
                .padding(.vertical, 12)
            }
            .buttonStyle(.borderedProminent)
            .disabled(isRunning)
        }
        .padding()
        .background(Color(.secondarySystemBackground))
        .clipShape(RoundedRectangle(cornerRadius: 14))
    }

    private var trainingCard: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("On-device training proof")
                .font(.headline)

            Text("Runs 8 next-token training steps over a tiny deterministic token pattern. MLX autodiff computes gradients for all 9.44M parameters; AdamW updates BF16 weights with FP32 optimizer moments.")
                .font(.caption)
                .foregroundColor(.secondary)

            Button {
                runTrainingProof()
            } label: {
                HStack {
                    Image(systemName: "brain.head.profile")
                    Text(isTraining ? "Training…" : "Run Nano Training Proof")
                        .fontWeight(.semibold)
                }
                .frame(maxWidth: .infinity)
                .padding(.vertical, 12)
            }
            .buttonStyle(.borderedProminent)
            .tint(.purple)
            .disabled(isTraining || isRunning)
        }
        .padding()
        .background(Color(.secondarySystemBackground))
        .clipShape(RoundedRectangle(cornerRadius: 14))
    }

    private var codePipelineCard: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Phase 3 — real code pipeline")
                .font(.headline)

            Text("Tokenizes real Python, Swift, C, Rust and JavaScript source text, trains all 9.44M parameters for 16 steps with a 32-token context, saves a .safetensors checkpoint, reloads it, then generates from the reloaded weights.")
                .font(.caption)
                .foregroundColor(.secondary)

            Button {
                runCodePipeline()
            } label: {
                HStack {
                    Image(systemName: "chevron.left.forwardslash.chevron.right")
                    Text(
                        isCodeTraining
                            ? "Training real code…"
                            : "Train Tiny Code Corpus + Save Checkpoint"
                    )
                    .fontWeight(.semibold)
                }
                .frame(maxWidth: .infinity)
                .padding(.vertical, 12)
            }
            .buttonStyle(.borderedProminent)
            .tint(.orange)
            .disabled(
                isCodeTraining ||
                isTraining ||
                isRunning ||
                isGenerating
            )
        }
        .padding()
        .background(Color(.secondarySystemBackground))
        .clipShape(RoundedRectangle(cornerRadius: 14))
    }

    @ViewBuilder
    private func codePipelineResultCard(
        _ result: HazeCoderCodePipelineResult
    ) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            Label(
                result.passed ? "CODE PIPELINE PASS" : "CODE PIPELINE FAIL",
                systemImage: result.passed
                    ? "checkmark.circle.fill"
                    : "xmark.octagon.fill"
            )
            .font(.title2)
            .fontWeight(.bold)
            .foregroundColor(result.passed ? .green : .red)

            Text(result.message)
                .font(.subheadline)

            Divider()

            metricRow("Corpus tokens", formatted(result.corpusTokenCount))
            metricRow("Steps", "\(result.steps)")
            metricRow("Context", "\(result.contextLength)")
            metricRow("Parameters", formatted(result.parameterCount))
            metricRow("Initial loss", finiteFloat(result.initialLoss))
            metricRow("Final loss", finiteFloat(result.finalLoss))
            metricRow(
                "Weight Δ MSE",
                finiteScientific(result.parameterChangeMeanSquare)
            )
            metricRow(
                "Checkpoint",
                result.checkpointBytes > 0
                    ? String(
                        format: "%.2f MB",
                        Double(result.checkpointBytes) /
                            1024.0 / 1024.0
                    )
                    : "n/a"
            )
            metricRow(
                "Elapsed",
                String(format: "%.1f ms", result.elapsedMilliseconds)
            )

            if !result.checkpointPath.isEmpty {
                Text("Checkpoint")
                    .font(.caption)
                    .foregroundColor(.secondary)
                    .padding(.top, 4)
                Text(result.checkpointPath)
                    .font(.system(.caption2, design: .monospaced))
                    .textSelection(.enabled)
                    .fixedSize(horizontal: false, vertical: true)
            }

            Divider()

            Text("Generation after checkpoint reload")
                .font(.headline)

            Text("Prompt")
                .font(.caption)
                .foregroundColor(.secondary)

            Text(result.prompt)
                .font(.system(.body, design: .monospaced))
                .textSelection(.enabled)

            Text("HazeCoder continuation")
                .font(.caption)
                .foregroundColor(.secondary)
                .padding(.top, 4)

            Text(
                result.generatedText.isEmpty
                    ? "(no text generated)"
                    : result.generatedText
            )
            .font(.system(.body, design: .monospaced))
            .textSelection(.enabled)
            .padding(10)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(Color(.tertiarySystemBackground))
            .clipShape(RoundedRectangle(cornerRadius: 8))
        }
        .padding()
        .background(Color(.secondarySystemBackground))
        .clipShape(RoundedRectangle(cornerRadius: 14))
    }

    private var generationCard: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Generate from latest checkpoint")
                .font(.headline)

            Text("Edit the prefix below. This loads the saved Phase 3 weights and performs greedy generation without retraining.")
                .font(.caption)
                .foregroundColor(.secondary)

            TextField(
                "Code prompt",
                text: $generationPrompt,
                axis: .vertical
            )
            .font(.system(.body, design: .monospaced))
            .textFieldStyle(.roundedBorder)
            .lineLimit(3 ... 8)

            Button {
                runCheckpointGeneration()
            } label: {
                HStack {
                    Image(systemName: "sparkles")
                    Text(
                        isGenerating
                            ? "Generating…"
                            : "Generate from Saved HazeCoder"
                    )
                    .fontWeight(.semibold)
                }
                .frame(maxWidth: .infinity)
                .padding(.vertical, 12)
            }
            .buttonStyle(.borderedProminent)
            .tint(.green)
            .disabled(
                isGenerating ||
                isCodeTraining ||
                generationPrompt.isEmpty
            )
        }
        .padding()
        .background(Color(.secondarySystemBackground))
        .clipShape(RoundedRectangle(cornerRadius: 14))
    }

    @ViewBuilder
    private func generationResultCard(
        _ result: HazeCoderGenerationResult
    ) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            Label(
                result.passed ? "GENERATION PASS" : "GENERATION FAIL",
                systemImage: result.passed
                    ? "checkmark.circle.fill"
                    : "xmark.octagon.fill"
            )
            .font(.title2)
            .fontWeight(.bold)
            .foregroundColor(result.passed ? .green : .red)

            Text(result.message)
                .font(.subheadline)

            metricRow(
                "Elapsed",
                String(format: "%.1f ms", result.elapsedMilliseconds)
            )

            Divider()

            Text(result.prompt)
                .font(.system(.body, design: .monospaced))
                .foregroundColor(.secondary)
                .textSelection(.enabled)

            Text(
                result.generatedText.isEmpty
                    ? "(no text generated)"
                    : result.generatedText
            )
            .font(.system(.body, design: .monospaced))
            .textSelection(.enabled)
            .padding(10)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(Color(.tertiarySystemBackground))
            .clipShape(RoundedRectangle(cornerRadius: 8))
        }
        .padding()
        .background(Color(.secondarySystemBackground))
        .clipShape(RoundedRectangle(cornerRadius: 14))
    }

    @ViewBuilder
    private func trainingResultCard(
        _ result: HazeCoderTrainingProofResult
    ) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            Label(
                result.passed ? "LEARNING PASS" : "TRAINING FAIL",
                systemImage: result.passed
                    ? "checkmark.circle.fill"
                    : "xmark.octagon.fill"
            )
            .font(.title2)
            .fontWeight(.bold)
            .foregroundColor(result.passed ? .green : .red)

            Text(result.message)
                .font(.subheadline)

            Divider()

            metricRow("Steps", "\(result.steps)")
            metricRow("Sequence", "\(result.sequenceLength)")
            metricRow("Parameters", formatted(result.parameterCount))
            metricRow(
                "Initial loss",
                finiteFloat(result.initialLoss)
            )
            metricRow(
                "Final loss",
                finiteFloat(result.finalLoss)
            )

            if result.initialLoss.isFinite &&
               result.finalLoss.isFinite &&
               result.initialLoss != 0 {
                let reduction =
                    (1.0 - result.finalLoss / result.initialLoss) * 100.0
                metricRow(
                    "Loss reduction",
                    String(format: "%.2f%%", reduction)
                )
            }

            metricRow(
                "Weight Δ MSE",
                finiteScientific(result.parameterChangeMeanSquare)
            )
            metricRow(
                "Elapsed",
                String(format: "%.1f ms", result.elapsedMilliseconds)
            )

            if !result.lossHistory.isEmpty {
                Text(
                    "Loss history: " +
                    result.lossHistory
                        .map { finiteFloat($0) }
                        .joined(separator: " → ")
                )
                .font(.system(.caption2, design: .monospaced))
                .foregroundColor(.secondary)
                .fixedSize(horizontal: false, vertical: true)
                .padding(.top, 4)
            }
        }
        .padding()
        .background(Color(.secondarySystemBackground))
        .clipShape(RoundedRectangle(cornerRadius: 14))
    }

    @ViewBuilder
    private func resultCard(_ result: HazeCoderSelfTestResult) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            Label(
                result.passed ? "PASS" : "FAIL",
                systemImage: result.passed ? "checkmark.circle.fill" : "xmark.octagon.fill"
            )
            .font(.title2)
            .fontWeight(.bold)
            .foregroundColor(result.passed ? .green : .red)

            Text(result.message)
                .font(.subheadline)

            Divider()

            metricRow("Input", shape(result.inputShape))
            metricRow("Logits", shape(result.outputShape))
            metricRow("Parameters", formatted(result.parameterCount))
            metricRow("Expected", formatted(result.expectedParameterCount))
            metricRow("NaN", result.nanCount >= 0 ? "\(result.nanCount)" : "n/a")
            metricRow("Inf", result.infCount >= 0 ? "\(result.infCount)" : "n/a")
            metricRow("DType", result.dtypeName)
            metricRow(
                "Elapsed",
                String(format: "%.1f ms", result.elapsedMilliseconds)
            )
        }
        .padding()
        .background(Color(.secondarySystemBackground))
        .clipShape(RoundedRectangle(cornerRadius: 14))
    }

    private func metricRow(_ label: String, _ value: String) -> some View {
        HStack {
            Text(label)
                .foregroundColor(.secondary)
            Spacer()
            Text(value)
                .font(.system(.body, design: .monospaced))
                .multilineTextAlignment(.trailing)
        }
    }

    private func shape(_ shape: [Int]) -> String {
        guard !shape.isEmpty else { return "n/a" }
        return "[" + shape.map(String.init).joined(separator: ", ") + "]"
    }

    private func formatted(_ value: Int) -> String {
        value.formatted(.number.grouping(.automatic))
    }

    private func finiteFloat(_ value: Float) -> String {
        value.isFinite ? String(format: "%.4f", value) : "n/a"
    }

    private func finiteScientific(_ value: Float) -> String {
        value.isFinite ? String(format: "%.3e", value) : "n/a"
    }

    private func runCodePipeline() {
        isCodeTraining = true
        codePipelineResult = nil
        generationResult = nil

        // Keep the entire MLX training job on one background executor.
        // Only the Sendable scalar/string result returns to SwiftUI.
        Task {
            let completed = await Task.detached(
                priority: .userInitiated
            ) {
                HazeCoderTrainer.runTinyCodePipeline()
            }.value

            codePipelineResult = completed
            isCodeTraining = false
        }
    }

    private func runCheckpointGeneration() {
        isGenerating = true
        generationResult = nil
        let prompt = generationPrompt

        Task {
            let completed = await Task.detached(
                priority: .userInitiated
            ) {
                HazeCoderTrainer.generateFromLatestCodeCheckpoint(
                    prompt: prompt
                )
            }.value

            generationResult = completed
            isGenerating = false
        }
    }

    private func runTrainingProof() {
        isTraining = true
        trainingResult = nil

        trainingResult = HazeCoderTrainer.runNanoTrainingProof()
        isTraining = false
    }

    private func runTest() {
        isRunning = true
        result = nil

        // Phase 1 intentionally keeps the complete MLX graph on one thread.
        // The result contains only scalar/shape metadata.
        result = HazeCoderModel.runNanoSelfTest(sequenceLength: 32)
        isRunning = false
    }
}
