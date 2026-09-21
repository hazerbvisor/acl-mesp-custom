//
//  HazeCoderNanoView.swift
//  MLXMeSPExample
//

import SwiftUI
import MLXMeSP

struct HazeCoderNanoView: View {
    @State private var result: HazeCoderSelfTestResult?
    @State private var isRunning = false

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

            Text("Phase 1 — random weights → full causal forward pass → logits")
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

            Text("No pretrained weights, tokenizer, optimizer, JEPA, SSM, MoE or quantization are used in this milestone.")
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

    private func runTest() {
        isRunning = true
        result = nil

        // Phase 1 intentionally keeps the complete MLX graph on one thread.
        // The result contains only scalar/shape metadata.
        result = HazeCoderModel.runNanoSelfTest(sequenceLength: 32)
        isRunning = false
    }
}
