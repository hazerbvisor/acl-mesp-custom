//
//  For licensing see accompanying LICENSE file.
//  Copyright (C) 2025 Apple Inc. All Rights Reserved.
//
//  DataVisualizationView.swift
//  MLXMeSPExample
//
//  Created by Congzheng Song on 9/30/25.
//

import SwiftUI
import MLXMeSP

struct DataVisualizationView: View {
    @Environment(\.dismiss) private var dismiss
    @State private var trainingData: [Messages] = []
    @State private var preTokenizedData: [PreTokenizedSample] = []
    @State private var dataFormat: TrainingDataFormat = .chatFormat
    @State private var dataFormatInfo: String = ""
    @State private var dataCount: Int = 0

    var body: some View {
        NavigationView {
            VStack(spacing: 16) {
                headerView

                if dataCount == 0 {
                    loadingView
                } else {
                    dataContentView
                }
            }
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Done") {
                        dismiss()
                    }
                }
            }
        }
        .onAppear {
            loadTrainingData()
        }
    }

    private var headerView: some View {
        VStack(spacing: 4) {
            Text("Training Data")
                .font(.largeTitle)
                .fontWeight(.bold)
            if !dataFormatInfo.isEmpty {
                Text(dataFormatInfo)
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
        }
        .padding(.top)
    }

    private var loadingView: some View {
        VStack {
            ProgressView()
                .scaleEffect(1.5)
            Text("Loading training data...")
                .foregroundColor(.secondary)
                .padding(.top)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private var dataContentView: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Loaded \(dataCount) training examples")
                .font(.headline)
                .foregroundColor(.secondary)

            ScrollView {
                LazyVStack(spacing: 12) {
                    switch dataFormat {
                    case .preTokenized:
                        ForEach(Array(preTokenizedData.prefix(100).enumerated()), id: \.offset) { index, sample in
                            preTokenizedSampleView(index: index, sample: sample)
                        }
                        if preTokenizedData.count > 100 {
                            Text("... and \(preTokenizedData.count - 100) more samples")
                                .font(.caption)
                                .foregroundColor(.secondary)
                                .padding()
                        }
                    case .chatFormat:
                        ForEach(Array(trainingData.enumerated()), id: \.offset) { index, conversation in
                            conversationView(index: index, conversation: conversation)
                        }
                    }
                }
                .padding(.horizontal)
            }
        }
    }

    private func preTokenizedSampleView(index: Int, sample: PreTokenizedSample) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text("Sample \(index + 1)")
                    .font(.headline)
                    .foregroundColor(.primary)
                Spacer()
                Text("\(sample.tokens.count) tokens")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }

            // Show first 50 token IDs
            let displayTokens = Array(sample.tokens.prefix(50))
            Text("Tokens: \(displayTokens.map { String($0) }.joined(separator: ", "))\(sample.tokens.count > 50 ? "..." : "")")
                .font(.system(.caption, design: .monospaced))
                .foregroundColor(.secondary)
                .lineLimit(3)
        }
        .padding()
        .background(Color(.systemGray6))
        .cornerRadius(12)
    }

    private func conversationView(index: Int, conversation: Messages) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Example \(index + 1)")
                .font(.headline)
                .foregroundColor(.primary)

            ForEach(Array(conversation.enumerated()), id: \.offset) { msgIndex, message in
                messageView(message: message)
            }
        }
        .padding()
        .background(Color(.systemGray6))
        .cornerRadius(12)
    }

    private func messageView(message: [String: String]) -> some View {
        HStack(alignment: .top, spacing: 8) {
            Text("\(message["role"] ?? "unknown"):")
                .font(.caption)
                .fontWeight(.semibold)
                .foregroundColor(message["role"] == "user" ? .blue : .green)
                .frame(width: 60, alignment: .leading)

            VStack(alignment: .leading, spacing: 4) {
                Text(message["content"] ?? "")
                    .font(.body)
                    .multilineTextAlignment(.leading)
                    .fixedSize(horizontal: false, vertical: true)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
        }
        .padding(.vertical, 4)
    }

    private func loadTrainingData() {
        dataCount = MLXMeSPExampleApp.getDataCount()
        dataFormatInfo = MLXMeSPExampleApp.getDataFormatInfo() ?? ""

        if let format = MLXMeSPExampleApp.getDataFormat() {
            dataFormat = format
            switch format {
            case .preTokenized:
                if let data = MLXMeSPExampleApp.getPreTokenizedData() {
                    preTokenizedData = data
                }
            case .chatFormat:
                if let data = MLXMeSPExampleApp.getTrainingData() {
                    trainingData = data
                }
            }
        }
    }
}
