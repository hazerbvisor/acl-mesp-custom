//
//  HazeCoderModel.swift
//  MLXMeSP
//
//  Phase 1 only: random initialization + complete causal forward pass.
//

import Foundation
import MLX

public enum HazeCoderModelError: Error, CustomStringConvertible {
    case invalidInput(String)

    public var description: String {
        switch self {
        case .invalidInput(let message):
            return message
        }
    }
}

public struct HazeCoderSelfTestResult: Sendable {
    public let passed: Bool
    public let message: String
    public let inputShape: [Int]
    public let outputShape: [Int]
    public let parameterCount: Int
    public let expectedParameterCount: Int
    public let nanCount: Int
    public let infCount: Int
    public let dtypeName: String
    public let elapsedMilliseconds: Double

    public init(
        passed: Bool,
        message: String,
        inputShape: [Int],
        outputShape: [Int],
        parameterCount: Int,
        expectedParameterCount: Int,
        nanCount: Int,
        infCount: Int,
        dtypeName: String,
        elapsedMilliseconds: Double
    ) {
        self.passed = passed
        self.message = message
        self.inputShape = inputShape
        self.outputShape = outputShape
        self.parameterCount = parameterCount
        self.expectedParameterCount = expectedParameterCount
        self.nanCount = nanCount
        self.infCount = infCount
        self.dtypeName = dtypeName
        self.elapsedMilliseconds = elapsedMilliseconds
    }
}

public final class HazeCoderModel {
    public let config: HazeCoderConfig

    private let dtype: DType
    private let tokenEmbedding: MLXArray
    private let blocks: [HazeCoderTransformerBlock]
    private let finalNorm: HazeCoderRMSNorm

    public init(config: HazeCoderConfig = .nano) throws {
        try config.validate()
        self.config = config
        self.dtype = config.useBFloat16 ? .bfloat16 : .float32

        MLXRandom.seed(config.seed)

        let embeddingScale = 1.0 / sqrt(Float(config.hiddenSize))
        self.tokenEmbedding = MLXRandom.normal(
            [config.vocabSize, config.hiddenSize],
            dtype: dtype,
            scale: embeddingScale
        )

        self.blocks = (0 ..< config.numLayers).map { _ in
            HazeCoderTransformerBlock(config: config, dtype: dtype)
        }

        self.finalNorm = HazeCoderRMSNorm(
            dimensions: config.hiddenSize,
            epsilon: config.rmsNormEpsilon,
            dtype: dtype
        )
    }

    /// Forward pass:
    /// token IDs [B, S] -> logits [B, S, vocab].
    ///
    /// The output projection is tied to tokenEmbedding.T, so no separate LM-head
    /// parameter matrix is allocated.
    public func callAsFunction(_ tokenIDs: MLXArray) throws -> MLXArray {
        guard tokenIDs.ndim == 2 else {
            throw HazeCoderModelError.invalidInput(
                "token IDs must have shape [batch, sequence]"
            )
        }

        let batchSize = tokenIDs.shape[0]
        let sequenceLength = tokenIDs.shape[1]

        guard sequenceLength > 0 else {
            throw HazeCoderModelError.invalidInput(
                "sequence length must be positive"
            )
        }
        guard sequenceLength <= config.maxSequenceLength else {
            throw HazeCoderModelError.invalidInput(
                "sequence length \(sequenceLength) exceeds max context \(config.maxSequenceLength)"
            )
        }

        let flatTokenIDs = tokenIDs.reshaped([-1])
        var hidden = MLX.take(tokenEmbedding, flatTokenIDs, axis: 0)
            .reshaped([batchSize, sequenceLength, config.hiddenSize])

        for block in blocks {
            hidden = block(hidden)
        }

        hidden = finalNorm(hidden)

        // Tied embedding/output projection.
        return MLX.matmul(hidden, tokenEmbedding.T)
    }

    public var parameterCount: Int {
        tokenEmbedding.size +
        blocks.reduce(0) { $0 + $1.parameterCount } +
        finalNorm.parameterCount
    }

    public var dtypeName: String {
        config.useBFloat16 ? "bfloat16" : "float32"
    }

    /// Runs the first on-device HazeCoder milestone without a tokenizer or trainer.
    /// All MLX arrays stay local to this synchronous function; only scalar results
    /// leave the function.
    public static func runNanoSelfTest(
        sequenceLength: Int = 32
    ) -> HazeCoderSelfTestResult {
        let started = Date()
        let config = HazeCoderConfig.nano

        do {
            guard sequenceLength > 0 && sequenceLength <= config.maxSequenceLength else {
                throw HazeCoderModelError.invalidInput(
                    "self-test sequence length must be 1...\(config.maxSequenceLength)"
                )
            }

            let model = try HazeCoderModel(config: config)

            let tokenIDs = MLXRandom.randInt(
                Int32(0) ..< Int32(config.vocabSize),
                [1, sequenceLength]
            )

            let logits = try model(tokenIDs)
            MLX.eval(logits)

            let nanCount = Int(
                MLX.sum(MLX.isNaN(logits).asType(.int32)).item(Int32.self)
            )
            let infCount = Int(
                MLX.sum(MLX.isInf(logits).asType(.int32)).item(Int32.self)
            )

            let expectedShape = [1, sequenceLength, config.vocabSize]
            let shapeIsCorrect = logits.shape == expectedShape
            let countIsCorrect = model.parameterCount == config.estimatedParameterCount
            let finite = nanCount == 0 && infCount == 0
            let passed = shapeIsCorrect && countIsCorrect && finite

            let elapsed = Date().timeIntervalSince(started) * 1000.0

            return HazeCoderSelfTestResult(
                passed: passed,
                message: passed
                    ? "PASS — random weights completed the full HazeCoder-Nano forward pass."
                    : "FAIL — one or more forward-pass checks did not match.",
                inputShape: tokenIDs.shape,
                outputShape: logits.shape,
                parameterCount: model.parameterCount,
                expectedParameterCount: config.estimatedParameterCount,
                nanCount: nanCount,
                infCount: infCount,
                dtypeName: model.dtypeName,
                elapsedMilliseconds: elapsed
            )
        } catch {
            let elapsed = Date().timeIntervalSince(started) * 1000.0
            return HazeCoderSelfTestResult(
                passed: false,
                message: "ERROR — \(error)",
                inputShape: [],
                outputShape: [],
                parameterCount: 0,
                expectedParameterCount: config.estimatedParameterCount,
                nanCount: -1,
                infCount: -1,
                dtypeName: config.useBFloat16 ? "bfloat16" : "float32",
                elapsedMilliseconds: elapsed
            )
        }
    }
}
