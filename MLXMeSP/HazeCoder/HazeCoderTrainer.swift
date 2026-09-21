//
//  HazeCoderTrainer.swift
//  MLXMeSP
//
//  Phase 2: minimal from-scratch training proof using core MLX autodiff.
//

import Foundation
import MLX

public struct HazeCoderTrainingProofResult: Sendable {
    public let passed: Bool
    public let message: String
    public let steps: Int
    public let sequenceLength: Int
    public let parameterCount: Int
    public let initialLoss: Float
    public let finalLoss: Float
    public let parameterChangeMeanSquare: Float
    public let elapsedMilliseconds: Double
    public let lossHistory: [Float]

    public init(
        passed: Bool,
        message: String,
        steps: Int,
        sequenceLength: Int,
        parameterCount: Int,
        initialLoss: Float,
        finalLoss: Float,
        parameterChangeMeanSquare: Float,
        elapsedMilliseconds: Double,
        lossHistory: [Float]
    ) {
        self.passed = passed
        self.message = message
        self.steps = steps
        self.sequenceLength = sequenceLength
        self.parameterCount = parameterCount
        self.initialLoss = initialLoss
        self.finalLoss = finalLoss
        self.parameterChangeMeanSquare = parameterChangeMeanSquare
        self.elapsedMilliseconds = elapsedMilliseconds
        self.lossHistory = lossHistory
    }
}

public enum HazeCoderTrainer {
    /// Minimal training milestone:
    /// - initialize the complete HazeCoder-Nano from random weights
    /// - next-token cross entropy on a deterministic token pattern
    /// - MLX autodiff through all model parameters
    /// - AdamW update with FP32 optimizer moments
    /// - verify loss falls and parameters actually changed
    ///
    /// This is deliberately not the real coding-data pipeline yet. It proves
    /// that the architecture can learn on-device before tokenizer/dataset work.
    public static func runNanoTrainingProof(
        steps: Int = 8,
        sequenceLength: Int = 16,
        learningRate: Float = 0.002,
        weightDecay: Float = 0.01
    ) -> HazeCoderTrainingProofResult {
        let started = Date()
        let config = HazeCoderConfig.nano

        guard steps > 0,
              sequenceLength > 0,
              sequenceLength <= config.maxSequenceLength
        else {
            return failure(
                "Invalid training-proof configuration.",
                steps: steps,
                sequenceLength: sequenceLength,
                expectedParameterCount: config.estimatedParameterCount,
                started: started
            )
        }

        MLXRandom.seed(config.seed)

        var parameters = initializeParameters(config: config)
        let parameterCount = parameters.reduce(0) { $0 + $1.size }

        guard parameterCount == config.estimatedParameterCount else {
            return failure(
                "Parameter layout mismatch: got \(parameterCount), expected \(config.estimatedParameterCount).",
                steps: steps,
                sequenceLength: sequenceLength,
                expectedParameterCount: parameterCount,
                started: started
            )
        }

        // Repeating pattern makes the tiny proof deterministic and easy to
        // memorize. These are raw token IDs, not tokenizer output.
        let motif: [Int32] = [101, 205, 309, 413]
        let fullSequence: [Int32] = (0 ... sequenceLength).map {
            motif[$0 % motif.count]
        }

        let inputIDs = MLXArray(
            Array(fullSequence.dropLast()),
            [1, sequenceLength]
        )
        let targetIDs = MLXArray(
            Array(fullSequence.dropFirst()),
            [1, sequenceLength]
        )

        let constants = makeForwardConstants(
            config: config,
            sequenceLength: sequenceLength,
            dtype: config.useBFloat16 ? .bfloat16 : .float32
        )

        func loss(_ candidateParameters: [MLXArray]) -> MLXArray {
            let logits = forward(
                tokenIDs: inputIDs,
                parameters: candidateParameters,
                config: config,
                constants: constants
            )

            return nextTokenCrossEntropy(
                logits: logits,
                targets: targetIDs,
                vocabSize: config.vocabSize
            )
        }

        let initialEmbedding = parameters[0]
        let firstLossArray = loss(parameters)
        MLX.eval(firstLossArray)
        let initialLoss = firstLossArray.item(Float.self)

        let differentiableIndices = Array(parameters.indices)
        let lossAndGrad = valueAndGrad(
            { candidateParameters in
                [loss(candidateParameters)]
            },
            argumentNumbers: differentiableIndices
        )

        var firstMoments = parameters.map {
            MLXArray.zeros($0.shape, dtype: .float32)
        }
        var secondMoments = parameters.map {
            MLXArray.zeros($0.shape, dtype: .float32)
        }

        let beta1: Float = 0.9
        let beta2: Float = 0.999
        let epsilon: Float = 1e-8
        var lossHistory = [Float]()
        lossHistory.reserveCapacity(steps + 1)
        lossHistory.append(initialLoss)

        for step in 1 ... steps {
            let (values, gradients) = lossAndGrad(parameters)

            guard let stepLoss = values.first,
                  gradients.count == parameters.count
            else {
                return failure(
                    "Autodiff returned an unexpected value/gradient layout.",
                    steps: steps,
                    sequenceLength: sequenceLength,
                    expectedParameterCount: parameterCount,
                    started: started,
                    initialLoss: initialLoss,
                    lossHistory: lossHistory
                )
            }

            let beta1Correction = Float(
                1.0 - Foundation.pow(Double(beta1), Double(step))
            )
            let beta2Correction = Float(
                1.0 - Foundation.pow(Double(beta2), Double(step))
            )

            var updatedParameters = [MLXArray]()
            updatedParameters.reserveCapacity(parameters.count)

            for i in parameters.indices {
                let parameter32 = parameters[i].asType(.float32)
                let gradient32 = gradients[i].asType(.float32)

                let newFirstMoment =
                    beta1 * firstMoments[i] +
                    (1.0 - beta1) * gradient32

                let newSecondMoment =
                    beta2 * secondMoments[i] +
                    (1.0 - beta2) * gradient32 * gradient32

                let firstMomentHat = newFirstMoment / beta1Correction
                let secondMomentHat = newSecondMoment / beta2Correction

                let adamUpdate =
                    firstMomentHat /
                    (MLX.sqrt(secondMomentHat) + epsilon)

                let decayedParameter =
                    parameter32 * (1.0 - learningRate * weightDecay)

                let updated =
                    (decayedParameter - learningRate * adamUpdate)
                    .asType(parameters[i].dtype)

                firstMoments[i] = newFirstMoment
                secondMoments[i] = newSecondMoment
                updatedParameters.append(updated)
            }

            // Materialize the update before the next step so the graph does not
            // grow across iterations.
            MLX.eval(
                updatedParameters +
                firstMoments +
                secondMoments +
                [stepLoss]
            )

            parameters = updatedParameters
            lossHistory.append(stepLoss.item(Float.self))
        }

        let finalLossArray = loss(parameters)
        let embeddingDifference =
            parameters[0].asType(.float32) -
            initialEmbedding.asType(.float32)
        let parameterChangeArray = MLX.mean(
            embeddingDifference * embeddingDifference
        )

        MLX.eval(finalLossArray, parameterChangeArray)

        let finalLoss = finalLossArray.item(Float.self)
        let parameterChangeMeanSquare =
            parameterChangeArray.item(Float.self)

        lossHistory.append(finalLoss)

        let finite =
            initialLoss.isFinite &&
            finalLoss.isFinite &&
            parameterChangeMeanSquare.isFinite

        let lossFell = finalLoss < initialLoss
        let parametersChanged = parameterChangeMeanSquare > 0
        let passed = finite && lossFell && parametersChanged

        let elapsedMilliseconds =
            Date().timeIntervalSince(started) * 1000.0

        return HazeCoderTrainingProofResult(
            passed: passed,
            message: passed
                ? "PASS — HazeCoder-Nano learned: autodiff + AdamW changed the weights and reduced next-token loss."
                : "FAIL — training completed, but the learning checks did not all pass.",
            steps: steps,
            sequenceLength: sequenceLength,
            parameterCount: parameterCount,
            initialLoss: initialLoss,
            finalLoss: finalLoss,
            parameterChangeMeanSquare: parameterChangeMeanSquare,
            elapsedMilliseconds: elapsedMilliseconds,
            lossHistory: lossHistory
        )
    }

    // MARK: - Parameter initialization

    private static func initializeParameters(
        config: HazeCoderConfig
    ) -> [MLXArray] {
        let dtype: DType =
            config.useBFloat16 ? .bfloat16 : .float32

        let embeddingScale =
            1.0 / sqrt(Float(config.hiddenSize))

        var parameters = [MLXArray]()
        parameters.reserveCapacity(2 + config.numLayers * 9)

        // Tied token embedding / LM head.
        parameters.append(
            MLXRandom.normal(
                [config.vocabSize, config.hiddenSize],
                dtype: dtype,
                scale: embeddingScale
            )
        )

        for _ in 0 ..< config.numLayers {
            // Pre-attention RMSNorm.
            parameters.append(
                MLXArray.ones([config.hiddenSize], dtype: dtype)
            )

            let attentionScale =
                1.0 / sqrt(Float(config.hiddenSize))
            let qDimensions =
                config.numQueryHeads * config.headDimension
            let kvDimensions =
                config.numKVHeads * config.headDimension

            parameters.append(
                MLXRandom.normal(
                    [config.hiddenSize, qDimensions],
                    dtype: dtype,
                    scale: attentionScale
                )
            )
            parameters.append(
                MLXRandom.normal(
                    [config.hiddenSize, kvDimensions],
                    dtype: dtype,
                    scale: attentionScale
                )
            )
            parameters.append(
                MLXRandom.normal(
                    [config.hiddenSize, kvDimensions],
                    dtype: dtype,
                    scale: attentionScale
                )
            )
            parameters.append(
                MLXRandom.normal(
                    [qDimensions, config.hiddenSize],
                    dtype: dtype,
                    scale: attentionScale
                )
            )

            // Pre-FFN RMSNorm.
            parameters.append(
                MLXArray.ones([config.hiddenSize], dtype: dtype)
            )

            let ffnInputScale =
                1.0 / sqrt(Float(config.hiddenSize))
            let ffnOutputScale =
                1.0 / sqrt(Float(config.intermediateSize))

            parameters.append(
                MLXRandom.normal(
                    [config.hiddenSize, config.intermediateSize],
                    dtype: dtype,
                    scale: ffnInputScale
                )
            )
            parameters.append(
                MLXRandom.normal(
                    [config.hiddenSize, config.intermediateSize],
                    dtype: dtype,
                    scale: ffnInputScale
                )
            )
            parameters.append(
                MLXRandom.normal(
                    [config.intermediateSize, config.hiddenSize],
                    dtype: dtype,
                    scale: ffnOutputScale
                )
            )
        }

        // Final RMSNorm.
        parameters.append(
            MLXArray.ones([config.hiddenSize], dtype: dtype)
        )

        return parameters
    }

    // MARK: - Functional forward

    private struct ForwardConstants {
        let ropeCos: MLXArray
        let ropeSin: MLXArray
        let ropeFirstHalfIndices: MLXArray
        let ropeSecondHalfIndices: MLXArray
        let causalMask: MLXArray
    }

    private static func makeForwardConstants(
        config: HazeCoderConfig,
        sequenceLength: Int,
        dtype: DType
    ) -> ForwardConstants {
        let half = config.headDimension / 2
        var cosValues = [Float]()
        var sinValues = [Float]()
        cosValues.reserveCapacity(sequenceLength * half)
        sinValues.reserveCapacity(sequenceLength * half)

        for position in 0 ..< sequenceLength {
            for i in 0 ..< half {
                let exponent =
                    Float(2 * i) / Float(config.headDimension)
                let inverseFrequency = Float(
                    Foundation.pow(
                        Double(config.ropeTheta),
                        Double(-exponent)
                    )
                )
                let angle = Float(position) * inverseFrequency
                cosValues.append(
                    Float(Foundation.cos(Double(angle)))
                )
                sinValues.append(
                    Float(Foundation.sin(Double(angle)))
                )
            }
        }

        return ForwardConstants(
            ropeCos: MLXArray(
                cosValues,
                [1, 1, sequenceLength, half]
            ).asType(dtype),
            ropeSin: MLXArray(
                sinValues,
                [1, 1, sequenceLength, half]
            ).asType(dtype),
            ropeFirstHalfIndices: MLXArray(
                (0 ..< half).map { Int32($0) }
            ),
            ropeSecondHalfIndices: MLXArray(
                (half ..< config.headDimension).map { Int32($0) }
            ),
            causalMask: MLX.tri(
                sequenceLength,
                m: sequenceLength,
                k: 0,
                dtype: .bool
            ).reshaped([
                1, 1, sequenceLength, sequenceLength
            ])
        )
    }

    private static func forward(
        tokenIDs: MLXArray,
        parameters: [MLXArray],
        config: HazeCoderConfig,
        constants: ForwardConstants
    ) -> MLXArray {
        let batchSize = tokenIDs.shape[0]
        let sequenceLength = tokenIDs.shape[1]

        var index = 0
        let embedding = parameters[index]
        index += 1

        var hidden = MLX.take(
            embedding,
            tokenIDs.reshaped([-1]),
            axis: 0
        ).reshaped([
            batchSize,
            sequenceLength,
            config.hiddenSize
        ])

        for _ in 0 ..< config.numLayers {
            let attentionNormWeight = parameters[index]
            let qWeight = parameters[index + 1]
            let kWeight = parameters[index + 2]
            let vWeight = parameters[index + 3]
            let oWeight = parameters[index + 4]
            let ffnNormWeight = parameters[index + 5]
            let gateWeight = parameters[index + 6]
            let upWeight = parameters[index + 7]
            let downWeight = parameters[index + 8]
            index += 9

            let normalizedForAttention = rmsNorm(
                hidden,
                weight: attentionNormWeight,
                epsilon: config.rmsNormEpsilon
            )

            let attentionOutput = groupedQueryAttention(
                normalizedForAttention,
                qWeight: qWeight,
                kWeight: kWeight,
                vWeight: vWeight,
                oWeight: oWeight,
                config: config,
                constants: constants
            )

            let afterAttention = hidden + attentionOutput

            let normalizedForFFN = rmsNorm(
                afterAttention,
                weight: ffnNormWeight,
                epsilon: config.rmsNormEpsilon
            )

            let gate = MLX.matmul(
                normalizedForFFN,
                gateWeight
            )
            let up = MLX.matmul(
                normalizedForFFN,
                upWeight
            )
            let siluGate = gate * MLX.sigmoid(gate)
            let ffnOutput = MLX.matmul(
                siluGate * up,
                downWeight
            )

            hidden = afterAttention + ffnOutput
        }

        let finalNormWeight = parameters[index]
        hidden = rmsNorm(
            hidden,
            weight: finalNormWeight,
            epsilon: config.rmsNormEpsilon
        )

        // Tied output projection.
        return MLX.matmul(hidden, embedding.T)
    }

    private static func rmsNorm(
        _ x: MLXArray,
        weight: MLXArray,
        epsilon: Float
    ) -> MLXArray {
        let meanSquare =
            MLX.mean(x * x, axis: -1, keepDims: true)
        return x / MLX.sqrt(meanSquare + epsilon) * weight
    }

    private static func groupedQueryAttention(
        _ x: MLXArray,
        qWeight: MLXArray,
        kWeight: MLXArray,
        vWeight: MLXArray,
        oWeight: MLXArray,
        config: HazeCoderConfig,
        constants: ForwardConstants
    ) -> MLXArray {
        let batchSize = x.shape[0]
        let sequenceLength = x.shape[1]
        let headDimension = config.headDimension

        var q = MLX.matmul(x, qWeight)
            .reshaped([
                batchSize,
                sequenceLength,
                config.numQueryHeads,
                headDimension
            ])
            .transposed(0, 2, 1, 3)

        var k = MLX.matmul(x, kWeight)
            .reshaped([
                batchSize,
                sequenceLength,
                config.numKVHeads,
                headDimension
            ])
            .transposed(0, 2, 1, 3)

        var v = MLX.matmul(x, vWeight)
            .reshaped([
                batchSize,
                sequenceLength,
                config.numKVHeads,
                headDimension
            ])
            .transposed(0, 2, 1, 3)

        q = applyRoPE(q, constants: constants)
        k = applyRoPE(k, constants: constants)

        let repeats =
            config.numQueryHeads / config.numKVHeads

        if repeats > 1 {
            k = MLX.repeated(k, count: repeats, axis: 1)
            v = MLX.repeated(v, count: repeats, axis: 1)
        }

        let scale =
            1.0 / sqrt(Float(headDimension))

        var scores =
            MLX.matmul(
                q,
                k.transposed(0, 1, 3, 2)
            ) * scale

        scores = MLX.where(
            constants.causalMask,
            scores,
            MLXArray.maskFill(for: scores.dtype)
        )

        let probabilities =
            MLX.softmax(scores, axis: -1)

        let attended =
            MLX.matmul(probabilities, v)

        let merged = attended
            .transposed(0, 2, 1, 3)
            .reshaped([
                batchSize,
                sequenceLength,
                config.hiddenSize
            ])

        return MLX.matmul(merged, oWeight)
    }

    private static func applyRoPE(
        _ x: MLXArray,
        constants: ForwardConstants
    ) -> MLXArray {
        let firstHalf = MLX.take(
            x,
            constants.ropeFirstHalfIndices,
            axis: -1
        )
        let secondHalf = MLX.take(
            x,
            constants.ropeSecondHalfIndices,
            axis: -1
        )

        let rotatedFirst =
            firstHalf * constants.ropeCos -
            secondHalf * constants.ropeSin
        let rotatedSecond =
            secondHalf * constants.ropeCos +
            firstHalf * constants.ropeSin

        return MLX.concatenated(
            [rotatedFirst, rotatedSecond],
            axis: -1
        )
    }

    // MARK: - Loss

    private static func nextTokenCrossEntropy(
        logits: MLXArray,
        targets: MLXArray,
        vocabSize: Int
    ) -> MLXArray {
        let flatLogits = logits
            .asType(.float32)
            .reshaped([-1, vocabSize])

        let flatTargets =
            targets.reshaped([-1])

        let logNormalizer =
            MLX.logSumExp(flatLogits, axis: -1)

        let targetScores = MLX.takeAlong(
            flatLogits,
            flatTargets.expandedDimensions(axis: -1),
            axis: -1
        ).squeezed(axis: -1)

        return MLX.mean(logNormalizer - targetScores)
    }

    private static func failure(
        _ message: String,
        steps: Int,
        sequenceLength: Int,
        expectedParameterCount: Int,
        started: Date,
        initialLoss: Float = .nan,
        lossHistory: [Float] = []
    ) -> HazeCoderTrainingProofResult {
        HazeCoderTrainingProofResult(
            passed: false,
            message: "ERROR — \(message)",
            steps: steps,
            sequenceLength: sequenceLength,
            parameterCount: expectedParameterCount,
            initialLoss: initialLoss,
            finalLoss: .nan,
            parameterChangeMeanSquare: .nan,
            elapsedMilliseconds:
                Date().timeIntervalSince(started) * 1000.0,
            lossHistory: lossHistory
        )
    }
}
