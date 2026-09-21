//
//  GQAAttention.swift
//  MLXMeSP
//

import Foundation
import MLX

final class HazeCoderGQAAttention {
    private let config: HazeCoderConfig
    private let qWeight: MLXArray
    private let kWeight: MLXArray
    private let vWeight: MLXArray
    private let oWeight: MLXArray
    private let rotary: HazeCoderRotaryEmbedding

    init(config: HazeCoderConfig, dtype: DType) {
        self.config = config

        let scale = 1.0 / sqrt(Float(config.hiddenSize))
        let qDimensions = config.numQueryHeads * config.headDimension
        let kvDimensions = config.numKVHeads * config.headDimension

        self.qWeight = MLXRandom.normal(
            [config.hiddenSize, qDimensions],
            dtype: dtype,
            scale: scale
        )
        self.kWeight = MLXRandom.normal(
            [config.hiddenSize, kvDimensions],
            dtype: dtype,
            scale: scale
        )
        self.vWeight = MLXRandom.normal(
            [config.hiddenSize, kvDimensions],
            dtype: dtype,
            scale: scale
        )
        self.oWeight = MLXRandom.normal(
            [qDimensions, config.hiddenSize],
            dtype: dtype,
            scale: scale
        )
        self.rotary = HazeCoderRotaryEmbedding(
            headDimension: config.headDimension,
            theta: config.ropeTheta
        )
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let batchSize = x.shape[0]
        let sequenceLength = x.shape[1]
        let headDimension = config.headDimension

        var q = MLX.matmul(x, qWeight)
            .reshaped([batchSize, sequenceLength, config.numQueryHeads, headDimension])
            .transposed(0, 2, 1, 3)

        var k = MLX.matmul(x, kWeight)
            .reshaped([batchSize, sequenceLength, config.numKVHeads, headDimension])
            .transposed(0, 2, 1, 3)

        var v = MLX.matmul(x, vWeight)
            .reshaped([batchSize, sequenceLength, config.numKVHeads, headDimension])
            .transposed(0, 2, 1, 3)

        q = rotary(q)
        k = rotary(k)

        let repeats = config.numQueryHeads / config.numKVHeads
        if repeats > 1 {
            k = MLX.repeated(k, count: repeats, axis: 1)
            v = MLX.repeated(v, count: repeats, axis: 1)
        }

        let attentionScale = 1.0 / sqrt(Float(headDimension))
        var scores = MLX.matmul(q, k.transposed(0, 1, 3, 2)) * attentionScale

        let causalMask = MLX.tri(
            sequenceLength,
            m: sequenceLength,
            k: 0,
            dtype: .bool
        ).reshaped([1, 1, sequenceLength, sequenceLength])

        scores = MLX.where(
            causalMask,
            scores,
            MLXArray.maskFill(for: scores.dtype)
        )

        let probabilities = MLX.softmax(scores, axis: -1)
        let attended = MLX.matmul(probabilities, v)

        let merged = attended
            .transposed(0, 2, 1, 3)
            .reshaped([batchSize, sequenceLength, config.hiddenSize])

        return MLX.matmul(merged, oWeight)
    }

    var parameterCount: Int {
        qWeight.size + kWeight.size + vWeight.size + oWeight.size
    }
}
