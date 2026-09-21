//
//  TransformerBlock.swift
//  MLXMeSP
//

import MLX

final class HazeCoderTransformerBlock {
    private let attentionNorm: HazeCoderRMSNorm
    private let attention: HazeCoderGQAAttention
    private let ffnNorm: HazeCoderRMSNorm
    private let ffn: HazeCoderSwiGLU

    init(config: HazeCoderConfig, dtype: DType) {
        self.attentionNorm = HazeCoderRMSNorm(
            dimensions: config.hiddenSize,
            epsilon: config.rmsNormEpsilon,
            dtype: dtype
        )
        self.attention = HazeCoderGQAAttention(config: config, dtype: dtype)
        self.ffnNorm = HazeCoderRMSNorm(
            dimensions: config.hiddenSize,
            epsilon: config.rmsNormEpsilon,
            dtype: dtype
        )
        self.ffn = HazeCoderSwiGLU(
            hiddenSize: config.hiddenSize,
            intermediateSize: config.intermediateSize,
            dtype: dtype
        )
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let afterAttention = x + attention(attentionNorm(x))
        return afterAttention + ffn(ffnNorm(afterAttention))
    }

    var parameterCount: Int {
        attentionNorm.parameterCount +
        attention.parameterCount +
        ffnNorm.parameterCount +
        ffn.parameterCount
    }
}
