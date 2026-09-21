//
//  RMSNorm.swift
//  MLXMeSP
//

import MLX

final class HazeCoderRMSNorm {
    private let epsilon: Float
    let weight: MLXArray

    init(dimensions: Int, epsilon: Float, dtype: DType) {
        self.epsilon = epsilon
        self.weight = MLXArray.ones([dimensions], dtype: dtype)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let meanSquare = MLX.mean(x * x, axis: -1, keepDims: true)
        let normalized = x / MLX.sqrt(meanSquare + epsilon)
        return normalized * weight
    }

    var parameterCount: Int {
        weight.size
    }
}
