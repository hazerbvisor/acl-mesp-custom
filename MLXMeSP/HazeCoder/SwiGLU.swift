//
//  SwiGLU.swift
//  MLXMeSP
//

import Foundation
import MLX

final class HazeCoderSwiGLU {
    private let gateWeight: MLXArray
    private let upWeight: MLXArray
    private let downWeight: MLXArray

    init(hiddenSize: Int, intermediateSize: Int, dtype: DType) {
        let inputScale = 1.0 / sqrt(Float(hiddenSize))
        let outputScale = 1.0 / sqrt(Float(intermediateSize))

        self.gateWeight = MLXRandom.normal(
            [hiddenSize, intermediateSize],
            dtype: dtype,
            scale: inputScale
        )
        self.upWeight = MLXRandom.normal(
            [hiddenSize, intermediateSize],
            dtype: dtype,
            scale: inputScale
        )
        self.downWeight = MLXRandom.normal(
            [intermediateSize, hiddenSize],
            dtype: dtype,
            scale: outputScale
        )
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let gate = MLX.matmul(x, gateWeight)
        let up = MLX.matmul(x, upWeight)
        let siluGate = gate * MLX.sigmoid(gate)
        return MLX.matmul(siluGate * up, downWeight)
    }

    var parameterCount: Int {
        gateWeight.size + upWeight.size + downWeight.size
    }
}
