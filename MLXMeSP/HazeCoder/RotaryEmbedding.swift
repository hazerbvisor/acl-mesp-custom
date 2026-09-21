//
//  RotaryEmbedding.swift
//  MLXMeSP
//

import Foundation
import MLX

final class HazeCoderRotaryEmbedding {
    private let headDimension: Int
    private let theta: Float
    private let firstHalfIndices: MLXArray
    private let secondHalfIndices: MLXArray

    init(headDimension: Int, theta: Float) {
        self.headDimension = headDimension
        self.theta = theta

        let half = headDimension / 2
        self.firstHalfIndices = MLXArray((0 ..< half).map { Int32($0) })
        self.secondHalfIndices = MLXArray((half ..< headDimension).map { Int32($0) })
    }

    /// Applies GPT-NeoX-style rotary position embedding by pairing the
    /// first and second halves of each attention head.
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let sequenceLength = x.shape[x.ndim - 2]
        let half = headDimension / 2

        var cosValues = [Float]()
        var sinValues = [Float]()
        cosValues.reserveCapacity(sequenceLength * half)
        sinValues.reserveCapacity(sequenceLength * half)

        for position in 0 ..< sequenceLength {
            for i in 0 ..< half {
                let exponent = Float(2 * i) / Float(headDimension)
                let inverseFrequency = pow(theta, -exponent)
                let angle = Float(position) * inverseFrequency
                cosValues.append(Foundation.cos(angle))
                sinValues.append(Foundation.sin(angle))
            }
        }

        let cosTable = MLXArray(
            cosValues,
            [1, 1, sequenceLength, half]
        ).asType(x.dtype)
        let sinTable = MLXArray(
            sinValues,
            [1, 1, sequenceLength, half]
        ).asType(x.dtype)

        let firstHalf = MLX.take(x, firstHalfIndices, axis: -1)
        let secondHalf = MLX.take(x, secondHalfIndices, axis: -1)

        let rotatedFirst = firstHalf * cosTable - secondHalf * sinTable
        let rotatedSecond = secondHalf * cosTable + firstHalf * sinTable

        return MLX.concatenated([rotatedFirst, rotatedSecond], axis: -1)
    }
}
