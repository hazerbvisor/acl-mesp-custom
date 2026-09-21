//
//  HazeCoderConfig.swift
//  MLXMeSP
//
//  HazeCoder Phase 1: tiny from-scratch coding model configuration.
//

import Foundation

public enum HazeCoderConfigError: Error, CustomStringConvertible {
    case invalid(String)

    public var description: String {
        switch self {
        case .invalid(let message):
            return message
        }
    }
}

public struct HazeCoderConfig: Sendable {
    public let vocabSize: Int
    public let maxSequenceLength: Int
    public let numLayers: Int
    public let hiddenSize: Int
    public let numQueryHeads: Int
    public let numKVHeads: Int
    public let headDimension: Int
    public let intermediateSize: Int
    public let rmsNormEpsilon: Float
    public let ropeTheta: Float
    public let seed: UInt64
    public let useBFloat16: Bool

    public init(
        vocabSize: Int = 4096,
        maxSequenceLength: Int = 256,
        numLayers: Int = 5,
        hiddenSize: Int = 384,
        numQueryHeads: Int = 6,
        numKVHeads: Int = 2,
        headDimension: Int = 64,
        intermediateSize: Int = 1024,
        rmsNormEpsilon: Float = 1e-5,
        ropeTheta: Float = 10_000,
        seed: UInt64 = 42,
        useBFloat16: Bool = true
    ) {
        self.vocabSize = vocabSize
        self.maxSequenceLength = maxSequenceLength
        self.numLayers = numLayers
        self.hiddenSize = hiddenSize
        self.numQueryHeads = numQueryHeads
        self.numKVHeads = numKVHeads
        self.headDimension = headDimension
        self.intermediateSize = intermediateSize
        self.rmsNormEpsilon = rmsNormEpsilon
        self.ropeTheta = ropeTheta
        self.seed = seed
        self.useBFloat16 = useBFloat16
    }

    public static let nano = HazeCoderConfig()

    public func validate() throws {
        guard vocabSize > 0 else {
            throw HazeCoderConfigError.invalid("vocabSize must be positive")
        }
        guard maxSequenceLength > 0 else {
            throw HazeCoderConfigError.invalid("maxSequenceLength must be positive")
        }
        guard numLayers > 0 else {
            throw HazeCoderConfigError.invalid("numLayers must be positive")
        }
        guard hiddenSize == numQueryHeads * headDimension else {
            throw HazeCoderConfigError.invalid(
                "hiddenSize must equal numQueryHeads × headDimension"
            )
        }
        guard numQueryHeads % numKVHeads == 0 else {
            throw HazeCoderConfigError.invalid(
                "numQueryHeads must be divisible by numKVHeads for GQA"
            )
        }
        guard headDimension % 2 == 0 else {
            throw HazeCoderConfigError.invalid(
                "headDimension must be even for RoPE"
            )
        }
        guard intermediateSize > 0 else {
            throw HazeCoderConfigError.invalid("intermediateSize must be positive")
        }
    }

    /// Expected trainable parameter count for the dense Nano architecture.
    /// The LM head is tied to the token embedding and therefore is not counted twice.
    public var estimatedParameterCount: Int {
        let embedding = vocabSize * hiddenSize

        let attentionPerLayer =
            hiddenSize * (numQueryHeads * headDimension) +
            hiddenSize * (numKVHeads * headDimension) +
            hiddenSize * (numKVHeads * headDimension) +
            hiddenSize * hiddenSize

        let swigluPerLayer =
            hiddenSize * intermediateSize +
            hiddenSize * intermediateSize +
            intermediateSize * hiddenSize

        let normsPerLayer = 2 * hiddenSize
        let block = attentionPerLayer + swigluPerLayer + normsPerLayer
        let finalNorm = hiddenSize

        return embedding + numLayers * block + finalNorm
    }
}
