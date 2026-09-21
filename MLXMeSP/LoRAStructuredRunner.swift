//
//  For licensing see accompanying LICENSE file.
//  Copyright (C) 2025 Apple Inc. All Rights Reserved.
//
//  LoRAStructuredRunner.swift
//  MLXMeSP
//
//  Implements LoRA-Structured Backward in Swift for memory-efficient on-device training.
//
//  Key insight: Instead of storing h = x @ A during forward, we recompute it during backward.
//  This saves memory proportional to (batch * seq * rank) per LoRA linear layer.
//
//  Gradient formulas (from idea.md):
//    grad_B = (scale * grad_output)^T @ h
//    grad_h = (scale * grad_output) @ B^T
//    grad_A = x^T @ grad_h
//    grad_x = grad_h @ A^T + grad_output @ W0
//

import Foundation
import MLX
#if canImport(MLXFast)
import MLXFast
#endif

// MARK: - Structured Backward Primitives

/// LoRA linear backward with structured gradient computation.
/// Recomputes h = x @ A instead of storing it during forward.
///
/// - Parameters:
///   - x: Input tensor [batch, seq, in_dim]
///   - loraA: LoRA A matrix [in_dim, rank]
///   - loraB: LoRA B matrix [rank, out_dim]
///   - scale: LoRA scaling factor
///   - gradOutput: Gradient from upstream [batch, seq, out_dim]
///   - w0Weight: Quantized base weight
///   - w0Scales: Quantization scales
///   - w0Biases: Quantization biases
///   - groupSize: Quantization group size
///   - bits: Quantization bits
///
/// - Returns: (grad_x, grad_A, grad_B)
public func loraLinearBackward(
    x: MLXArray,
    loraA: MLXArray,
    loraB: MLXArray,
    scale: Float,
    gradOutput: MLXArray,
    w0Weight: MLXArray,
    w0Scales: MLXArray,
    w0Biases: MLXArray,
    groupSize: Int,
    bits: Int
) -> (MLXArray, MLXArray, MLXArray) {
    // Recompute h (memory efficient - key optimization from idea.md)
    // h: [batch, seq, rank]
    let h = matmul(x, loraA)

    // Scaled gradient
    // grad_scaled: [batch, seq, out_dim]
    let gradScaled = scale * gradOutput

    // grad_B = h^T @ grad_scaled, summed over batch and seq
    // h: [batch, seq, rank] -> transpose to [batch, rank, seq]
    // grad_scaled: [batch, seq, out_dim]
    // h^T @ grad_scaled: [batch, rank, seq] @ [batch, seq, out_dim] = [batch, rank, out_dim]
    // Sum over batch: [rank, out_dim]
    let hTransposed = h.transposed(0, 2, 1)  // [batch, rank, seq]
    let gradBBatch = matmul(hTransposed, gradScaled)  // [batch, rank, out_dim]
    let gradB = sum(gradBBatch, axis: 0)  // [rank, out_dim]

    // grad_h = grad_scaled @ B^T
    // grad_h: [batch, seq, rank]
    let gradH = matmul(gradScaled, loraB.T)

    // grad_A = x^T @ grad_h, summed over batch and seq
    // x: [batch, seq, in_dim] -> transpose to [batch, in_dim, seq]
    // grad_h: [batch, seq, rank]
    // x^T @ grad_h: [batch, in_dim, seq] @ [batch, seq, rank] = [batch, in_dim, rank]
    // Sum over batch: [in_dim, rank]
    let xTransposed = x.transposed(0, 2, 1)  // [batch, in_dim, seq]
    let gradABatch = matmul(xTransposed, gradH)  // [batch, in_dim, rank]
    let gradA = sum(gradABatch, axis: 0)  // [in_dim, rank]

    // grad_x from LoRA branch
    // grad_x_lora: [batch, seq, in_dim]
    let gradXLora = matmul(gradH, loraA.T)

    // grad_x from base branch: grad_output @ W0 (using quantized matmul)
    let gradXBase = quantizedMatmul(
        gradOutput,
        w0Weight,
        scales: w0Scales,
        biases: w0Biases,
        transpose: false,
        groupSize: groupSize,
        bits: bits
    )

    let gradX = gradXLora + gradXBase

    return (gradX, gradA, gradB)
}

/// RMSNorm backward pass.
///
/// For RMSNorm: y = weight * x / norm, where norm = sqrt(mean(x^2) + eps)
///
/// grad_x = scaled_grad / norm - x * mean(scaled_grad * x) / norm^3
///
/// - Parameters:
///   - x: Input to RMSNorm [batch, seq, dim]
///   - weight: RMSNorm weight [dim]
///   - gradOutput: Gradient from upstream [batch, seq, dim]
///   - eps: Epsilon for numerical stability
///
/// - Returns: grad_x
public func rmsNormBackward(
    x: MLXArray,
    weight: MLXArray,
    gradOutput: MLXArray,
    eps: Float = 1e-6
) -> MLXArray {
    let normSq = mean(x * x, axis: -1, keepDims: true) + eps
    let norm = sqrt(normSq)

    let scaledGrad = gradOutput * weight

    // grad_x = scaled_grad / norm - x * mean(scaled_grad * x) / norm^3
    let meanTerm = mean(scaledGrad * x, axis: -1, keepDims: true)
    let gradX = scaledGrad / norm - x * meanTerm / (normSq * norm)

    return gradX
}

/// Softmax backward pass.
///
/// For softmax: y = exp(x) / sum(exp(x))
/// grad_x = y * (grad_y - sum(grad_y * y))
///
/// - Parameters:
///   - softmaxOutput: Output from softmax forward [batch, heads, seq, seq]
///   - gradOutput: Gradient from upstream [batch, heads, seq, seq]
///
/// - Returns: grad_x (gradient w.r.t. softmax input)
public func softmaxBackward(
    softmaxOutput: MLXArray,
    gradOutput: MLXArray
) -> MLXArray {
    let sumTerm = sum(gradOutput * softmaxOutput, axis: -1, keepDims: true)
    return softmaxOutput * (gradOutput - sumTerm)
}

/// SiLU backward pass.
///
/// For SiLU: y = x * sigmoid(x)
/// grad_x = grad_y * (sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x)))
///
/// - Parameters:
///   - x: Input to SiLU
///   - gradOutput: Gradient from upstream
///
/// - Returns: grad_x
public func siluBackward(
    x: MLXArray,
    gradOutput: MLXArray
) -> MLXArray {
    let sigmoidX = sigmoid(x)
    return gradOutput * (sigmoidX + x * sigmoidX * (1 - sigmoidX))
}

/// GQA backward: sum gradients over repeated heads.
///
/// During forward, K and V are repeated n_rep times for GQA.
/// During backward, we need to sum gradients over these repeated heads.
///
/// - Parameters:
///   - gradExpanded: Gradient w.r.t expanded tensor [B, n_heads, L, D]
///   - nRep: Number of repetitions (n_heads / n_kv_heads)
///
/// - Returns: grad w.r.t original tensor [B, n_kv_heads, L, D]
public func gqaBackward(
    gradExpanded: MLXArray,
    nRep: Int
) -> MLXArray {
    if nRep == 1 {
        return gradExpanded
    }
    // gradExpanded: [B, n_heads, L, D] where n_heads = n_kv_heads * n_rep
    // Reshape to [B, n_kv_heads, n_rep, L, D] and sum over n_rep axis
    let shape = gradExpanded.shape
    let B = shape[0]
    let nHeads = shape[1]
    let L = shape[2]
    let D = shape[3]
    let nKvHeads = nHeads / nRep

    // Reshape: [B, n_kv_heads, n_rep, L, D]
    let reshaped = gradExpanded.reshaped([B, nKvHeads, nRep, L, D])
    // Sum over n_rep axis
    return sum(reshaped, axis: 2)
}


// MARK: - LoRA Structured Context

/// Context for LoRA Structured training
public struct LoRAStructuredContext {
    /// Whether to enable memory debugging
    public var debugMemory: Bool

    public init(debugMemory: Bool = false) {
        self.debugMemory = debugMemory
    }
}


// MARK: - LoRA Structured Runner

/// Runner that implements LoRA-Structured Backward in Swift.
///
/// This runner:
/// 1. Uses exported forward functions (standard)
/// 2. Implements backward pass in Swift using structured gradient computation
/// 3. Achieves memory savings by not storing LoRA intermediate h = x @ A
///
/// Memory savings compared to MeBP:
/// - Each LoRA linear saves: batch * seq * rank * dtype_size bytes
/// - For 7 LoRA projections per layer: significant savings
public class LoRAStructuredRunner: BaseRunner<LoRAStructuredContext> {

    /// Trainable parameter metadata for structured backward
    private struct LoRAParam {
        let name: String
        let loraAName: String
        let loraBName: String
        let scale: Float
        let groupSize: Int
        let bits: Int
    }

    private var loraParams: [LoRAParam] = []
    private let numLayers: Int
    private let loraScale: Float

    // Attention configuration for Q/K/V backward
    private let nHeads: Int
    private let nKvHeads: Int
    private let headDim: Int
    private var nRep: Int { nHeads / nKvHeads }
    private var attnScale: Float { 1.0 / sqrt(Float(headDim)) }

    public init(
        functionPaths: [String],
        trainableParamsLoader: @escaping () throws -> [String: MLXArray],
        paramsLoaders: [() throws -> [String: MLXArray]],
        configs: [RunFunctionConfig],
        storageType: StorageType,
        numLayers: Int,
        loraScale: Float = 2.0,  // alpha / rank = 16 / 8
        // Attention config for Q/K/V backward (defaults for Qwen3-4B)
        nHeads: Int = 40,
        nKvHeads: Int = 8,
        headDim: Int = 64
    ) throws {
        self.numLayers = numLayers
        self.loraScale = loraScale
        self.nHeads = nHeads
        self.nKvHeads = nKvHeads
        self.headDim = headDim

        try super.init(
            functionPaths: functionPaths,
            trainableParamsLoader: trainableParamsLoader,
            paramsLoaders: paramsLoaders,
            configs: configs,
            storageType: storageType
        )

        // Index LoRA parameters
        for name in trainableParams.keys {
            if name.hasSuffix(".lora_a") {
                let baseName = String(name.dropLast(7))  // Remove ".lora_a"
                let loraAName = name
                let loraBName = baseName + ".lora_b"

                if trainableParams[loraBName] != nil {
                    loraParams.append(LoRAParam(
                        name: baseName,
                        loraAName: loraAName,
                        loraBName: loraBName,
                        scale: loraScale,
                        groupSize: 64,  // Default quantization params
                        bits: 4
                    ))
                }
            }
        }

        print("[LoRAStructured] Indexed \(loraParams.count) LoRA parameter pairs")
    }

    /// Run forward pass only (using exported functions)
    private func forwardPass(inputs: [String: MLXArray], debugMemory: Bool = false) async throws {
        checkpoints.removeAll()
        inputs.forEach { checkpoints[$0.key] = $0.value }

        // Run forward configs only (skip backward configs)
        for (i, config) in configs.enumerated() {
            try Task.checkCancellation()

            // Skip backward functions
            if config.functionName.contains("backward") {
                continue
            }

            let paramsLoader = paramsLoaders[i]
            guard let f = functions[config.functionName] else {
                throw MLXError.caught("Function runner not found for \(config.functionName)")
            }

            var params = try paramsLoader()
            params.merge(trainableParams) { (_, new) in new }

            let outputs = try runFunction(f, inputs: checkpoints, params: params, config: config)
            checkpoints.merge(outputs) { _, new in new }

            if debugMemory {
                let phase = "FWD"
                print("[\(String(format: "%02d", i))] \(phase) → ckpts:\(checkpoints.count)")
            }
        }
    }

    /// Compute structured backward pass in Swift for a single transformer block.
    ///
    /// This implements the full backward pass using structured LoRA gradients.
    /// Key optimization: Recomputes h = x @ A instead of storing it.
    ///
    /// - Parameters:
    ///   - layerIdx: Layer index for parameter naming
    ///   - gradOutput: Gradient from upstream (next layer or loss)
    ///   - intermediates: Saved intermediates from forward pass
    ///   - params: Frozen parameters for this layer
    ///   - debugMemory: Whether to print debug info
    ///
    /// - Returns: (gradInput, loraGrads) - gradient for previous layer and LoRA parameter gradients
    private func transformerBlockBackward(
        layerIdx: Int,
        gradOutput: MLXArray,
        intermediates: TransformerIntermediates,
        params: [String: MLXArray],
        debugMemory: Bool = false
    ) -> (MLXArray, [String: MLXArray]) {
        var grads: [String: MLXArray] = [:]
        let gradH = gradOutput

        // Helper to get param with formatted name
        func getParam(_ name: String) -> MLXArray {
            let fullName = "layers.\(layerIdx).\(name)"
            return params[fullName] ?? trainableParams[fullName]!
        }

        // === MLP Block Backward ===
        // Residual: gradH already includes gradient for residual path

        // MLP backward through down_proj (LoRA structured)
        let (gradDownInput, gradDownA, gradDownB) = loraLinearBackward(
            x: intermediates.siluOut * intermediates.upOut,  // Input to down_proj
            loraA: getParam("mlp.down_proj.lora_a"),
            loraB: getParam("mlp.down_proj.lora_b"),
            scale: loraScale,
            gradOutput: gradH,
            w0Weight: getParam("mlp.down_proj.linear.weight"),
            w0Scales: getParam("mlp.down_proj.linear.scales"),
            w0Biases: getParam("mlp.down_proj.linear.biases"),
            groupSize: 64,
            bits: 4
        )
        grads["layers.\(layerIdx).mlp.down_proj.lora_a"] = gradDownA
        grads["layers.\(layerIdx).mlp.down_proj.lora_b"] = gradDownB

        // Backward through silu * up multiplication
        let gradSiluOut = gradDownInput * intermediates.upOut
        let gradUpOut = gradDownInput * intermediates.siluOut

        // SiLU backward
        let gradGateOut = siluBackward(x: intermediates.gateOut, gradOutput: gradSiluOut)

        // Gate proj backward (LoRA structured)
        let (gradGateInput, gradGateA, gradGateB) = loraLinearBackward(
            x: intermediates.normed2,
            loraA: getParam("mlp.gate_proj.lora_a"),
            loraB: getParam("mlp.gate_proj.lora_b"),
            scale: loraScale,
            gradOutput: gradGateOut,
            w0Weight: getParam("mlp.gate_proj.linear.weight"),
            w0Scales: getParam("mlp.gate_proj.linear.scales"),
            w0Biases: getParam("mlp.gate_proj.linear.biases"),
            groupSize: 64,
            bits: 4
        )
        grads["layers.\(layerIdx).mlp.gate_proj.lora_a"] = gradGateA
        grads["layers.\(layerIdx).mlp.gate_proj.lora_b"] = gradGateB

        // Up proj backward (LoRA structured)
        let (gradUpInput, gradUpA, gradUpB) = loraLinearBackward(
            x: intermediates.normed2,
            loraA: getParam("mlp.up_proj.lora_a"),
            loraB: getParam("mlp.up_proj.lora_b"),
            scale: loraScale,
            gradOutput: gradUpOut,
            w0Weight: getParam("mlp.up_proj.linear.weight"),
            w0Scales: getParam("mlp.up_proj.linear.scales"),
            w0Biases: getParam("mlp.up_proj.linear.biases"),
            groupSize: 64,
            bits: 4
        )
        grads["layers.\(layerIdx).mlp.up_proj.lora_a"] = gradUpA
        grads["layers.\(layerIdx).mlp.up_proj.lora_b"] = gradUpB

        // Combine gate and up gradients
        let gradNormed2 = gradGateInput + gradUpInput

        // RMSNorm backward for post_attention_layernorm
        let gradHAfterAttn = gradH + rmsNormBackward(
            x: intermediates.hAfterAttn,
            weight: getParam("post_attention_layernorm.weight"),
            gradOutput: gradNormed2
        )

        // === Attention Block Backward ===
        // O projection backward (LoRA structured)
        let (gradAttnOut, gradOA, gradOB) = loraLinearBackward(
            x: intermediates.attnOutput,
            loraA: getParam("self_attn.o_proj.lora_a"),
            loraB: getParam("self_attn.o_proj.lora_b"),
            scale: loraScale,
            gradOutput: gradHAfterAttn,
            w0Weight: getParam("self_attn.o_proj.linear.weight"),
            w0Scales: getParam("self_attn.o_proj.linear.scales"),
            w0Biases: getParam("self_attn.o_proj.linear.biases"),
            groupSize: 64,
            bits: 4
        )
        grads["layers.\(layerIdx).self_attn.o_proj.lora_a"] = gradOA
        grads["layers.\(layerIdx).self_attn.o_proj.lora_b"] = gradOB

        // === Full Q/K/V Backward ===
        // Step 1: Reshape gradAttnOut from [B, L, H*D] to [B, H, L, D]
        let attnOutShape = intermediates.attnOutput.shape  // [B, L, H*D]
        let B = attnOutShape[0]
        let L = attnOutShape[1]
        let gradAttnOutReshaped = gradAttnOut.reshaped([B, L, nHeads, headDim]).transposed(0, 2, 1, 3)
        // gradAttnOutReshaped: [B, H, L, D]

        // Step 2: Recompute Q, K, V from normed1
        // Q: [B, L, H*D], K: [B, L, Hkv*D], V: [B, L, Hkv*D]
        let q = quantizedMatmul(
            intermediates.normed1,
            getParam("self_attn.q_proj.linear.weight"),
            scales: getParam("self_attn.q_proj.linear.scales"),
            biases: getParam("self_attn.q_proj.linear.biases"),
            transpose: true,
            groupSize: 64,
            bits: 4
        ) + matmul(matmul(intermediates.normed1, getParam("self_attn.q_proj.lora_a")), getParam("self_attn.q_proj.lora_b")) * loraScale

        let k = quantizedMatmul(
            intermediates.normed1,
            getParam("self_attn.k_proj.linear.weight"),
            scales: getParam("self_attn.k_proj.linear.scales"),
            biases: getParam("self_attn.k_proj.linear.biases"),
            transpose: true,
            groupSize: 64,
            bits: 4
        ) + matmul(matmul(intermediates.normed1, getParam("self_attn.k_proj.lora_a")), getParam("self_attn.k_proj.lora_b")) * loraScale

        let v = quantizedMatmul(
            intermediates.normed1,
            getParam("self_attn.v_proj.linear.weight"),
            scales: getParam("self_attn.v_proj.linear.scales"),
            biases: getParam("self_attn.v_proj.linear.biases"),
            transpose: true,
            groupSize: 64,
            bits: 4
        ) + matmul(matmul(intermediates.normed1, getParam("self_attn.v_proj.lora_a")), getParam("self_attn.v_proj.lora_b")) * loraScale

        // Reshape to multi-head format
        let qReshaped = q.reshaped([B, L, nHeads, headDim]).transposed(0, 2, 1, 3)  // [B, H, L, D]
        let kReshaped = k.reshaped([B, L, nKvHeads, headDim]).transposed(0, 2, 1, 3)  // [B, Hkv, L, D]
        let vReshaped = v.reshaped([B, L, nKvHeads, headDim]).transposed(0, 2, 1, 3)  // [B, Hkv, L, D]

        // GQA expansion for K and V
        var kExpanded = kReshaped
        var vExpanded = vReshaped
        if nRep > 1 {
            kExpanded = MLX.repeated(kReshaped, count: nRep, axis: 1)  // [B, H, L, D]
            vExpanded = MLX.repeated(vReshaped, count: nRep, axis: 1)  // [B, H, L, D]
        }

        // Step 3: Attention backward
        // attn_output = attn_weights @ V_expanded
        // grad_attn_weights = grad_attn_output @ V_expanded^T
        // grad_V_expanded = attn_weights^T @ grad_attn_output
        let gradAttnWeights = matmul(gradAttnOutReshaped, vExpanded.transposed(0, 1, 3, 2))  // [B, H, L, L]
        let gradVExpanded = matmul(intermediates.attnWeights.transposed(0, 1, 3, 2), gradAttnOutReshaped)  // [B, H, L, D]

        // Step 4: Softmax backward
        let gradScores = softmaxBackward(softmaxOutput: intermediates.attnWeights, gradOutput: gradAttnWeights)

        // Step 5: Scores backward (scores = Q @ K^T * scale)
        // grad_Q = grad_scores @ K * scale
        // grad_K = grad_scores^T @ Q * scale (need to transpose properly)
        let gradQ = matmul(gradScores, kExpanded) * attnScale  // [B, H, L, D]
        let gradKExpanded = matmul(gradScores.transposed(0, 1, 3, 2), qReshaped) * attnScale  // [B, H, L, D]

        // Step 6: GQA backward - sum over repeated heads for K and V
        let gradK = gqaBackward(gradExpanded: gradKExpanded, nRep: nRep)  // [B, Hkv, L, D]
        let gradV = gqaBackward(gradExpanded: gradVExpanded, nRep: nRep)  // [B, Hkv, L, D]

        // Step 7: Reshape gradients back to [B, L, dim]
        let gradQFlat = gradQ.transposed(0, 2, 1, 3).reshaped([B, L, nHeads * headDim])  // [B, L, H*D]
        let gradKFlat = gradK.transposed(0, 2, 1, 3).reshaped([B, L, nKvHeads * headDim])  // [B, L, Hkv*D]
        let gradVFlat = gradV.transposed(0, 2, 1, 3).reshaped([B, L, nKvHeads * headDim])  // [B, L, Hkv*D]

        // Step 8: Q/K/V projection backward (LoRA structured)
        let (gradQInput, gradQA, gradQB) = loraLinearBackward(
            x: intermediates.normed1,
            loraA: getParam("self_attn.q_proj.lora_a"),
            loraB: getParam("self_attn.q_proj.lora_b"),
            scale: loraScale,
            gradOutput: gradQFlat,
            w0Weight: getParam("self_attn.q_proj.linear.weight"),
            w0Scales: getParam("self_attn.q_proj.linear.scales"),
            w0Biases: getParam("self_attn.q_proj.linear.biases"),
            groupSize: 64,
            bits: 4
        )
        grads["layers.\(layerIdx).self_attn.q_proj.lora_a"] = gradQA
        grads["layers.\(layerIdx).self_attn.q_proj.lora_b"] = gradQB

        let (gradKInput, gradKA, gradKB) = loraLinearBackward(
            x: intermediates.normed1,
            loraA: getParam("self_attn.k_proj.lora_a"),
            loraB: getParam("self_attn.k_proj.lora_b"),
            scale: loraScale,
            gradOutput: gradKFlat,
            w0Weight: getParam("self_attn.k_proj.linear.weight"),
            w0Scales: getParam("self_attn.k_proj.linear.scales"),
            w0Biases: getParam("self_attn.k_proj.linear.biases"),
            groupSize: 64,
            bits: 4
        )
        grads["layers.\(layerIdx).self_attn.k_proj.lora_a"] = gradKA
        grads["layers.\(layerIdx).self_attn.k_proj.lora_b"] = gradKB

        let (gradVInput, gradVA, gradVB) = loraLinearBackward(
            x: intermediates.normed1,
            loraA: getParam("self_attn.v_proj.lora_a"),
            loraB: getParam("self_attn.v_proj.lora_b"),
            scale: loraScale,
            gradOutput: gradVFlat,
            w0Weight: getParam("self_attn.v_proj.linear.weight"),
            w0Scales: getParam("self_attn.v_proj.linear.scales"),
            w0Biases: getParam("self_attn.v_proj.linear.biases"),
            groupSize: 64,
            bits: 4
        )
        grads["layers.\(layerIdx).self_attn.v_proj.lora_a"] = gradVA
        grads["layers.\(layerIdx).self_attn.v_proj.lora_b"] = gradVB

        // Step 9: Combine Q/K/V input gradients
        let gradNormed1 = gradQInput + gradKInput + gradVInput

        // Step 10: RMSNorm backward for input_layernorm
        let gradXFromAttn = rmsNormBackward(
            x: intermediates.xSaved,
            weight: getParam("input_layernorm.weight"),
            gradOutput: gradNormed1
        )

        // Final gradient: residual from MLP + gradient from attention
        let gradX = gradHAfterAttn + gradXFromAttn

        if debugMemory {
            print("[LoRAStructured] Layer \(layerIdx) backward: \(grads.count) LoRA grads computed")
        }

        return (gradX, grads)
    }

    /// Intermediate values saved during forward for backward computation.
    /// FULL: 8 intermediates for complete Q/K/V backward
    private struct TransformerIntermediates {
        let xSaved: MLXArray       // Layer input (for residual gradient)
        let normed1: MLXArray      // After input_layernorm (for Q/K/V projection backward)
        let attnWeights: MLXArray  // Attention weights [B, heads, seq, seq] (for attention backward)
        let attnOutput: MLXArray   // After attention (for O proj backward)
        let hAfterAttn: MLXArray   // After attention + residual (for MLP input)
        let normed2: MLXArray      // After post_attention_layernorm (for MLP backward)
        let gateOut: MLXArray      // Gate projection output (for SiLU backward)
        let upOut: MLXArray        // Up projection output (for MLP backward)

        // OPTIMIZATION: siluOut is recomputed from gateOut to save memory
        var siluOut: MLXArray {
            return MLX.sigmoid(gateOut) * gateOut  // silu(x) = x * sigmoid(x)
        }
    }

    /// Parse intermediates from checkpoints for a given layer.
    /// FULL: 8 intermediates for complete Q/K/V backward
    /// The checkpoint keys use "layer{idx}" prefix (set by module_name in export.py).
    private func getIntermediates(layerIdx: Int) -> TransformerIntermediates? {
        let prefix = "layer\(layerIdx)"

        guard
            let xSaved = checkpoints["\(prefix).x_saved"],
            let normed1 = checkpoints["\(prefix).normed1"],
            let attnWeights = checkpoints["\(prefix).attn_weights"],
            let attnOutput = checkpoints["\(prefix).attn_output"],
            let hAfterAttn = checkpoints["\(prefix).h_after_attn"],
            let normed2 = checkpoints["\(prefix).normed2"],
            let gateOut = checkpoints["\(prefix).gate_out"],
            let upOut = checkpoints["\(prefix).up_out"]
        else {
            return nil
        }

        // FULL: 8 intermediates for complete Q/K/V backward
        return TransformerIntermediates(
            xSaved: xSaved,
            normed1: normed1,
            attnWeights: attnWeights,
            attnOutput: attnOutput,
            hAfterAttn: hAfterAttn,
            normed2: normed2,
            gateOut: gateOut,
            upOut: upOut
        )
    }

    /// Clean up intermediates for a specific layer after backward pass.
    /// FULL: 8 arrays to remove for complete Q/K/V backward
    private func cleanupLayerIntermediates(layerIdx: Int) {
        let prefix = "layer\(layerIdx)"
        let keysToRemove = [
            "\(prefix).x_saved",
            "\(prefix).normed1",
            "\(prefix).attn_weights",
            "\(prefix).attn_output",
            "\(prefix).h_after_attn",
            "\(prefix).normed2",
            "\(prefix).gate_out",
            "\(prefix).up_out",
        ]
        for key in keysToRemove {
            _ = checkpoints.removeValue(forKey: key)
        }
    }

    /// Compute structured backward pass in Swift for all layers.
    private func structuredBackward(debugMemory: Bool = false) throws -> [String: MLXArray] {
        var grads: [String: MLXArray] = [:]

        // Get loss gradient (should be computed from loss function)
        guard let lossGradName = configs.last?.outputNames.first(where: { $0.contains(".grad") }),
              let lossGrad = checkpoints[lossGradName] else {
            print("[LoRAStructured] Warning: No loss gradient found in checkpoints")
            return grads
        }

        var gradH = lossGrad

        // Iterate through layers in reverse order
        for layerIdx in stride(from: numLayers - 1, through: 0, by: -1) {
            // Get intermediates for this layer
            guard let intermediates = getIntermediates(layerIdx: layerIdx) else {
                print("[LoRAStructured] Warning: Missing intermediates for layer \(layerIdx)")
                continue
            }

            // Get frozen parameters for this layer
            let paramsLoader = paramsLoaders[layerIdx + 1]  // +1 for embedding
            let params = try paramsLoader()

            // Compute structured backward
            let (gradInput, layerGrads) = transformerBlockBackward(
                layerIdx: layerIdx,
                gradOutput: gradH,
                intermediates: intermediates,
                params: params,
                debugMemory: debugMemory
            )

            // Accumulate gradients
            grads.merge(layerGrads) { _, new in new }

            // Pass gradient to previous layer
            gradH = gradInput

            // CRITICAL: Clean up this layer's intermediates to free memory
            cleanupLayerIntermediates(layerIdx: layerIdx)

            // Eval to release memory and trigger cleanup
            eval(gradH)

            if debugMemory {
                let memMB = Double(GPU.activeMemory) / (1024 * 1024)
                print("[LoRAStructured] Layer \(layerIdx) backward complete, mem: \(String(format: "%.0f", memMB))MB")
            }
        }

        return grads
    }

    /// Whether to use Swift structured backward (true) or fall back to exported backward (false)
    private var useSwiftBackward: Bool = false

    /// Whether to use gradient checkpointing (recompute intermediates during backward)
    private var useGradientCheckpointing: Bool = true

    override public func gradients(
        inputs: [String: MLXArray],
        metricNames: [String],
        context: LoRAStructuredContext
    ) async throws -> (grads: [String: MLXArray], metrics: [String: MLXArray]) {
        var grads: [String: MLXArray] = [:]

        if useSwiftBackward {
            if useGradientCheckpointing {
                // GRADIENT CHECKPOINTING MODE
                // Forward: only store layer outputs (not intermediates)
                // Backward: recompute intermediates for each layer
                try await forwardPassCheckpointed(inputs: inputs, debugMemory: context.debugMemory)

                let metrics = getMetrics(metricsName: metricNames)

                // Compute backward with recomputation
                grads = try await structuredBackwardCheckpointed(debugMemory: context.debugMemory)

                checkpoints.removeAll()
                return (grads: grads, metrics: metrics)
            } else {
                // STANDARD MODE: Store all intermediates during forward
                try await forwardPassWithIntermediates(inputs: inputs, debugMemory: context.debugMemory)
                let metrics = getMetrics(metricsName: metricNames)
                grads = try structuredBackward(debugMemory: context.debugMemory)
                checkpoints.removeAll()
                return (grads: grads, metrics: metrics)
            }
        } else {
            // Fall back to exported backward functions (standard MeBP approach)
            try await step(inputs: inputs, debugMemory: context.debugMemory)

            let metrics = getMetrics(metricsName: metricNames)

            // Extract gradients from checkpoints
            for name in trainableParams.keys {
                if let grad = checkpoints[getGradName(name)] {
                    grads[name] = grad
                }
            }

            checkpoints.removeAll()
            return (grads: grads, metrics: metrics)
        }
    }

    // MARK: - Gradient Checkpointing Implementation

    /// Forward pass with gradient checkpointing.
    /// Only stores layer OUTPUTS (hidden_states), not intermediates.
    /// Memory: O(L) for layer outputs instead of O(L * intermediates).
    private func forwardPassCheckpointed(inputs: [String: MLXArray], debugMemory: Bool = false) async throws {
        checkpoints.removeAll()
        inputs.forEach { checkpoints[$0.key] = $0.value }

        for (i, config) in configs.enumerated() {
            try Task.checkCancellation()

            if config.functionName.contains("backward") {
                continue
            }

            let paramsLoader = paramsLoaders[i]
            guard let f = functions[config.functionName] else {
                throw MLXError.caught("Function runner not found for \(config.functionName)")
            }

            var params = try paramsLoader()
            params.merge(trainableParams) { (_, new) in new }

            let outputs = try runFunction(f, inputs: checkpoints, params: params, config: config)

            // GRADIENT CHECKPOINTING: Only keep the FIRST output (layer output / hidden_states)
            // DO NOT store x_saved - we derive it from previous layer's output during backward
            if config.functionName.contains("_forward_swift") {
                // For transformer layers: only keep y (first output = hidden_states)
                if let yName = config.outputNames.first {
                    let yValue = outputs[yName]!
                    eval(yValue)  // Force evaluation
                    checkpoints[yName] = yValue
                }
                // Intermediates are NOT stored - will be recomputed during backward
            } else {
                // For embedding/loss: keep all outputs
                for (key, value) in outputs {
                    eval(value)
                    checkpoints[key] = value
                }
            }

            // Clear GPU cache after each layer
            GPU.clearCache()

            if debugMemory {
                let memMB = Double(GPU.activeMemory) / (1024 * 1024)
                print("[\(String(format: "%02d", i))] FWD-CKPT → ckpts:\(checkpoints.count) mem:\(String(format: "%.0f", memMB))MB")
            }
        }
    }

    /// Recompute intermediates for a single layer by re-running its forward pass.
    /// This is the key to gradient checkpointing - trade compute for memory.
    private func recomputeLayerIntermediates(layerIdx: Int, layerInput: MLXArray) throws -> TransformerIntermediates? {
        // Find the config for this layer's forward function
        let functionName = "layer\(layerIdx)_forward_swift"
        guard let f = functions[functionName] else {
            print("[GradientCheckpoint] Warning: Function not found: \(functionName)")
            return nil
        }

        // Find the config index for this layer
        guard let configIdx = configs.firstIndex(where: { $0.functionName == functionName }) else {
            print("[GradientCheckpoint] Warning: Config not found for \(functionName)")
            return nil
        }

        let config = configs[configIdx]
        let paramsLoader = paramsLoaders[configIdx]

        // Prepare inputs - layer input goes to the x input
        var recomputeInputs: [String: MLXArray] = [:]
        if let xInputName = config.inputNames.first {
            recomputeInputs[xInputName] = layerInput
        }

        // Load parameters
        var params = try paramsLoader()
        params.merge(trainableParams) { (_, new) in new }

        // Re-run forward to get intermediates
        let outputs = try runFunction(f, inputs: recomputeInputs, params: params, config: config)

        // Parse intermediates from outputs
        let prefix = "layer\(layerIdx)"
        guard
            let xSaved = outputs["\(prefix).x_saved"],
            let normed1 = outputs["\(prefix).normed1"],
            let attnWeights = outputs["\(prefix).attn_weights"],
            let attnOutput = outputs["\(prefix).attn_output"],
            let hAfterAttn = outputs["\(prefix).h_after_attn"],
            let normed2 = outputs["\(prefix).normed2"],
            let gateOut = outputs["\(prefix).gate_out"],
            let upOut = outputs["\(prefix).up_out"]
        else {
            print("[GradientCheckpoint] Warning: Missing outputs for layer \(layerIdx)")
            return nil
        }

        return TransformerIntermediates(
            xSaved: xSaved,
            normed1: normed1,
            attnWeights: attnWeights,
            attnOutput: attnOutput,
            hAfterAttn: hAfterAttn,
            normed2: normed2,
            gateOut: gateOut,
            upOut: upOut
        )
    }

    /// Backward pass with gradient checkpointing.
    /// Recomputes intermediates for each layer instead of using stored values.
    private func structuredBackwardCheckpointed(debugMemory: Bool = false) async throws -> [String: MLXArray] {
        var grads: [String: MLXArray] = [:]

        // Get loss gradient
        guard let lossGradName = configs.last?.outputNames.first(where: { $0.contains(".grad") }),
              let lossGrad = checkpoints[lossGradName] else {
            print("[GradientCheckpoint] Warning: No loss gradient found")
            return grads
        }

        var gradH = lossGrad
        eval(gradH)

        for layerIdx in stride(from: numLayers - 1, through: 0, by: -1) {
            try Task.checkCancellation()

            // Step 1: Get layer input from stored checkpoints
            // Layer i's input = layer (i-1)'s output, or input_embeds for layer 0
            let layerInputName: String
            if layerIdx == 0 {
                layerInputName = "input_embeds"
            } else {
                layerInputName = "layer\(layerIdx - 1).hidden_states"
            }

            guard let layerInput = checkpoints[layerInputName] else {
                print("[GradientCheckpoint] Warning: Missing input for layer \(layerIdx): \(layerInputName)")
                continue
            }

            // Step 2: RECOMPUTE intermediates by re-running layer forward
            guard let intermediates = try recomputeLayerIntermediates(layerIdx: layerIdx, layerInput: layerInput) else {
                print("[GradientCheckpoint] Warning: Failed to recompute intermediates for layer \(layerIdx)")
                continue
            }

            // Step 3: Get frozen parameters for backward
            let paramsLoader = paramsLoaders[layerIdx + 1]  // +1 for embedding
            let params = try paramsLoader()

            // Step 4: Compute structured backward
            let (gradInput, layerGrads) = transformerBlockBackward(
                layerIdx: layerIdx,
                gradOutput: gradH,
                intermediates: intermediates,
                params: params,
                debugMemory: debugMemory
            )

            // Step 5: Accumulate gradients and eval immediately
            for (key, grad) in layerGrads {
                eval(grad)
                grads[key] = grad
            }

            // Step 6: Pass gradient to previous layer
            gradH = gradInput
            eval(gradH)

            // Step 7: Clean up this layer's output (no longer needed for backward)
            _ = checkpoints.removeValue(forKey: "layer\(layerIdx).hidden_states")

            // Clear GPU cache to release memory
            GPU.clearCache()

            if debugMemory {
                let memMB = Double(GPU.activeMemory) / (1024 * 1024)
                print("[GradientCheckpoint] Layer \(layerIdx) backward complete, mem: \(String(format: "%.0f", memMB))MB")
            }
        }

        return grads
    }

    /// Run forward pass with intermediate outputs for Swift backward (non-checkpointed).
    private func forwardPassWithIntermediates(inputs: [String: MLXArray], debugMemory: Bool = false) async throws {
        checkpoints.removeAll()
        inputs.forEach { checkpoints[$0.key] = $0.value }

        for (i, config) in configs.enumerated() {
            try Task.checkCancellation()

            if config.functionName.contains("backward") {
                continue
            }

            let isSwiftForward = config.functionName.contains("_forward_swift")

            let paramsLoader = paramsLoaders[i]
            guard let f = functions[config.functionName] else {
                throw MLXError.caught("Function runner not found for \(config.functionName)")
            }

            var params = try paramsLoader()
            params.merge(trainableParams) { (_, new) in new }

            let outputs = try runFunction(f, inputs: checkpoints, params: params, config: config)
            checkpoints.merge(outputs) { _, new in new }

            if debugMemory {
                let phase = isSwiftForward ? "FWD-SWIFT" : "FWD"
                let memMB = Double(GPU.activeMemory) / (1024 * 1024)
                print("[\(String(format: "%02d", i))] \(phase) → ckpts:\(checkpoints.count) mem:\(String(format: "%.0f", memMB))MB")
            }
        }
    }

    /// Enable Swift structured backward mode.
    ///
    /// Call this after loading forward_swift functions that output intermediates.
    public func enableSwiftBackward() {
        useSwiftBackward = true
        print("[LoRAStructured] Swift structured backward enabled")
    }

    /// Disable Swift structured backward mode (use exported backward functions).
    public func disableSwiftBackward() {
        useSwiftBackward = false
        print("[LoRAStructured] Swift structured backward disabled, using exported backward")
    }

    /// Convenience run method
    public func run(
        batchedInputs: [[String: MLXArray]],
        metricNames: [String],
        learningRate: Float,
        numSteps: Int,
        numAccumulationSteps: Int,
        onIterationStart: IterationStartHook? = nil,
        onIterationEnd: IterationEndHook? = nil,
        verbose: Bool = false,
        debugMemory: Bool = false
    ) async throws -> [[String: MLXArray]] {
        try await run(
            batchedInputs: batchedInputs,
            metricNames: metricNames,
            learningRate: learningRate,
            numSteps: numSteps,
            numAccumulationSteps: numAccumulationSteps,
            gradientsContext: LoRAStructuredContext(debugMemory: debugMemory),
            onIterationStart: onIterationStart,
            onIterationEnd: onIterationEnd,
            verbose: verbose
        )
    }
}
