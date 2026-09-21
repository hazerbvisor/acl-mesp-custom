// Vendored from apple/swift-numerics 1.1.1; see ../LICENSE.txt

// ===== Sources/RealModule/Float16+Real.swift =====
//===--- Float16+Real.swift -----------------------------------*- swift -*-===//
//
// This source file is part of the Swift Numerics open source project
//
// Copyright (c) 2020 Apple Inc. and the Swift Numerics project authors
// Licensed under Apache License v2.0 with Runtime Library Exception
//
// See https://swift.org/LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

import _NumericsShims

// Float16 is only available on macOS when targeting arm64.
#if !((os(macOS) || targetEnvironment(macCatalyst)) && arch(x86_64))

@available(macOS 11.0, iOS 14.0, tvOS 14.0, watchOS 7.0, *)
extension Float16: Real {
  @_transparent
  public static func cos(_ x: Float16) -> Float16 {
    Float16(.cos(Float(x)))
  }
  
  @_transparent
  public static func sin(_ x: Float16) -> Float16 {
    Float16(.sin(Float(x)))
  }
  
  @_transparent
  public static func tan(_ x: Float16) -> Float16 {
    Float16(.tan(Float(x)))
  }
  
  @_transparent
  public static func acos(_ x: Float16) -> Float16 {
    Float16(.acos(Float(x)))
  }
  
  @_transparent
  public static func asin(_ x: Float16) -> Float16 {
    Float16(.asin(Float(x)))
  }
  
  @_transparent
  public static func atan(_ x: Float16) -> Float16 {
    Float16(.atan(Float(x)))
  }
  
  @_transparent
  public static func cosh(_ x: Float16) -> Float16 {
    Float16(.cosh(Float(x)))
  }
  
  @_transparent
  public static func sinh(_ x: Float16) -> Float16 {
    Float16(.sinh(Float(x)))
  }
  
  @_transparent
  public static func tanh(_ x: Float16) -> Float16 {
    Float16(.tanh(Float(x)))
  }
  
  @_transparent
  public static func acosh(_ x: Float16) -> Float16 {
    Float16(.acosh(Float(x)))
  }
  
  @_transparent
  public static func asinh(_ x: Float16) -> Float16 {
    Float16(.asinh(Float(x)))
  }
  
  @_transparent
  public static func atanh(_ x: Float16) -> Float16 {
    Float16(.atanh(Float(x)))
  }
  
  @_transparent
  public static func exp(_ x: Float16) -> Float16 {
    Float16(.exp(Float(x)))
  }
  
  @_transparent
  public static func expMinusOne(_ x: Float16) -> Float16 {
    Float16(.expMinusOne(Float(x)))
  }
  
  @_transparent
  public static func log(_ x: Float16) -> Float16 {
    Float16(.log(Float(x)))
  }
  
  @_transparent
  public static func log(onePlus x: Float16) -> Float16 {
    Float16(.log(onePlus: Float(x)))
  }
  
  @_transparent
  public static func erf(_ x: Float16) -> Float16 {
    Float16(.erf(Float(x)))
  }
  
  @_transparent
  public static func erfc(_ x: Float16) -> Float16 {
    Float16(.erfc(Float(x)))
  }
  
  @_transparent
  public static func exp2(_ x: Float16) -> Float16 {
    Float16(.exp2(Float(x)))
  }
  
  @_transparent
  public static func exp10(_ x: Float16) -> Float16 {
    Float16(.exp10(Float(x)))
  }
  
  @_transparent
  public static func hypot(_ x: Float16, _ y: Float16) -> Float16 {
    if x.isInfinite || y.isInfinite { return .infinity }
    let xf = Float(x)
    let yf = Float(y)
    return Float16(.sqrt(xf*xf + yf*yf))
  }
  
  @_transparent
  public static func gamma(_ x: Float16) -> Float16 {
    Float16(.gamma(Float(x)))
  }
  
  @_transparent
  public static func log2(_ x: Float16) -> Float16 {
    Float16(.log2(Float(x)))
  }
  
  @_transparent
  public static func log10(_ x: Float16) -> Float16 {
    Float16(.log10(Float(x)))
  }
  
  @_transparent
  public static func pow(_ x: Float16, _ y: Float16) -> Float16 {
    Float16(.pow(Float(x), Float(y)))
  }
  
  @_transparent
  public static func pow(_ x: Float16, _ n: Int) -> Float16 {
    // Float16 is simpler than Float or Double, because the range of
    // "interesting" exponents is pretty small; anything outside of
    // -22707 ... 34061 simply overflows or underflows for every
    // x that isn't zero or one. This whole range is representable
    // as Float, so we can just use powf as long as we're a little
    // bit (get it?) careful to preserve parity.
    let clamped = min(max(n, -0x10000), 0x10000) | (n & 1)
    return Float16(libm_powf(Float(x), Float(clamped)))
  }
  
  @_transparent
  public static func root(_ x: Float16, _ n: Int) -> Float16 {
    Float16(.root(Float(x), n))
  }
  
  @_transparent
  public static func atan2(y: Float16, x: Float16) -> Float16 {
    Float16(.atan2(y: Float(y), x: Float(x)))
  }
  
  #if !os(Windows)
  @_transparent
  public static func logGamma(_ x: Float16) -> Float16 {
    Float16(.logGamma(Float(x)))
  }
  #endif
  
  #if !arch(wasm32)
  // WASM doesn't have _Float16 on the C side, so we can't define the C hooks
  // that these use. TODO: implement these as Swift builtins instead.
  
  @_transparent
  public static func _relaxedAdd(_ a: Float16, _ b: Float16) -> Float16 {
    _numerics_relaxed_addf16(a, b)
  }
  
  @_transparent
  public static func _relaxedMul(_ a: Float16, _ b: Float16) -> Float16 {
    _numerics_relaxed_mulf16(a, b)
  }
  #endif
}

#endif


// ===== Sources/RealModule/Float80+Real.swift =====
//===--- Float80+Real.swift -----------------------------------*- swift -*-===//
//
// This source file is part of the Swift Numerics open source project
//
// Copyright (c) 2019 Apple Inc. and the Swift Numerics project authors
// Licensed under Apache License v2.0 with Runtime Library Exception
//
// See https://swift.org/LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

import _NumericsShims

// Restrict extension to platforms for which Float80 exists.
#if (arch(i386) || arch(x86_64)) && !os(Windows) && !os(Android)
extension Float80: Real {
  @_transparent
  public static func cos(_ x: Float80) -> Float80 {
    libm_cosl(x)
  }
  
  @_transparent
  public static func sin(_ x: Float80) -> Float80 {
    libm_sinl(x)
  }
  
  @_transparent
  public static func tan(_ x: Float80) -> Float80 {
    libm_tanl(x)
  }
  
  @_transparent
  public static func acos(_ x: Float80) -> Float80 {
    libm_acosl(x)
  }
  
  @_transparent
  public static func asin(_ x: Float80) -> Float80 {
    libm_asinl(x)
  }
  
  @_transparent
  public static func atan(_ x: Float80) -> Float80 {
    libm_atanl(x)
  }
  
  @_transparent
  public static func cosh(_ x: Float80) -> Float80 {
    libm_coshl(x)
  }
  
  @_transparent
  public static func sinh(_ x: Float80) -> Float80 {
    libm_sinhl(x)
  }
  
  @_transparent
  public static func tanh(_ x: Float80) -> Float80 {
    libm_tanhl(x)
  }
  
  @_transparent
  public static func acosh(_ x: Float80) -> Float80 {
    libm_acoshl(x)
  }
  
  @_transparent
  public static func asinh(_ x: Float80) -> Float80 {
    libm_asinhl(x)
  }
  
  @_transparent
  public static func atanh(_ x: Float80) -> Float80 {
    libm_atanhl(x)
  }
  
  @_transparent
  public static func exp(_ x: Float80) -> Float80 {
    libm_expl(x)
  }
  
  @_transparent
  public static func expMinusOne(_ x: Float80) -> Float80 {
    libm_expm1l(x)
  }
  
  @_transparent
  public static func log(_ x: Float80) -> Float80 {
    libm_logl(x)
  }
  
  @_transparent
  public static func log(onePlus x: Float80) -> Float80 {
    libm_log1pl(x)
  }
  
  @_transparent
  public static func erf(_ x: Float80) -> Float80 {
    libm_erfl(x)
  }
  
  @_transparent
  public static func erfc(_ x: Float80) -> Float80 {
    libm_erfcl(x)
  }
  
  @_transparent
  public static func exp2(_ x: Float80) -> Float80 {
    libm_exp2l(x)
  }
  
  @_transparent
  public static func hypot(_ x: Float80, _ y: Float80) -> Float80 {
    libm_hypotl(x, y)
  }
  
  @_transparent
  public static func gamma(_ x: Float80) -> Float80 {
    libm_tgammal(x)
  }
  
  @_transparent
  public static func log2(_ x: Float80) -> Float80 {
    libm_log2l(x)
  }
  
  @_transparent
  public static func log10(_ x: Float80) -> Float80 {
    libm_log10l(x)
  }
  
  @_transparent
  public static func pow(_ x: Float80, _ y: Float80) -> Float80 {
    guard x >= 0 else { return .nan }
    if x == 0 && y == 0 { return .nan }
    return libm_powl(x, y)
  }
  
  @_transparent
  public static func pow(_ x: Float80, _ n: Int) -> Float80 {
    // Every Int value is exactly representable as Float80, so we don't need
    // to do anything fancy--unlike Float and Double, we can just call the
    // libm pow function.
    libm_powl(x, Float80(n))
  }
  
  @_transparent
  public static func root(_ x: Float80, _ n: Int) -> Float80 {
    guard x >= 0 || n % 2 != 0 else { return .nan }
    // Workaround the issue mentioned below for the specific case of n = 3
    // where we can fallback on cbrt.
    if n == 3 { return libm_cbrtl(x) }
    // TODO: this implementation is not quite correct, because either n or
    // 1/n may be not be representable as Float80.
    return Float80(signOf: x, magnitudeOf: libm_powl(x.magnitude, 1/Float80(n)))
  }
  
  @_transparent
  public static func atan2(y: Float80, x: Float80) -> Float80 {
    libm_atan2l(y, x)
  }
  
  @_transparent
  public static func logGamma(_ x: Float80) -> Float80 {
    var dontCare: Int32 = 0
    return libm_lgammal(x, &dontCare)
  }
  
  @_transparent
  public static func _relaxedAdd(_ a: Float80, _ b: Float80) -> Float80 {
    _numerics_relaxed_addl(a, b)
  }
  
  @_transparent
  public static func _relaxedMul(_ a: Float80, _ b: Float80) -> Float80 {
    _numerics_relaxed_mull(a, b)
  }
}
#endif


// ===== Sources/RealModule/Real.swift =====
//===--- Real.swift -------------------------------------------*- swift -*-===//
//
// This source file is part of the Swift Numerics open source project
//
// Copyright (c) 2019-2025 Apple Inc. and the Swift Numerics project authors
// Licensed under Apache License v2.0 with Runtime Library Exception
//
// See https://swift.org/LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

/// A type that models the real numbers.
///
/// Types conforming to this protocol provide the arithmetic and utility
/// operations defined by the `FloatingPoint` protocol, and provide all of the
/// math functions defined by the `ElementaryFunctions` and `RealFunctions`
/// protocols. This protocol does not add any additional conformances itself,
/// but is very useful as a protocol against which to write generic code. For
/// example, we can naturally write a generic implementation of a sigmoid
/// function:
/// ```
/// func sigmoid<T: Real>(_ x: T) -> T {
///   return 1/(1 + .exp(-x))
/// }
/// ```
/// See also `ElementaryFunctions`, `RealFunctions` and `AlgebraicField`.
public protocol Real: FloatingPoint, RealFunctions, AlgebraicField { }

//  While `Real` does not provide any additional customization points,
//  it does allow us to default the implementation of a few operations,
//  and also provides `signGamma`.
extension Real {
  // Most math libraries do not provide exp10, so we need a default
  // implementation. This is not a great one (if the underlying math
  // library does not have a sub-ulp accurate pow, this will not get
  // exact powers of ten right), but suffices in the short term.
  @_transparent
  public static func exp10(_ x: Self) -> Self {
    pow(10, x)
  }
  
  /// cos(x) - 1, computed in such a way as to maintain accuracy for small x.
  ///
  /// See also ``ElementaryFunctions/expMinusOne(_:)``.
  @_transparent
  public static func cosMinusOne(_ x: Self) -> Self {
    let sinxOver2 = sin(x/2)
    return -2*sinxOver2*sinxOver2
  }
  
  #if !os(Windows)
  public static func signGamma(_ x: Self) -> FloatingPointSign {
    // Gamma is strictly positive for x >= 0.
    if x >= 0 { return .plus }
    // For negative x, we arbitrarily choose to assign a sign of .plus to the
    // poles.
    let trunc = x.rounded(.towardZero)
    if x == trunc { return .plus }
    // Otherwise, signGamma is .minus if the integral part of x is even.
    return trunc.isEven ? .minus : .plus
  }
  
  //  Determines if this value is even, assuming that it is an integer.
  @inline(__always)
  private var isEven: Bool {
    if Self.radix == 2 {
      // For binary types, we can just check if x/2 is an integer. This works
      // because x/2 is always computed exactly.
      let half = self/2
      return half == half.rounded(.towardZero)
    } else {
      // For decimal types, it's not quite that simple, because x/2 is not
      // necessarily computed exactly. As an example, suppose that we had a
      // decimal type with a one digit significand, and self = 7. Then self/2
      // would round to 4, and we would (wrongly) conclude that it was an
      // integer, and hence that self was even.
      //
      // Instead, for decimal types, we check if 2*trunc(self/2) == self,
      // using an FMA; this is always correct; this approach works for any
      // radix, but the previous method is more efficient for radix == 2.
      let half = self/2
      return self.addingProduct(-2, half.rounded(.towardZero)) == 0
    }
  }
  #endif
  
  @_transparent
  public static func _mulAdd(_ a: Self, _ b: Self, _ c: Self) -> Self {
    a*b + c
  }
  
  @_transparent
  public static func sqrt(_ x: Self) -> Self {
    x.squareRoot()
  }
  
  /// The (approximate) reciprocal (multiplicative inverse) of this number,
  /// if it is representable.
  ///
  /// If `a` if finite and nonzero, and `1/a` overflows or underflows,
  /// then `a.reciprocal` is `nil`. Otherwise, `a.reciprocal` is `1/a`.
  ///
  /// If `b.reciprocal` is non-nil, you may be able to replace division by `b`
  /// with multiplication by this value. It is not advantageous to do this
  /// for an isolated division unless it is a compile-time constant visible
  /// to the compiler, but if you are dividing many values by a single
  /// denominator, this will often be a significant performance win.
  ///
  /// A typical use case looks something like this:
  /// ```
  /// func divide<T: Real>(data: [T], by divisor: T) -> [T] {
  ///   // If divisor is well-scaled, multiply by reciprocal.
  ///   if let recip = divisor.reciprocal {
  ///     return data.map { $0 * recip }
  ///   }
  ///   // Fallback on using division.
  ///   return data.map { $0 / divisor }
  /// }
  /// ```
  ///
  /// Error Bounds:
  ///
  /// Multiplying by the reciprocal instead of dividing will slightly
  /// perturb results. For example `5.0 / 3` is 1.6666666666666667, but
  /// `5.0 * 3.reciprocal!` is 1.6666666666666665.
  ///
  /// The error of a normal division is bounded by half an ulp of the
  /// result; we can derive a quick error bound for multiplication by
  /// the real reciprocal (when it exists) as follows (I will use circle
  /// operators to denote real-number arithmetic, and normal operators
  /// for floating-point arithmetic):
  /// ```
  /// a * b.reciprocal! = a * (1/b)
  ///                   = a * (1 ⊘ b)(1 + δ₁)
  ///                   = (a ⊘ b)(1 + δ₁)(1 + δ₂)
  ///                   = (a ⊘ b)(1 + δ₁ + δ₂ + δ₁δ₂)
  /// ```
  /// where `0 < δᵢ <= ulpOfOne/2`. This gives a roughly 1-ulp error,
  /// about twice the error bound we get using division. For most
  /// purposes this is an acceptable error, but if you need to match
  /// results obtained using division, you should not use this.
  @inlinable
  public var reciprocal: Self? {
    let recip = 1/self
    if recip.isNormal || isZero || !isFinite {
      return recip
    }
    return nil
  }
}


// ===== Sources/RealModule/RealFunctions.swift =====
//===--- RealFunctions.swift ----------------------------------*- swift -*-===//
//
// This source file is part of the Swift Numerics open source project
//
// Copyright (c) 2019-2025 Apple Inc. and the Swift Numerics project authors
// Licensed under Apache License v2.0 with Runtime Library Exception
//
// See https://swift.org/LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

public protocol RealFunctions: ElementaryFunctions {
  /// The signed angle formed in the plane between the vector `(x,y)` and the
  /// positive real axis, measured in radians.
  ///
  /// The result is in the interval `[-π, π]`.
  ///
  /// The argument order to `atan2` may be surprising to new programmers.
  /// The convention of `y` being the first argument goes back at least to
  /// Fortran IV in 1961 and is generally followed in computing with a few
  /// notable exceptions (e.g. Mathematica and Excel). This convention was
  /// originally chosen because of the mathematical definition of the
  /// function:
  ///
  /// ```
  /// atan2(y,x) = atan(y/x) if x > 0
  /// ```
  ///
  /// See also ``ElementaryFunctions/atan(_:)``, as well as the `phase` and
  /// `polar` properties defined on the `Complex` type.
  static func atan2(y: Self, x: Self) -> Self
  
  /// The [error function](https://en.wikipedia.org/wiki/Error_function)
  /// evaluated at `x`.
  static func erf(_ x: Self) -> Self
  
  /// The complimentary [error function](https://en.wikipedia.org/wiki/Error_function)
  /// evaluated at `x`.
  static func erfc(_ x: Self) -> Self
  
  /// 2 raised to the power x.
  ///
  /// See also ``log2(_:)``, ``ElementaryFunctions/exp(_:)``,
  /// ``ElementaryFunctions/expMinusOne(_:)``
  /// and ``ElementaryFunctions/pow(_:_:)-2qmul``.
  static func exp2(_ x: Self) -> Self
  
  /// 10 raised to the power x.
  ///
  /// See also ``log10(_:)``, ``ElementaryFunctions/exp(_:)``,
  /// ``ElementaryFunctions/expMinusOne(_:)``
  /// and ``ElementaryFunctions/pow(_:_:)-2qmul``.
  static func exp10(_ x: Self) -> Self
  
  /// The length of the vector `(x,y)`, computed in a manner that avoids
  /// spurious overflow or underflow.
  ///
  /// See also the `length` and `polar` properties defined on the `Complex`
  /// type.
  static func hypot(_ x: Self, _ y: Self) -> Self
  
  /// The [gamma function](https://en.wikipedia.org/wiki/Gamma_function) Γ(x).
  static func gamma(_ x: Self) -> Self
  
  /// The base-2 logarithm of `x`.
  ///
  /// See also ``exp2(_:)``, ``ElementaryFunctions/log(_:)``,
  /// and ``ElementaryFunctions/log(onePlus:)``.
  static func log2(_ x: Self) -> Self
  
  /// The base-10 logarithm of `x`.
  ///
  /// See also ``exp10(_:)``, ``ElementaryFunctions/log(_:)``,
  /// and ``ElementaryFunctions/log(onePlus:)``.
  static func log10(_ x: Self) -> Self
  
#if !os(Windows)
  /// The logarithm of the absolute value of the
  /// [gamma function](https://en.wikipedia.org/wiki/Gamma_function),
  /// log(|Γ(x)|).
  ///
  /// Not available on Windows targets.
  static func logGamma(_ x: Self) -> Self
  
  /// The sign of the
  /// [gamma function](https://en.wikipedia.org/wiki/Gamma_function), Γ(x).
  ///
  /// For `x >= 0`, `signGamma(x)` is `.plus`. For negative `x`, `signGamma(x)`
  /// is `.plus` when `x` is an integer, and otherwise it is `.minus` whenever
  /// `trunc(x)` is even, and `.plus` when `trunc(x)` is odd.
  ///
  /// This function is used together with ``logGamma(_:)``, which computes the
  /// logarithm of the absolute value of Γ(x), to recover the sign information.
  ///
  /// Not available on Windows targets.
  static func signGamma(_ x: Self) -> FloatingPointSign
#endif
}


// ===== Sources/RealModule/RelaxedArithmetic.swift =====
//===--- RelaxedArithmetic.swift ------------------------------*- swift -*-===//
//
// This source file is part of the Swift Numerics open source project
//
// Copyright (c) 2021-2025 Apple Inc. and the Swift Numerics project authors
// Licensed under Apache License v2.0 with Runtime Library Exception
//
// See https://swift.org/LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

import _NumericsShims

public enum Relaxed { }

extension Relaxed {
  /// a+b, but grants the optimizer permission to reassociate expressions
  /// and form FMAs.
  ///
  /// Floating-point addition is not an associative operation, so the Swift
  /// compiler does not have any flexibility in how it evaluates an expression
  /// like:
  /// ```
  /// func sum(array: [Float]) -> Float {
  ///   array.reduce(0, +)
  /// }
  /// ```
  /// Using `Relaxed.sum` instead of `+` permits the compiler to reorder the
  /// terms in the summation, which unlocks loop unrolling and vectorization.
  /// In a benchmark, simply using `Relaxed.sum` provides about an 8x speedup
  /// for release builds, without any unsafe flags or other optimizations.
  /// Further improvement should be possible by improving LLVM optimizations
  /// or adding attributes to license more aggressive unrolling and taking
  /// advantage of vector ISA extensions for swift.
  ///
  /// If you want to compute `a-b` with relaxed semantics, use
  /// `Relaxed.sum(a, -b)`.
  ///
  /// If a type or toolchain does not support reassociation for optimization
  /// purposes, this operation decays to a normal addition; it is a license
  /// for the compiler to optimize, not a guarantee that any change occurs.
  @_transparent
  public static func sum<T: AlgebraicField>(_ a: T, _ b: T) -> T {
    T._relaxedAdd(a, b)
  }
  
  /// a*b, but grants the optimizer permission to reassociate expressions
  /// and form FMAs.
  ///
  /// Floating-point addition and multiplication are not associative operations,
  /// so the Swift compiler does not have any flexibility in how it evaluates
  /// an expression like:
  /// ```
  /// func sumOfSquares(array: [Float]) -> Float {
  ///   array.reduce(0) { $0 + $1*$1 }
  /// }
  /// ```
  /// Using `Relaxed.sum` and `Relaxed.product` instead of `+` and `*` permits
  /// the compiler to reorder the terms in the summation, which unlocks loop
  /// unrolling and vectorization, and form fused multiply-adds, which allows
  /// us to achieve twice the throughput on some hardware.
  ///
  /// If a type or toolchain does not support reassociation for optimization
  /// purposes, this operation decays to a normal multiplication; it is a
  /// license for the compiler to optimize, not a guarantee that any change
  /// occurs.
  @_transparent
  public static func product<T: AlgebraicField>(_ a: T, _ b: T) -> T {
    T._relaxedMul(a, b)
  }
}

extension Relaxed {
  /// a*b + c, computed _either_ with an FMA or with separate multiply and add,
  /// whichever is fastest according to the optimizer's heuristics.
  @_transparent
  public static func multiplyAdd<T: AlgebraicField>(
    _ a: T, _ b: T, _ c: T
  ) -> T {
    T._relaxedAdd(c, T._relaxedMul(a, b))
  }
}

