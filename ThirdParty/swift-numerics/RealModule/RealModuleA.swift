// Vendored from apple/swift-numerics 1.1.1; see ../LICENSE.txt

// ===== Sources/RealModule/AlgebraicField.swift =====
//===--- AlgebraicField.swift ---------------------------------*- swift -*-===//
//
// This source file is part of the Swift Numerics open source project
//
// Copyright (c) 2019-2025 Apple Inc. and the Swift Numerics project authors
// Licensed under Apache License v2.0 with Runtime Library Exception
//
// See https://swift.org/LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

/// A type modeling an algebraic [field]. Refines the `SignedNumeric` protocol,
/// adding division.
///
/// A field is a set on which addition, subtraction, multiplication, and
/// division are defined, and behave basically like those operations on
/// the real numbers. More precisely, a field is a commutative group under
/// its addition, the non-zero elements of the field form a commutative
/// group under its multiplication, and the distributive law holds.
///
/// Some common examples of fields include:
///
/// - the rational numbers
/// - the real numbers
/// - the complex numbers
/// - the integers modulo a prime
///
/// The most familiar example of a thing that is *not* a field is the integers.
/// This may be surprising, since integers seem to have addition, subtraction,
/// multiplication and division. Why don't they form a field?
///
/// Because integer multiplication does not form a group; it's commutative and
/// associative, but integers do not have multiplicative inverses.
/// I.e. if a is any integer other than 1 or -1, there is no integer b such
/// that `a*b = 1`. The existence of inverses is requried to form a field.
///
/// If a type `T` conforms to the ``Real`` protocol, then `T` and `Complex<T>`
/// both conform to `AlgebraicField`.
///
/// See also Swift's `SignedNumeric`, `Numeric` and `AdditiveArithmetic`
/// protocols.
///
/// [field]: https://en.wikipedia.org/wiki/Field_(mathematics)
public protocol AlgebraicField: SignedNumeric where Magnitude: AlgebraicField {
  
  /// Replaces a with the (approximate) quotient `a/b`.
  static func /=(a: inout Self, b: Self)
  
  /// The (approximate) quotient `a/b`.
  static func /(a: Self, b: Self) -> Self
  
  /// The (approximate) reciprocal (multiplicative inverse) of this number,
  /// if it is representable.
  ///
  /// If reciprocal is non-nil, you can replace division by self with
  /// multiplication by reciprocal and either get exact the same result
  /// (for finite fields) or approximately the same result up to a typical
  /// rounding error (for floating-point formats).
  ///
  /// If self is zero and the type has no representation for infinity (as
  /// in a typical finite field implementation), or if a reciprocal would
  /// overflow or underflow such that it cannot be accurately represented,
  /// the result is nil.
  ///
  /// Note that `.zero.reciprocal`, somewhat surprisingly, is *not* nil
  /// for `Real` or `Complex` types, because these types have an
  /// `.infinity` value that acts as the reciprocal of `.zero`.
  ///
  /// If `b.reciprocal` is non-nil, you may be able to replace division by `b`
  /// with multiplication by this value. It is not advantageous to do this
  /// for an isolated division unless it is a compile-time constant visible
  /// to the compiler, but if you are dividing many values by a single
  /// denominator, this will often be a significant performance win.
  ///
  /// Note that this will slightly perturb results for fields with approximate
  /// arithmetic, such as real or complex types--using a normal division
  /// is generally more accurate--but no catastrophic loss of accuracy will
  /// result. For fields with exact arithmetic, the results are necessarily
  /// identical.
  ///
  /// A typical use case looks something like this:
  /// ```
  /// func divide<T: AlgebraicField>(data: [T], by divisor: T) -> [T] {
  ///   // If divisor is well-scaled, multiply by reciprocal.
  ///   if let recip = divisor.reciprocal {
  ///     return data.map { $0 * recip }
  ///   }
  ///   // Fallback on using division.
  ///   return data.map { $0 / divisor }
  /// }
  /// ```
  var reciprocal: Self? { get }
  
  /// `a + b`, with the optimizer licensed to reassociate and form FMAs.
  static func _relaxedAdd(_ a: Self, _ b: Self) -> Self
  
  /// `a * b`, with the optimizer licensed to reassociate and form FMAs.
  static func _relaxedMul(_ a: Self, _ b: Self) -> Self
}

extension AlgebraicField {
  @_transparent
  public static func /(a: Self, b: Self) -> Self {
    var result = a
    result /= b
    return result
  }
  
  // Implementations should be *conservative* with the reciprocal property;
  // it is OK to return `nil` even in cases where a reciprocal could be
  // represented. For this reason, a default implementation that simply
  // always returns `nil` is correct, but conforming types should provide
  // a better implementation if possible.
  @_transparent
  public var reciprocal: Self? {
    return nil
  }
  
  // It's always OK to simply fall back on normal arithmetic, and for any
  // field with exact arithmetic, this is the correct definition.
  @_transparent
  public static func _relaxedAdd(_ a: Self, _ b: Self) -> Self {
    a + b
  }
  
  // It's always OK to simply fall back on normal arithmetic, and for any
  // field with exact arithmetic, this is the correct definition.
  @_transparent
  public static func _relaxedMul(_ a: Self, _ b: Self) -> Self {
    a * b
  }
}


// ===== Sources/RealModule/ApproximateEquality.swift =====
//===--- ApproximateEquality.swift ----------------------------*- swift -*-===//
//
// This source file is part of the Swift Numerics open source project
//
// Copyright (c) 2020-2025 Apple Inc. and the Swift Numerics project authors
// Licensed under Apache License v2.0 with Runtime Library Exception
//
// See https://swift.org/LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

extension Numeric where Magnitude: FloatingPoint {
  /// Test if `self` and `other` are approximately equal.
  ///
  /// `true` if `self` and `other` are equal, or if they are finite and
  /// ```
  /// norm(self - other) <= relativeTolerance * scale
  /// ```
  /// where `scale` is
  /// ```
  /// max(norm(self), norm(other), .leastNormalMagnitude)
  /// ```
  ///
  /// The default value of `relativeTolerance` is `.ulpOfOne.squareRoot()`,
  /// which corresponds to expecting "about half the digits" in the computed
  /// results to be good. This is the usual guidance in numerical analysis,
  /// if you don't know anything about the computation being performed, but
  /// is not suitable for all use cases.
  ///
  /// Mathematical Properties:
  ///
  /// - `isApproximatelyEqual(to:relativeTolerance:norm:)` is _reflexive_ for
  ///   non-exceptional values (such as NaN).
  ///
  /// - `isApproximatelyEqual(to:relativeTolerance:norm:)` is _symmetric_.
  ///
  /// - `isApproximatelyEqual(to:relativeTolerance:norm:)` is __not__
  ///   _transitive_. Because of this, approximately equality is __not an
  ///   equivalence relation__, even when restricted to non-exceptional values.
  ///
  ///   This means that you must not use approximate equality to implement
  ///   a conformance to Equatable, as it will violate the invariants of
  ///   code written against that protocol.
  ///
  /// - For any point `a`, the set of values that compare approximately equal
  ///   to `a` is _convex_. (Under the assumption that the `.magnitude`
  ///   property implements a valid norm.)
  ///
  /// - `isApproximatelyEqual(to:relativeTolerance:norm:)` is _scale invariant_,
  ///   so long as no underflow or overflow has occurred, and no exceptional
  ///   value is produced by the scaling.
  ///
  /// See also `isApproximatelyEqual(to:absoluteTolerance:[relativeTolerance:norm:])`.
  ///
  /// - Parameters:
  ///
  ///   - other: The value to which `self` is compared.
  ///
  ///   - relativeTolerance: The tolerance to use for the comparison.
  ///     Defaults to `.ulpOfOne.squareRoot()`.
  ///
  ///     This value should be non-negative and less than or equal to 1.
  ///     This constraint on is only checked in debug builds, because a
  ///     mathematically well-defined result exists for any tolerance,
  ///     even one out of range.
  ///
  ///   - norm: The [norm] to use for the comparison.
  ///     Defaults to `\.magnitude`.
  ///
  /// [norm]: https://en.wikipedia.org/wiki/Norm_(mathematics)
  @inlinable @inline(__always)
  public func isApproximatelyEqual(
    to other: Self,
    relativeTolerance: Magnitude = Magnitude.ulpOfOne.squareRoot(),
    norm: (Self) -> Magnitude = \.magnitude
  ) -> Bool {
    return isApproximatelyEqual(
      to: other,
      absoluteTolerance: relativeTolerance * Magnitude.leastNormalMagnitude,
      relativeTolerance: relativeTolerance,
      norm: norm
    )
  }
  
  /// Test if `self` and `other` are approximately equal with specified tolerances.
  ///
  /// `true` if `self` and `other` are equal, or if they are finite and either
  /// ```
  /// (self - other).magnitude <= absoluteTolerance
  /// ```
  /// or
  /// ```
  /// (self - other).magnitude <= relativeTolerance * scale
  /// ```
  /// where `scale` is `max(self.magnitude, other.magnitude)`.
  ///
  /// Mathematical Properties:
  ///
  /// - `isApproximatelyEqual(to:absoluteTolerance:relativeTolerance:)`
  ///   is _reflexive_ for non-exceptional values (such as NaN).
  ///
  /// - `isApproximatelyEqual(to:absoluteTolerance:relativeTolerance:)`
  ///   is _symmetric_.
  ///
  /// - `isApproximatelyEqual(to:relativeTolerance:norm:)` is __not__
  ///   _transitive_. Because of this, approximately equality is __not an
  ///   equivalence relation__, even when restricted to non-exceptional values.
  ///
  ///   This means that you must not use approximate equality to implement
  ///   a conformance to Equatable, as it will violate the invariants of
  ///   code written against that protocol.
  ///
  /// - For any point `a`, the set of values that compare approximately equal
  ///   to `a` is _convex_. (Under the assumption that `norm` implements a
  ///   valid norm, which cannot be checked by this function.)
  ///
  /// See also `isApproximatelyEqual(to:[relativeTolerance:])`.
  ///
  /// - Parameters:
  ///
  ///   - other: The value to which `self` is compared.
  ///
  ///   - absoluteTolerance: The absolute tolerance to use in the comparison.
  ///
  ///     This value should be non-negative and finite.
  ///     This constraint on is only checked in debug builds, because a
  ///     mathematically well-defined result exists for any tolerance,
  ///     even one out of range.
  ///
  ///   - relativeTolerance: The relative tolerance to use in the comparison.
  ///     Defaults to zero.
  ///
  ///     This value should be non-negative and less than or equal to 1.
  ///     This constraint on is only checked in debug builds, because a
  ///     mathematically well-defined result exists for any tolerance,
  ///     even one out of range.
  @inlinable @inline(__always)
  public func isApproximatelyEqual(
    to other: Self,
    absoluteTolerance: Magnitude,
    relativeTolerance: Magnitude = 0
  ) -> Bool {
    self.isApproximatelyEqual(
      to: other,
      absoluteTolerance: absoluteTolerance,
      relativeTolerance: relativeTolerance,
      norm: \.magnitude
    )
  }
}

extension AdditiveArithmetic {
  /// Test if `self` and `other` are approximately equal with specified
  /// tolerances and norm.
  ///
  /// `true` if `self` and `other` are equal, or if they are finite and either
  /// ```
  /// norm(self - other) <= absoluteTolerance
  /// ```
  /// or
  /// ```
  /// norm(self - other) <= relativeTolerance * scale
  /// ```
  /// where `scale` is `max(norm(self), norm(other))`.
  ///
  /// Mathematical Properties:
  ///
  /// - `isApproximatelyEqual(to:absoluteTolerance:relativeTolerance:norm:)`
  ///   is _reflexive_ for non-exceptional values (such as NaN).
  ///
  /// - `isApproximatelyEqual(to:absoluteTolerance:relativeTolerance:norm:)`
  ///   is _symmetric_.
  ///
  /// - `isApproximatelyEqual(to:absoluteTolerance:relativeTolerance:norm:)`
  ///   is __not__ _transitive_. Because of this, approximately equality is
  ///   __not an equivalence relation__, even when restricted to
  ///   non-exceptional values.
  ///
  ///   This means that you must not use approximate equality to implement
  ///   a conformance to Equatable, as it will violate the invariants of
  ///   code written against that protocol.
  ///
  /// - For any point `a`, the set of values that compare approximately equal
  ///   to `a` is _convex_ (under the assumption that `norm` implements a
  ///   valid norm, which cannot be checked by this function or a protocol).
  ///
  /// See also `isApproximatelyEqual(to:[relativeTolerance:norm:])` and
  /// `isApproximatelyEqual(to:absoluteTolerance:[relativeTolerance:])`.
  ///
  /// - Parameters:
  ///
  ///   - other: The value to which `self` is compared.
  ///
  ///   - absoluteTolerance: The absolute tolerance to use in the comparison.
  ///
  ///     This value should be non-negative and finite.
  ///     This constraint on is only checked in debug builds, because a
  ///     mathematically well-defined result exists for any tolerance, even
  ///     one out of range.
  ///
  ///   - relativeTolerance: The relative tolerance to use in the comparison.
  ///     Defaults to zero.
  ///
  ///     This value should be non-negative and less than or equal to 1.
  ///     This constraint on is only checked in debug builds, because a
  ///     mathematically well-defined result exists for any tolerance,
  ///     even one out of range.
  ///
  ///   - norm: The norm to use for the comparison.
  ///     Defaults to `\.magnitude`.
  ///
  ///     For example, if we wanted to test if a complex value was inside a
  ///     circle of radius 0.001 centered at (1 + 0i), we could use:
  ///     ```
  ///     z.isApproximatelyEqual(
  ///       to: 1,
  ///       absoluteTolerance: 0.001,
  ///       norm: \.length
  ///     )
  ///     ```
  ///     (if we used the default norm, `\.magnitude`, we would be testing if
  ///     `z` were inside a square region instead.)
  @inlinable
  public func isApproximatelyEqual<Magnitude>(
    to other: Self,
    absoluteTolerance: Magnitude,
    relativeTolerance: Magnitude = 0,
    norm: (Self) -> Magnitude
  ) -> Bool
  where Magnitude: FloatingPoint {
    assert(
      absoluteTolerance >= 0 && absoluteTolerance.isFinite,
      "absoluteTolerance should be non-negative and finite, " +
      "but is \(absoluteTolerance)."
    )
    assert(
      relativeTolerance >= 0 && relativeTolerance <= 1,
      "relativeTolerance should be non-negative and <= 1, " +
      "but is \(relativeTolerance)."
    )
    if self == other { return true }
    let delta = norm(self - other)
    let scale = max(norm(self), norm(other))
    let bound = max(absoluteTolerance, scale*relativeTolerance)
    return delta.isFinite && delta <= bound
  }
}


// ===== Sources/RealModule/AugmentedArithmetic.swift =====
//===--- AugmentedArithmetic.swift ----------------------------*- swift -*-===//
//
// This source file is part of the Swift Numerics open source project
//
// Copyright (c) 2020-2025 Apple Inc. and the Swift Numerics project authors
// Licensed under Apache License v2.0 with Runtime Library Exception
//
// See https://swift.org/LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

public enum Augmented { }

extension Augmented {
  /// The product `a * b` represented as an implicit sum `head + tail`.
  ///
  /// `head` is the correctly rounded value of `a*b`. If no overflow or
  /// underflow occurs, `tail` represents the rounding error incurred in
  /// computing `head`, such that the exact product is the sum of `head`
  /// and `tail` computed without rounding.
  ///
  /// This operation is sometimes called "twoProd" or "twoProduct".
  ///
  /// Edge Cases:
  ///
  /// - `head` is always the IEEE 754 product `a * b`.
  /// - If `head` is not finite, `tail` is unspecified and should not be
  ///   interpreted as having any meaning (it may be `NaN` or `infinity`).
  /// - When `head` is close to the underflow boundary, the rounding error
  ///   may not be representable due to underflow, and `tail` will be rounded.
  ///   If `head` is very small, `tail` may even be zero, even though the
  ///   product is not exact.
  /// - If `head` is zero, `tail` is also a zero with unspecified sign.
  ///
  /// Postconditions:
  ///
  /// - If `head` is normal, then `abs(tail) < head.ulp`.
  ///   Assuming IEEE 754 default rounding, `abs(tail) <= head.ulp/2`.
  /// - If both `head` and `tail` are normal, then `a * b` is exactly
  ///   equal to `head + tail` when computed as real numbers.
  @_transparent
  public static func product<T:FloatingPoint>(
    _ a: T, _ b: T
  ) -> (head: T, tail: T) {
    let head = a*b
    // TODO: consider providing an FMA-less implementation for use when
    // targeting platforms without hardware FMA support. This works everywhere,
    // falling back on the C math.h fma funcions, but may be slow on older x86.
    let tail = (-head).addingProduct(a, b)
    return (head, tail)
  }
  
  /// The sum `a + b` represented as an implicit sum `head + tail`.
  ///
  /// - Parameters:
  ///   - a: The summand with larger magnitude.
  ///   - b: The summand with smaller magnitude.
  ///
  /// `head` is the correctly rounded value of `a + b`. `tail` is the
  /// error from that computation rounded to the closest representable
  /// value.
  ///
  /// > Note:
  /// > `tail` is guaranteed to be the best approximation to the error of
  ///   the sum only if `large.magnitude` >= `small.magnitude`. If this is
  ///   not the case, then `head` is the correctly rounded sum, but `tail`
  ///   is not guaranteed to be the exact error. If you do not know a priori
  ///   how the magnitudes of `a` and `b` compare, you likely want to use
  ///   ``sum(_:_:)`` instead.
  ///
  /// Unlike ``product(_:_:)``, the rounding error of `sum` never underflows.
  ///
  /// This operation is sometimes called ["fastTwoSum"].
  ///
  /// > Note:
  /// > Classical fastTwoSum does not work when `radix` is 10. This function
  ///   will fall back on another algorithm for decimal floating-point types
  ///   to ensure correct results.
  ///
  /// Edge Cases:
  ///
  /// - `head` is always the IEEE 754 sum `a + b`.
  /// - If `head` is not finite, `tail` is unspecified and should not be
  ///   interpreted as having any meaning (it may be `NaN` or `infinity`).
  ///
  /// Postconditions:
  ///
  /// - If `head` is normal, then `abs(tail) < head.ulp`.
  ///   Assuming IEEE 754 default rounding, `abs(tail) <= head.ulp/2`.
  ///
  /// ["fastTwoSum"]:  https://en.wikipedia.org/wiki/2Sum
  @_transparent
  public static func sum<T: FloatingPoint>(
    large a: T, small b: T
  ) -> (head: T, tail: T) {
    // Fall back on 2Sum if radix != 2. Future implementations might use an
    // cheaper algorithm specialized for decimal FP, but must deliver a
    // correct result if the preconditions are satisfied.
    guard T.radix == 2 else { return sum(a, b) }
    // Fast2Sum:
    let head = a + b
    let tail = a - head + b
    return (head, tail)
  }
  
  /// The sum `a + b` represented as an implicit sum `head + tail`.
  ///
  /// `head` is the correctly rounded value of `a + b`. `tail` is the
  /// error from that computation rounded to the closest representable
  /// value.
  ///
  /// Unlike ``sum(large:small:)``, the magnitude of the summands does not
  /// matter. If you know statically that `a.magnitude >= b.magnitude`, you
  /// should use ``sum(large:small:)``. If you do not have such a static
  /// bound, you should use this function instead.
  ///
  /// Unlike ``product(_:_:)``, the rounding error of `sum` never underflows.
  ///
  /// This operation is sometimes called ["twoSum"].
  ///
  /// - Parameters:
  ///   - a: One of the summands
  ///   - b: The other summand
  ///
  /// Edge Cases:
  ///
  /// - `head` is always the IEEE 754 sum `a + b`.
  /// - If `head` is not finite, `tail` is unspecified and should not be
  ///   interpreted as having any meaning (it may be `NaN` or `infinity`).
  ///
  /// Postconditions:
  ///
  /// - If `head` is normal, then `abs(tail) < head.ulp`.
  ///   Assuming IEEE 754 default rounding, `abs(tail) <= head.ulp/2`.
  ///
  /// ["twoSum"]:  https://en.wikipedia.org/wiki/2Sum
  @_transparent
  public static func sum<T: FloatingPoint>(
    _ a: T, _ b: T
  ) -> (head: T, tail: T) {
    let head = a + b
    let x = head - b
    let y = head - x
    let tail = (a - x) + (b - y)
    return (head, tail)
  }
}


// ===== Sources/RealModule/Double+Real.swift =====
//===--- Double+Real.swift ------------------------------------*- swift -*-===//
//
// This source file is part of the Swift Numerics open source project
//
// Copyright (c) 2019-2025 Apple Inc. and the Swift Numerics project authors
// Licensed under Apache License v2.0 with Runtime Library Exception
//
// See https://swift.org/LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

import _NumericsShims

extension Double: Real {
  @_transparent
  public static func cos(_ x: Double) -> Double {
    libm_cos(x)
  }
  
  @_transparent
  public static func sin(_ x: Double) -> Double {
    libm_sin(x)
  }
  
  @_transparent
  public static func tan(_ x: Double) -> Double {
    libm_tan(x)
  }
  
  @_transparent
  public static func acos(_ x: Double) -> Double {
    libm_acos(x)
  }
  
  @_transparent
  public static func asin(_ x: Double) -> Double {
    libm_asin(x)
  }
  
  @_transparent
  public static func atan(_ x: Double) -> Double {
    libm_atan(x)
  }
  
  @_transparent
  public static func cosh(_ x: Double) -> Double {
    libm_cosh(x)
  }
  
  @_transparent
  public static func sinh(_ x: Double) -> Double {
    libm_sinh(x)
  }
  
  @_transparent
  public static func tanh(_ x: Double) -> Double {
    libm_tanh(x)
  }
  
  @_transparent
  public static func acosh(_ x: Double) -> Double {
    libm_acosh(x)
  }
  
  @_transparent
  public static func asinh(_ x: Double) -> Double {
    libm_asinh(x)
  }
  
  @_transparent
  public static func atanh(_ x: Double) -> Double {
    libm_atanh(x)
  }
  
  @_transparent
  public static func exp(_ x: Double) -> Double {
    libm_exp(x)
  }
  
  @_transparent
  public static func expMinusOne(_ x: Double) -> Double {
    libm_expm1(x)
  }
  
  @_transparent
  public static func log(_ x: Double) -> Double {
    libm_log(x)
  }
  
  @_transparent
  public static func log(onePlus x: Double) -> Double {
    libm_log1p(x)
  }
  
  @_transparent
  public static func erf(_ x: Double) -> Double {
    libm_erf(x)
  }
  
  @_transparent
  public static func erfc(_ x: Double) -> Double {
    libm_erfc(x)
  }
  
  @_transparent
  public static func exp2(_ x: Double) -> Double {
    libm_exp2(x)
  }
  
#if os(macOS) || os(iOS) || os(tvOS) || os(watchOS)
  @_transparent
  public static func exp10(_ x: Double) -> Double {
    libm_exp10(x)
  }
#endif
  
#if os(macOS) && arch(x86_64)
  // Workaround for macOS bug (<rdar://problem/56844150>) where hypot can
  // overflow for values very close to the overflow boundary of the naive
  // algorithm. Since this is only for macOS, we can just unconditionally
  // use Float80, which makes the implementation trivial.
  public static func hypot(_ x: Double, _ y: Double) -> Double {
    if x.isInfinite || y.isInfinite { return .infinity }
    let x80 = Float80(x)
    let y80 = Float80(y)
    return Double(Float80.sqrt(x80*x80 + y80*y80))
  }
#else
  @_transparent
  public static func hypot(_ x: Double, _ y: Double) -> Double {
    libm_hypot(x, y)
  }
#endif
  
  @_transparent
  public static func gamma(_ x: Double) -> Double {
    libm_tgamma(x)
  }
  
  @_transparent
  public static func log2(_ x: Double) -> Double {
    libm_log2(x)
  }
  
  @_transparent
  public static func log10(_ x: Double) -> Double {
    libm_log10(x)
  }
  
  @_transparent
  public static func pow(_ x: Double, _ y: Double) -> Double {
    guard x >= 0 else { return .nan }
    if x == 0 && y == 0 { return .nan }
    return libm_pow(x, y)
  }
  
  @_transparent
  public static func pow(_ x: Double, _ n: Int) -> Double {
    // If n is exactly representable as Double, we can just call pow:
    // Note that all calls on a 32b platform go down this path.
    if let y = Double(exactly: n) { return libm_pow(x, y) }
    // n is not representable in Double, so we will split it into two parts,
    // low and high, such that (high + low) = n, and use the identity:
    //
    //   x**(high + low) = x**high * x**low.
    //
    // We put the high-order 32 bits into high, and the remaining 32 bits
    // in low.
    //
    // The exact split isn't important; all we need is that both pieces get
    // less than 53 bits (so that they are exact) and that they both have
    // the same sign as n.
    //
    // This second point is a little bit subtle--why is
    // it necessary? Consider what would happen if we took x = 2 and
    // n = Int.min + Int(UInt32.max), and simply naively split n without
    // taking care with the sign. We would end up computing:
    //
    //   2**n = 2**Int.min * 2**UInt32.max
    //
    // The first exponent is negative, the second positive, so the first term
    // underflows to zero, and the second overflows to infinity, so the final
    // result is NaN, when it should be zero. In order to avoid this
    // situation, we make sure that high contains n rounded *towards zero*,
    // rather than using simple two's-complement truncation (which rounds
    // down).
    let mask = Int(truncatingIfNeeded: UInt32.max)
    let round = n < 0 ? mask : 0
    // The addition and subtraction below cannot actually overflow (proof:
    // round is positive if n is negative, and zero otherwise, so n + round
    // is guaranteed to be representable, and n and high have the same sign,
    // so n - high is also representable), but it's hard to tell the compiler
    // that, so I'm using wrapping operations instead.
    let high = (n &+ round) & ~mask
    let low = n &- high
    return libm_pow(x, Double(low)) * libm_pow(x, Double(high))
  }
  
  @_transparent
  public static func root(_ x: Double, _ n: Int) -> Double {
    guard x >= 0 || n % 2 != 0 else { return .nan }
    // Workaround the issue mentioned below for the specific case of n = 3
    // where we can fallback on cbrt.
    if n == 3 { return libm_cbrt(x) }
    // TODO: this implementation is not quite correct, because either n or
    // 1/n may be not be representable as Double.
    return Double(signOf: x, magnitudeOf: libm_pow(x.magnitude, 1/Double(n)))
  }
  
  @_transparent
  public static func atan2(y: Double, x: Double) -> Double {
    libm_atan2(y, x)
  }
  
#if !os(Windows)
  @_transparent
  public static func logGamma(_ x: Double) -> Double {
    var dontCare: Int32 = 0
    return libm_lgamma(x, &dontCare)
  }
#endif
  
  @_transparent
  public static func _relaxedAdd(_ a: Double, _ b: Double) -> Double {
    _numerics_relaxed_add(a, b)
  }
  
  @_transparent
  public static func _relaxedMul(_ a: Double, _ b: Double) -> Double {
    _numerics_relaxed_mul(a, b)
  }
}


// ===== Sources/RealModule/ElementaryFunctions.swift =====
//===--- ElementaryFunctions.swift ----------------------------*- swift -*-===//
//
// This source file is part of the Swift Numerics open source project
//
// Copyright (c) 2019-2025 Apple Inc. and the Swift Numerics project authors
// Licensed under Apache License v2.0 with Runtime Library Exception
//
// See https://swift.org/LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

public protocol ElementaryFunctions: AdditiveArithmetic {
  /// The [exponential function][wiki] e^x whose base `e` is the base of the
  /// natural logarithm.
  ///
  /// For types that conform to ``RealFunctions`` see
  /// ``RealFunctions/exp2(_:)`` and ``RealFunctions/exp10(_:)``.
  ///
  /// [wiki]: https://en.wikipedia.org/wiki/Exponential_function
  static func exp(_ x: Self) -> Self
  
  /// exp(x) - 1, computed in such a way as to maintain accuracy for small x.
  ///
  /// When `x` is close to zero, the expression `.exp(x) - 1` suffers from
  /// catastrophic cancellation and the result will not have full accuracy.
  /// The `.expMinusOne(x)` function gives you a means to address this problem.
  ///
  /// As an example, consider the expression `(x + 1) * .exp(x) - 1`.  When `x`
  /// is smaller than `.ulpOfOne`, this expression evaluates to `0.0`, when it
  /// should actually round to `2*x`. We can get a full-accuracy result by
  /// using the following instead:
  /// ```
  /// let t = .expMinusOne(x)
  /// return x*(t+1) + t       // x*exp(x) + (exp(x)-1) = (x+1)*exp(x) - 1
  /// ```
  /// This re-written expression delivers an accurate result for all values
  /// of `x`, not just for small values.
  ///
  /// For types that conform to ``RealFunctions`` see
  /// ``RealFunctions/exp2(_:)`` and ``RealFunctions/exp10(_:)``.
  static func expMinusOne(_ x: Self) -> Self
  
  /// The [hyperbolic cosine][wiki] of `x`.
  /// ```
  ///            e^x + e^-x
  /// cosh(x) = ------------
  ///                2
  /// ```
  ///
  /// See also ``acosh(_:)``.
  ///
  /// [wiki]: https://en.wikipedia.org/wiki/Hyperbolic_function
  static func cosh(_ x: Self) -> Self
  
  /// The [hyperbolic sine][wiki] of `x`.
  /// ```
  ///            e^x - e^-x
  /// sinh(x) = ------------
  ///                2
  /// ```
  ///
  /// See also ``asinh(_:)``.
  ///
  /// [wiki]: https://en.wikipedia.org/wiki/Hyperbolic_function
  static func sinh(_ x: Self) -> Self
  
  /// The [hyperbolic tangent][wiki] of `x`.
  /// ```
  ///            sinh(x)
  /// tanh(x) = ---------
  ///            cosh(x)
  /// ```
  ///
  /// See also ``atanh(_:)``.
  ///
  /// [wiki]: https://en.wikipedia.org/wiki/Hyperbolic_function
  static func tanh(_ x: Self) -> Self
  
  /// The [cosine][wiki] of `x`.
  ///
  /// For real types, `x` may be interpreted as an angle measured in radians.
  ///
  /// See also ``acos(_:)``.
  ///
  /// [wiki]: https://en.wikipedia.org/wiki/Cosine
  static func cos(_ x: Self) -> Self
  
  
  /// The [sine][wiki] of `x`.
  ///
  /// For real types, `x` may be interpreted as an angle measured in radians.
  ///
  /// See also ``asin(_:)``.
  ///
  /// [wiki]: https://en.wikipedia.org/wiki/Sine
  static func sin(_ x: Self) -> Self
  
  /// The [tangent][wiki] of `x`.
  ///
  /// For real types, `x` may be interpreted as an angle measured in radians.
  ///
  /// See also ``atan(_:)``.
  ///
  /// [wiki]: https://en.wikipedia.org/wiki/Tangent
  static func tan(_ x: Self) -> Self
  
  /// The [natural logarithm][wiki] of `x`.
  ///
  /// For types that conform to ``RealFunctions`` see also
  /// ``RealFunctions/log2(_:)`` and ``RealFunctions/log10(_:)``.
  ///
  /// [wiki]: https://en.wikipedia.org/wiki/Logarithm
  static func log(_ x: Self) -> Self
  
  /// log(1 + x), computed in such a way as to maintain accuracy for small x.
  ///
  /// For types that conform to ``RealFunctions`` see also
  /// ``RealFunctions/log2(_:)`` and ``RealFunctions/log10(_:)``.
  static func log(onePlus x: Self) -> Self
  
  /// The [inverse hyperbolic cosine][wiki] of `x`.
  ///
  /// ```
  /// cosh(acosh(x)) ≅ x
  /// ```
  ///
  /// See also ``cosh(_:)``.
  ///
  /// [wiki]: https://en.wikipedia.org/wiki/Inverse_hyperbolic_function
  static func acosh(_ x: Self) -> Self
  
  /// The [inverse hyperbolic sine][wiki] of `x`.
  ///
  /// ```
  /// sinh(asinh(x)) ≅ x
  /// ```
  ///
  /// See also ``sinh(_:)``.
  ///
  /// [wiki]: https://en.wikipedia.org/wiki/Inverse_hyperbolic_function
  static func asinh(_ x: Self) -> Self
  
  /// The [inverse hyperbolic tangent][wiki] of `x`.
  ///
  /// ```
  /// tanh(atanh(x)) ≅ x
  /// ```
  ///
  /// See also ``tanh(_:)``.
  ///
  /// [wiki]: https://en.wikipedia.org/wiki/Inverse_hyperbolic_function
  static func atanh(_ x: Self) -> Self
  
  /// The [arccosine][wiki] (inverse cosine) of `x`.
  ///
  /// For real types, the result may be interpreted as an angle measured in
  /// radians.
  ///
  /// ```
  /// cos(acos(x)) ≅ x
  /// ```
  ///
  /// See also ``cos(_:)``.
  ///
  /// [wiki]: https://en.wikipedia.org/wiki/Inverse_trigonometric_functions
  static func acos(_ x: Self) -> Self
  
  /// The [arcsine][wiki]  (inverse sine) of `x`.
  ///
  /// For real types, the result may be interpreted as an angle measured in
  /// radians.
  ///
  /// ```
  /// sin(asin(x)) ≅ x
  /// ```
  ///
  /// See also ``sin(_:)``.
  ///
  /// [wiki]: https://en.wikipedia.org/wiki/Inverse_trigonometric_functions
  static func asin(_ x: Self) -> Self
  
  /// The [arctangent][wiki]  (inverse tangent) of `x`.
  ///
  /// For real types, the result may be interpreted as an angle measured in
  /// radians.
  ///
  /// ```
  /// tan(atan(x)) ≅ x
  /// ```
  ///
  /// See also ``tan(_:)``.
  /// For types that conform to ``RealFunctions``, you will sometimes want
  /// to use ``RealFunctions/atan2(y:x:)`` instead.
  ///
  /// [wiki]: https://en.wikipedia.org/wiki/Inverse_trigonometric_functions
  static func atan(_ x: Self) -> Self
  
  /// exp(y * log(x)) computed with additional internal precision.
  ///
  /// The edge-cases of this function are defined based on the behavior of the
  /// expression `exp(y log x)`, matching IEEE 754's `powr` operation.
  /// In particular, this means that if `x` and `y` are both zero, `pow(x,y)`
  /// is `nan` for real types and `infinity` for complex types, rather than 1.
  ///
  /// There is also
  /// <doc:/documentation/RealModule/ElementaryFunctions/pow(_:_:)-9imp6>,
  /// whose behavior is defined in terms of repeated multiplication.
  static func pow(_ x: Self, _ y: Self) -> Self
  
  /// `x` raised to the nth power.
  ///
  /// The edge-cases of this function are defined in terms of repeated
  /// multiplication or division, rather than exp(n log x). In particular,
  /// `Float.pow(0, 0)` is 1.
  static func pow(_ x: Self, _ n: Int) -> Self
  
  /// The [square root][wiki] of `x`.
  ///
  /// [wiki]: https://en.wikipedia.org/wiki/Square_root
  static func sqrt(_ x: Self) -> Self
  
  /// The nth root of `x`.
  static func root(_ x: Self, _ n: Int) -> Self
}


// ===== Sources/RealModule/Float+Real.swift =====
//===--- Float+Real.swift -------------------------------------*- swift -*-===//
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

extension Float: Real {
  @_transparent
  public static func cos(_ x: Float) -> Float {
    libm_cosf(x)
  }
  
  @_transparent
  public static func sin(_ x: Float) -> Float {
    libm_sinf(x)
  }
  
  @_transparent
  public static func tan(_ x: Float) -> Float {
    libm_tanf(x)
  }
  
  @_transparent
  public static func acos(_ x: Float) -> Float {
    libm_acosf(x)
  }
  
  @_transparent
  public static func asin(_ x: Float) -> Float {
    libm_asinf(x)
  }
  
  @_transparent
  public static func atan(_ x: Float) -> Float {
    libm_atanf(x)
  }
  
  @_transparent
  public static func cosh(_ x: Float) -> Float {
    libm_coshf(x)
  }
  
  @_transparent
  public static func sinh(_ x: Float) -> Float {
    libm_sinhf(x)
  }
  
  @_transparent
  public static func tanh(_ x: Float) -> Float {
    libm_tanhf(x)
  }
  
  @_transparent
  public static func acosh(_ x: Float) -> Float {
    libm_acoshf(x)
  }
  
  @_transparent
  public static func asinh(_ x: Float) -> Float {
    libm_asinhf(x)
  }
  
  @_transparent
  public static func atanh(_ x: Float) -> Float {
    libm_atanhf(x)
  }
  
  @_transparent
  public static func exp(_ x: Float) -> Float {
    libm_expf(x)
  }
  
  @_transparent
  public static func expMinusOne(_ x: Float) -> Float {
    libm_expm1f(x)
  }
  
  @_transparent
  public static func log(_ x: Float) -> Float {
    libm_logf(x)
  }
  
  @_transparent
  public static func log(onePlus x: Float) -> Float {
    libm_log1pf(x)
  }
  
  @_transparent
  public static func erf(_ x: Float) -> Float {
    libm_erff(x)
  }
  
  @_transparent
  public static func erfc(_ x: Float) -> Float {
    libm_erfcf(x)
  }
  
  @_transparent
  public static func exp2(_ x: Float) -> Float {
    libm_exp2f(x)
  }
  
  #if os(macOS) || os(iOS) || os(tvOS) || os(watchOS)
  @_transparent
  public static func exp10(_ x: Float) -> Float {
    libm_exp10f(x)
  }
  #endif
  
  @_transparent
  public static func hypot(_ x: Float, _ y: Float) -> Float {
    libm_hypotf(x, y)
  }
  
  @_transparent
  public static func gamma(_ x: Float) -> Float {
    libm_tgammaf(x)
  }
  
  @_transparent
  public static func log2(_ x: Float) -> Float {
    libm_log2f(x)
  }
  
  @_transparent
  public static func log10(_ x: Float) -> Float {
    libm_log10f(x)
  }
  
  @_transparent
  public static func pow(_ x: Float, _ y: Float) -> Float {
    guard x >= 0 else { return .nan }
    if x == 0 && y == 0 { return .nan }
    return libm_powf(x, y)
  }
  
  @_transparent
  public static func pow(_ x: Float, _ n: Int) -> Float {
    // If n is exactly representable as Float, we can just call powf:
    if let y = Float(exactly: n) {
      return libm_powf(x, y)
    }
    // Otherwise, n is too large to losslessly represent as Float.
    // The range of "interesting" n is -1488522191 ... 1744361944; outside
    // of this range, all x != 1 overflow or underflow, so only the parity
    // of x matters. We don't really care about the specific range at all,
    // only that the bounds fit exactly into two Floats.
    //
    // We do, however, need to be careful that high and low both have the
    // same sign as n (consult the Double implementation for details of why
    // this matters), so we need to be a little bit careful constructing
    // them.
    //
    // Unlike the Double implementation, when n is very large, high will
    // get rounded here; that's OK because it does not change the sign or
    // parity, which are the only two bits that matter for such large
    // exponents in Float.
    let mask = Int(truncatingIfNeeded: 0xffffff)
    let round = n < 0 ? mask : 0
    let high = (n &+ round) & ~mask
    let low = n &- high
    return libm_powf(x, Float(low)) * libm_powf(x, Float(high))
  }
  
  @_transparent
  public static func root(_ x: Float, _ n: Int) -> Float {
    guard x >= 0 || n % 2 != 0 else { return .nan }
    // Workaround the issue mentioned below for the specific case of n = 3
    // where we can fallback on cbrt.
    if n == 3 { return libm_cbrtf(x) }
    // TODO: this implementation is not quite correct, because either n or
    // 1/n may be not be representable as Float.
    return Float(signOf: x, magnitudeOf: libm_powf(x.magnitude, 1/Float(n)))
  }
  
  @_transparent
  public static func atan2(y: Float, x: Float) -> Float {
    libm_atan2f(y, x)
  }
  
  #if !os(Windows)
  @_transparent
  public static func logGamma(_ x: Float) -> Float {
    var dontCare: Int32 = 0
    return libm_lgammaf(x, &dontCare)
  }
  #endif
  
  @_transparent
  public static func _relaxedAdd(_ a: Float, _ b: Float) -> Float {
    _numerics_relaxed_addf(a, b)
  }
  
  @_transparent
  public static func _relaxedMul(_ a: Float, _ b: Float) -> Float {
    _numerics_relaxed_mulf(a, b)
  }
}

