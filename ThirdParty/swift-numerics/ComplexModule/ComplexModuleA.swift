// Vendored from apple/swift-numerics 1.1.1; see ../LICENSE.txt

// ===== Sources/ComplexModule/Complex+AdditiveArithmetic.swift =====
//===--- Complex+AdditiveArithmetic.swift ---------------------*- swift -*-===//
//
// This source file is part of the Swift Numerics open source project
//
// Copyright (c) 2019-2025 Apple Inc. and the Swift Numerics project authors
// Licensed under Apache License v2.0 with Runtime Library Exception
//
// See https://swift.org/LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

import RealModule

extension Complex: AdditiveArithmetic {
  /// The additive identity, with real and imaginary parts both zero.
  ///
  /// See also: ``one``, ``i``, ``infinity``
  @_transparent
  public static var zero: Complex {
    Complex(0, 0)
  }
  
  @_transparent
  public static func +(z: Complex, w: Complex) -> Complex {
    return Complex(z.x + w.x, z.y + w.y)
  }
  
  @_transparent
  public static func -(z: Complex, w: Complex) -> Complex {
    return Complex(z.x - w.x, z.y - w.y)
  }
  
  @_transparent
  public static func +=(z: inout Complex, w: Complex) {
    z = z + w
  }
  
  @_transparent
  public static func -=(z: inout Complex, w: Complex) {
    z = z - w
  }
}


// ===== Sources/ComplexModule/Complex+AlgebraicField.swift =====
//===--- Complex+AlgebraicField.swift -------------------------*- swift -*-===//
//
// This source file is part of the Swift Numerics open source project
//
// Copyright (c) 2019-2025 Apple Inc. and the Swift Numerics project authors
// Licensed under Apache License v2.0 with Runtime Library Exception
//
// See https://swift.org/LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

import RealModule

extension Complex: AlgebraicField {
  /// The multiplicative identity `1 + 0i`.
  ///
  /// See also: ``zero``, ``i``, ``infinity``
  @_transparent
  public static var one: Complex {
    Complex(1, 0)
  }
  
  /// The [complex conjugate][conj] of this value.
  ///
  /// [conj]: https://en.wikipedia.org/wiki/Complex_conjugate
  @_transparent
  public var conjugate: Complex {
    Complex(x, -y)
  }
  
  @_transparent
  public static func /=(z: inout Complex, w: Complex) {
    z = z / w
  }
  
  @_transparent
  public static func /(z: Complex, w: Complex) -> Complex {
    // Try the naive expression z/w = z * (conj(w) / |w|^2); if we can
    // compute this without over/underflow, everything is fine and the
    // result is correct. If not, we have to rescale and do the
    // computation carefully (see below).
    let lenSq = w.lengthSquared
    guard lenSq.isNormal else { return rescaledDivide(z, w) }
    return z * w.conjugate.divided(by: lenSq)
  }
  
  @usableFromInline @_alwaysEmitIntoClient @inline(never)
  internal static func rescaledDivide(_ z: Complex, _ w: Complex) -> Complex {
    if w.isZero { return .infinity }
    if !w.isFinite { return .zero }
    //  Scaling algorithm adapted from Doug Priest's "Efficient Scaling for
    //  Complex Division":
    if w.magnitude < .leastNormalMagnitude {
      //  A difference from Priest's algorithm is that he didn't have to worry
      //  about types like Float16, where the significand width is comparable
      //  to the exponent range, such that |leastNormalMagnitude|^(-¾) isn't
      //  representable (e.g. for Float16 it would want to be 2¹⁸, but the
      //  largest allowed exponent is 15). Note that it's critical to use zʹ/wʹ
      //  after rescaling to avoid this, rather than falling through into the
      //  normal rescaling, because otherwise we might end up back in the
      //  situation where |w| ~ 1.
      let s = 1/(RealType(RealType.radix) * .leastNormalMagnitude)
      let wʹ = w.multiplied(by: s)
      let zʹ = z.multiplied(by: s)
      return zʹ / wʹ
    }
    //  Having handled that case, we proceed pretty similarly to Priest:
    //
    //  1. Choose real scale s ~ |w|^(-¾), an exact power of the radix.
    //  2. wʹ ← sw
    //  3. zʹ ← sz
    //  4. return zʹ * (wʹ.conjugate / wʹ.lengthSquared) (i.e. zʹ/wʹ).
    //
    //  Why is this safe and accurate? First, observe that wʹ and zʹ are both
    //  computed exactly because:
    //
    //  - s is an exact power of radix.
    //  - wʹ ~ |w|^(¼), and hence cannot overflow or underflow.
    //  - zʹ might overflow or underflow, but only if the final result also
    //       overflows or underflows. (This is more subtle than I make it
    //       sound. In particular, most of the fast ways one might try to
    //       compute s give rise to a situation where when |w| is close to
    //       one, multiplication by s is a dilation even though the actual
    //       division is a contraction or vice-versa, and thus intermediate
    //       computations might incorrectly overflow or underflow. Priest
    //       had to take some care to avoid this situation, but we do not,
    //       because we have already ruled out |w| ~ 1 before we call this
    //       function.)
    //
    //  Next observe that |wʹ.lengthSquared| ~ |w|^(½), so again this cannot
    //  overflow or underflow, and neither can (wʹ.conjugate/wʹ.lengthSquared),
    //  which has magnitude like |w|^(-¼).
    //
    //  Note that because the scale factor is always a power of the radix,
    //  the rescaling does not affect rounding, and so this algorithm is scale-
    //  invariant compared to the mainline `/` implementation, up to the
    //  underflow boundary.
    //
    //  Note that our final assembly of the result is different from Priest;
    //  he applies s to w twice, instead of once to w and once to z, and
    //  does the product as (zw̅ʺ)*(1/|wʹ|²), while we do zʹ(w̅ʹ/|wʹ|²). We
    //  prefer our version for three reasons:
    //
    //  1. it extracts a little more ILP
    //  2. it makes it so that we get exactly the same roundings on the
    //     rescaled divide path as on the fast path, so that z/w = tz/tw
    //     when tz and tw are computed exactly.
    //  3. it unlocks a future optimization where we hoist s and
    //     (w̅ʹ/|wʹ|²) and make divisions all fast-path without perturbing
    //     rounding.
    let s = RealType(
      sign: .plus,
      exponent: -3*w.magnitude.exponent/4,
      significand: 1
    )
    let wʹ = w.multiplied(by: s)
    let zʹ = z.multiplied(by: s)
    return zʹ * wʹ.conjugate.divided(by: wʹ.lengthSquared)
  }
  
  /// A normalized complex number with the same phase as this value.
  ///
  /// If such a value cannot be produced (because the phase of zero and
  /// infinity is undefined), `nil` is returned.
  @inlinable
  public var normalized: Complex? {
    if length.isNormal {
      return self.divided(by: length)
    }
    if isZero || !isFinite {
      return nil
    }
    return self.divided(by: magnitude).normalized
  }
  
  /// The reciprocal of this value, if it can be computed without undue
  /// overflow or underflow.
  ///
  /// If z.reciprocal is non-nil, you can safely replace division by z with
  /// multiplication by this value. It is not advantageous to do this for an
  /// isolated division, but if you are dividing many values by a single
  /// denominator, this will often be a significant performance win.
  ///
  /// A typical use case looks something like this:
  /// ```
  /// func divide<T: Real>(data: [Complex<T>], by divisor: Complex<T>) -> [Complex<T>] {
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
  /// Unlike real types, when working with complex types, multiplying by the
  /// reciprocal instead of dividing cannot change the result. If the
  /// reciprocal is non-nil, the two computations are always equivalent.
  @inlinable
  public var reciprocal: Complex? {
    let recip = 1/self
    if recip.isNormal || isZero || !isFinite {
      return recip
    }
    return nil
  }
  
  @_transparent
  public static func _relaxedAdd(_ a: Self, _ b: Self) -> Self {
    Complex(Relaxed.sum(a.x, b.x), Relaxed.sum(a.y, b.y))
  }
  
  @_transparent
  public static func _relaxedMul(_ a: Self, _ b: Self) -> Self {
    Complex(
      Relaxed.sum(Relaxed.product(a.x, b.x), -Relaxed.product(a.y, b.y)),
      Relaxed.sum(Relaxed.product(a.x, b.y),  Relaxed.product(a.y, b.x))
    )
  }
}


// ===== Sources/ComplexModule/Complex+Codable.swift =====
//===--- Complex+Codable.swift --------------------------------*- swift -*-===//
//
// This source file is part of the Swift Numerics open source project
//
// Copyright (c) 2019-2025 Apple Inc. and the Swift Numerics project authors
// Licensed under Apache License v2.0 with Runtime Library Exception
//
// See https://swift.org/LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

import RealModule

// FloatingPoint does not refine Codable, so this is a conditional conformance.
#if compiler(>=6.0)
@_unavailableInEmbedded
#endif
extension Complex: Decodable where RealType: Decodable {
  public init(from decoder: Decoder) throws {
    var unkeyedContainer = try decoder.unkeyedContainer()
    let x = try unkeyedContainer.decode(RealType.self)
    let y = try unkeyedContainer.decode(RealType.self)
    self.init(x, y)
  }
}

#if compiler(>=6.0)
@_unavailableInEmbedded
#endif
extension Complex: Encodable where RealType: Encodable {
  public func encode(to encoder: Encoder) throws {
    var unkeyedContainer = encoder.unkeyedContainer()
    try unkeyedContainer.encode(x)
    try unkeyedContainer.encode(y)
  }
}


// ===== Sources/ComplexModule/Complex+ElementaryFunctions.swift =====
//===--- Complex+ElementaryFunctions.swift --------------------*- swift -*-===//
//
// This source file is part of the Swift Numerics open source project
//
// Copyright (c) 2019-2025 Apple Inc. and the Swift Numerics project authors
// Licensed under Apache License v2.0 with Runtime Library Exception
//
// See https://swift.org/LICENSE.txt for license information
// See https://swift.org/CONTRIBUTORS.txt for the list of Swift project authors
//
//===----------------------------------------------------------------------===//

// Implementation goals, in order of priority:
//
// 1. Get the branch cuts right. We should match Kahan's /Much Ado/
//    paper for finite values.
// 2. Have good relative accuracy in the complex norm.
// 3. Preserve sign symmetries where possible.
// 4. Handle the point at infinity consistently with basic operations.
//    This means diverging from what C and C++ do for non-finite inputs.
// 5. Get good componentwise accuracy /when possible/. I don't know how
//    to do that for all of these functions off the top of my head, and
//    I don't think that other libraries have tried to do so in general,
//    so this is a research project. We should not sacrifice 1-4 for it.
//    Note that multiplication and division don't even provide good
//    componentwise relative accuracy, so it's _totally OK_ to not get
//    it for these functions too. But: it's a dynamite long-term research
//    project.
// 6. Give the best performance we can. We should care about performance,
//    but with lower precedence than the other considerations.
//
// Except where derivations are given, the expressions used here are all
// adapted from Kahan's 1986 paper "Branch Cuts for Complex Elementary
// Functions; or: Much Ado About Nothing's Sign Bit".

import RealModule

extension Complex: ElementaryFunctions {
  
  // MARK: - exp-like functions
  
  /// The complex exponential function e^z whose base `e` is the base of the
  /// natural logarithm.
  ///
  /// Mathematically, this operation can be expanded in terms of the `Real`
  /// operations `exp`, `cos` and `sin` as follows:
  /// ```
  /// exp(x + iy) = exp(x) exp(iy)
  ///             = exp(x) cos(y) + i exp(x) sin(y)
  /// ```
  /// Note that naive evaluation of this expression in floating-point would be
  /// prone to premature overflow, since `cos` and `sin` both have magnitude
  /// less than 1 for most inputs (i.e. `exp(x)` may be infinity when
  /// `exp(x) cos(y)` would not be).
  @inlinable
  public static func exp(_ z: Complex) -> Complex {
    guard z.isFinite else { return z }
    // If x < log(greatestFiniteMagnitude), then exp(x) does not overflow.
    // To protect ourselves against sketchy log or exp implementations in
    // an unknown host library, or slight rounding disagreements between
    // the two, subtract one from the bound for a little safety margin.
    guard z.x < RealType.log(.greatestFiniteMagnitude) - 1 else {
      let halfScale = RealType.exp(z.x/2)
      let phase = Complex(RealType.cos(z.y), RealType.sin(z.y))
      return phase.multiplied(by: halfScale).multiplied(by: halfScale)
    }
    return Complex(.cos(z.y), .sin(z.y)).multiplied(by: .exp(z.x))
  }
  
  @inlinable
  public static func expMinusOne(_ z: Complex) -> Complex {
    // exp(x + iy) - 1 = (exp(x) cos(y) - 1) + i exp(x) sin(y)
    //                   -------- u --------
    // Note that the imaginary part is just the usual exp(x) sin(y);
    // the only trick is computing the real part ("u"):
    //
    // u = exp(x) cos(y) - 1
    //   = exp(x) cos(y) - cos(y) + cos(y) - 1
    //   = (exp(x) - 1) cos(y) + (cos(y) - 1)
    //   = expMinusOne(x) cos(y) + cosMinusOne(y)
    //
    // Note: most implementations of expm1 for complex (e.g. Julia's)
    // factor the real part as follows instead:
    //
    //     exp(x) cosMinuxOne(y) + expMinusOne(x)
    //
    // The other implementation that is sometimes seen is:
    //
    //     expMinusOne(z) = 2*exp(z/2)*sinh(z/2)
    //
    // All three of these implementations provide good relative error
    // bounds _in the complex norm_, but the cosineMinusOne-based
    // implementation has the best _componentwise_ error characteristics,
    // which is why we use it here:
    //
    //     Implementation |        Real        |    Imaginary   |
    //     ---------------+--------------------+----------------+
    //          Ours      |    Hybrid bound    | Relative bound |
    //        Standard    |      No bound      | Relative bound |
    //       Half Angle   |    Hybrid bound    |  Hybrid bound  |
    //
    // FUTURE WORK: devise an algorithm that achieves good _relative_ error
    // in the real component as well. Doing this efficiently is a research
    // project--exp(x) cos(y) - 1 can be very nearly zero along a curve in
    // the complex plane, not only at zero. Evaluating it accurately
    // _without_ depending on arbitrary-precision exp and cos is an
    // interesting challenge.
    guard z.isFinite else { return z }
    // If exp(z) is close to the overflow boundary, we don't need to
    // worry about the "MinusOne" part of this function; we're just
    // computing exp(z). (Even when z.y is near a multiple of π/2,
    // it can't be close enough to overcome the scaling from exp(z.x),
    // so the -1 term is _always_ negligable). So we simply handle
    // these cases exactly the same as exp(z).
    guard z.x < RealType.log(.greatestFiniteMagnitude) - 1 else {
      let halfScale = RealType.exp(z.x/2)
      let phase = Complex(RealType.cos(z.y), RealType.sin(z.y))
      return phase.multiplied(by: halfScale).multiplied(by: halfScale)
    }
    // Special cases out of the way, evaluate as discussed above.
    return Complex(
      Relaxed.multiplyAdd(.cos(z.y), .expMinusOne(z.x), .cosMinusOne(z.y)),
      .exp(z.x) * .sin(z.y)
    )
  }
  
  // cosh(x + iy) = cosh(x) cos(y) + i sinh(x) sin(y).
  //
  // Like exp, cosh is entire, so we do not need to worry about where
  // branch cuts fall. Also like exp, cancellation never occurs in the
  // evaluation of the naive expression, so all we need to be careful
  // about is the behavior near the overflow boundary.
  //
  // Fortunately, if |x| >= -log(ulpOfOne), cosh(x) and sinh(x) are
  // both just exp(|x|)/2, and we already know how to compute that.
  //
  // This function and sinh should stay in sync; if you make a
  // modification here, you should almost surely make a parallel
  // modification to sinh below.
  @inlinable
  public static func cosh(_ z: Complex) -> Complex {
    guard z.isFinite else { return z }
    guard z.x.magnitude < -RealType.log(.ulpOfOne) else {
      let phase = Complex(RealType.cos(z.y), RealType.sin(z.y))
      let firstScale = RealType.exp(z.x.magnitude/2)
      let secondScale = firstScale/2
      return phase.multiplied(by: firstScale).multiplied(by: secondScale)
    }
    // Future optimization opportunity: expm1 is faster than cosh/sinh
    // on most platforms, and division is now commonly pipelined, so we
    // might replace the check above with a much more conservative one,
    // and then evaluate cosh(x) and sinh(x) as
    //
    // cosh(x) = 1 + 0.5*expm1(x)*expm1(x) / (1 + expm1(x))
    // sinh(x) = expm1(x) + 0.5*expm1(x) / (1 + expm1(x))
    //
    // This won't be a _big_ win except on platforms with a crappy sinh
    // and cosh, and for those we should probably just provide our own
    // implementations of _those_, so for now let's keep it simple and
    // obviously correct.
    return Complex(
      RealType.cosh(z.x) * RealType.cos(z.y),
      RealType.sinh(z.x) * RealType.sin(z.y)
    )
  }
  
  // sinh(x + iy) = sinh(x) cos(y) + i cosh(x) sin(y)
  //
  // See cosh above for algorithm details.
  @inlinable
  public static func sinh(_ z: Complex) -> Complex {
    guard z.isFinite else { return z }
    guard z.x.magnitude < -RealType.log(.ulpOfOne) else {
      let phase = Complex(RealType.cos(z.y), RealType.sin(z.y))
      let firstScale = RealType.exp(z.x.magnitude/2)
      let secondScale = RealType(signOf: z.x, magnitudeOf: firstScale/2)
      return phase.multiplied(by: firstScale).multiplied(by: secondScale)
    }
    return Complex(
      RealType.sinh(z.x) * RealType.cos(z.y),
      RealType.cosh(z.x) * RealType.sin(z.y)
    )
  }
  
  // tanh(z) = sinh(z) / cosh(z)
  @inlinable
  public static func tanh(_ z: Complex) -> Complex {
    guard z.isFinite else { return z }
    // Note that when |x| is larger than -log(.ulpOfOne),
    // sinh(x + iy) == ±cosh(x + iy), so tanh(x + iy) is just ±1.
    guard z.x.magnitude < -RealType.log(.ulpOfOne) else {
      return Complex(
        RealType(signOf: z.x, magnitudeOf: 1),
        RealType(signOf: z.y, magnitudeOf: 0)
      )
    }
    // Now we have z in a vertical strip where exp(x) is reasonable,
    // and y is finite, so we can simply evaluate sinh(z) and cosh(z).
    //
    // TODO: Kahan uses a different expression for evaluation here; it
    // isn't strictly necessary for numerics reasons--it's to avoid
    // doing the complex division, but it probably provides better
    // componentwise error bounds, and is likely more efficient (because
    // it avoids the complex division, which is painful even when well-
    // scaled). This suffices to get us up and running.
    return sinh(z) / cosh(z)
  }
  
  // cos(z) = cosh(iz)
  @inlinable
  public static func cos(_ z: Complex) -> Complex {
    return cosh(Complex(-z.y, z.x))
  }
  
  // sin(z) = -i*sinh(iz)
  @inlinable
  public static func sin(_ z: Complex) -> Complex {
    let w = sinh(Complex(-z.y, z.x))
    return Complex(w.y, -w.x)
  }
  
  // tan(z) = -i*tanh(iz)
  @inlinable
  public static func tan(_ z: Complex) -> Complex {
    let w = tanh(Complex(-z.y, z.x))
    return Complex(w.y, -w.x)
  }
  
  // MARK: - log-like functions
  @inlinable
  public static func log(_ z: Complex) -> Complex {
    // If z is zero or infinite, the phase is undefined, so the result is
    // the single exceptional value.
    guard z.isFinite && !z.isZero else { return .infinity }
    // Having eliminated non-finite values and zero, the imaginary part is
    // easy; it's just the phase, which is always computed with good
    // relative accuracy via atan2.
    let θ = z.phase
    // The real part of the result is trickier. In exact arithmetic, the
    // real part is just log |z|--many implementations of complex functions
    // simply use this expression as is. However, there are two problems
    // lurking here:
    //
    //   - length can overflow even when log(z) is finite.
    //
    //   - when length is close to 1, catastrophic cancellation is hidden
    //     in this expression. Consider, e.g. z = 1 + δi for small δ.
    //
    //     Because δ ≪ 1, |z| rounds to 1, and so log |z| produces zero.
    //     We can expand using Taylor series to see that the result should
    //     be:
    //
    //         log |z| = log √(1 + δ²)
    //                 = log(1 + δ²)/2
    //                 = δ²/2 + O(δ⁴)
    //
    //     So naively using log |z| results in a total loss of relative
    //     accuracy for this case. Note that this is _not_ constrained near
    //     a single point; it occurs everywhere close to the circle |z| = 1.
    //
    //     Note that this case still _does_ deliver a result with acceptable
    //     relative accuracy in the complex norm, because
    //
    //         Im(log z) ≈ δ ≫ δ²/2 ≈ Re(log z).
    //
    // There are a number of ways to try to tackle this problem. I'll begin
    // with a simple one that solves the first issue, and _sometimes_ the
    // second, then analyze when it doesn't work for the second case.
    //
    // To handle very large arguments without overflow, the standard
    // approach is to _rescale_ the problem. We can do this by finding
    // whichever of x and y has greater magnitude, and dividing through
    // by it. You can think of this as changing coordinates by reflections
    // so that we get a new value w = u + iv with |w| = |z| (and hence
    // Re(log w) = Re(log z), and 0 ≤ u, 0 ≤ v ≤ u.
    let u = max(z.x.magnitude, z.y.magnitude)
    let v = min(z.x.magnitude, z.y.magnitude)
    // Now expand out log |w|:
    //
    //     log |w| = log(u² + v²)/2
    //             = log u + log(onePlus: (v/u)²)/2
    //
    // This looks promising! It handles overflow well, because log(u) is
    // finite for every finite u, and we have 0 ≤ v/u ≤ 1, so the second
    // term is bounded by 0 ≤ log(1 + (v/u)²)/2 ≤ (log 2)/2. It also
    // handles the example I gave above well: we have u = 1, v = δ, and
    //
    //     log(1) + log(onePlus: δ²)/2 = 0 + δ²/2
    //
    // as expected.
    //
    // Unfortunately, it does not handle all points close to the unit
    // circle so well; it's easy to see why if we look at the two terms
    // that contribute to the result. Cancellation occurs when the result
    // is close to zero and the terms have opposing signs. By construction,
    // the second term is always positive, so the easiest observation is
    // that cancellation is only a problem for u < 1 (because otherwise
    // log u is also positive, and there can be no cancellation).
    //
    // We are not trying for sub-ulp accuracy, just a good relative error
    // bound, so for our purposes it suffices to have log u dominate the
    // result:
    if u >= 1 || u >= Relaxed.multiplyAdd(u, u, v*v) {
      let r = v / u
      return Complex(.log(u) + .log(onePlus: r*r)/2, θ)
    }
    // Here we're in the tricky case; cancellation is likely to occur.
    // Instead of the factorization used above, we will want to evaluate
    // log(onePlus: u² + v² - 1)/2. This all boils down to accurately
    // evaluating u² + v² - 1. To begin, calculate both squared terms
    // as exact head-tail products (u is guaranteed to be well scaled,
    // v may underflow, but if it does it doesn't matter, the u term is
    // all we need).
    let (a,b) = Augmented.product(u, u)
    let (c,d) = Augmented.product(v, v)
    // It would be nice if we could simply use a - 1, but unfortunately
    // we don't have a tight enough bound to guarantee that that expression
    // is exact; a may be as small as 1/4, so we could lose a single bit
    // to rounding if we did that.
    var (s,e) = Augmented.sum(large: -1, small: a)
    // Now we are ready to assemble the result. If cancellation happens,
    // then |c| > |e| > |b|, |d|, so this assembly order is safe. It's
    // also possible that |c| and |d| are small, but if that happens then
    // there is no significant cancellation, and the exact assembly doesn't
    // matter.
    s = (s + c) + e + b + d
    return Complex(.log(onePlus: s)/2, θ)
  }
  
  @inlinable
  public static func log(onePlus z: Complex) -> Complex {
    // If either |x| or |y| is bounded away from the origin, we don't need
    // any extra precision, and can just literally compute log(1+z). Note
    // that this includes part of the circle |1+z| = 1 where log(onePlus:)
    // vanishes (where x <= -0.5), but on this portion of the circle 1+x
    // is always exact by Sterbenz' lemma, so as long as log( ) produces
    // a good result, log(1+z) will too.
    guard 2*z.x.magnitude < 1 && z.y.magnitude < 1 else { return log(1+z) }
    // z is in (±0.5, ±1), so we need to evaluate more carefully.
    // The imaginary part is straightforward:
    let θ = (1+z).phase
    // For the real part, we _could_ use the same approach that we do for
    // log( ), but we'd need an extra-precise (1+x)², which can potentially
    // be quite painful to calculate. Instead, we can use an approach that
    // NevinBR suggested on the Swift forums:
    //
    //     Re(log(1+z)) = (log(1+z) + log(1+z̅)) / 2
    //                  = log((1+z)(1+z̅)) / 2
    //                  = log(1 + z + z̅ + zz̅) / 2
    //                  = log(1 + 2x + x² + y²) / 2
    //                  = log(onePlus: (2+x)x + y²) / 2
    //
    // So now we need to evaluate (2+x)x + y² accurately. To do this,
    // we employ augmented arithmetic; the key observation here is that
    // cancellation is only a problem when y² ≈ -(2+x)x
    let xp2 = Augmented.sum(large: 2, small: z.x) // Known that 2 > |x|.
    let a = Augmented.product(z.x, xp2.head)
    let y² = Augmented.product(z.y, z.y)
    let s = (a.head + y².head + a.tail + y².tail).addingProduct(z.x, xp2.tail)
    return Complex(.log(onePlus: s)/2, θ)
  }
  
  @inlinable
  public static func acos(_ z: Complex) -> Complex {
    Complex(
      2*RealType.atan2(y: sqrt(1-z).real, x: sqrt(1+z).real),
      RealType.asinh((sqrt(1+z).conjugate * sqrt(1-z)).imaginary)
    )
  }
  
  @inlinable
  public static func asin(_ z: Complex) -> Complex {
    Complex(
      RealType.atan2(y: z.x, x: (sqrt(1-z) * sqrt(1+z)).real),
      RealType.asinh((sqrt(1-z).conjugate * sqrt(1+z)).imaginary)
    )
  }
  
  // atan(z) = -i*atanh(iz)
  @inlinable
  public static func atan(_ z: Complex) -> Complex {
    let w = atanh(Complex(-z.y, z.x))
    return Complex(w.y, -w.x)
  }
  
  @inlinable
  public static func acosh(_ z: Complex) -> Complex {
    Complex(
      RealType.asinh((sqrt(z-1).conjugate * sqrt(z+1)).real),
      2*RealType.atan2(y: sqrt(z-1).imaginary, x: sqrt(z+1).real)
    )
  }
  
  // asinh(z) = -i*asin(iz)
  @inlinable
  public static func asinh(_ z: Complex) -> Complex {
    let w = asin(Complex(-z.y, z.x))
    return Complex(w.y, -w.x)
  }
  
  @inlinable
  public static func atanh(_ z: Complex) -> Complex {
    // TODO: Kahan uses a much more complicated expression here; possibly
    // simply because he didn't have a complex log(1+z) with good
    // characteristics. Investigate tradeoffs further.
    //
    // Further TODO: decide policy for point at infinity / NaN. Unlike most
    // of these functions, atanh doesn't have a pole at infinity; convention
    // in C-family languages is use one value in the upper half plane, and
    // another in the lower. Requires some thought about the most appropriate
    // way to handle this case in Swift.
    (log(onePlus: z) - log(onePlus:-z))/2
  }
  
  // MARK: - pow-like functions
  /// `exp(w*log(z))`
  ///
  /// Edge cases for this function are defined according to the defining
  /// expression exp(w log(z)), except that we define pow(0, w) to be 0
  /// instead of infinity when w is in the (strict) right half-plane, so that
  /// we agree with RealType.pow on the positive real line.
  @inlinable
  public static func pow(_ z: Complex, _ w: Complex) -> Complex {
    if z.isZero { return w.real > 0 ? zero : infinity }
    return exp(w * log(z))
  }
  
  @inlinable
  public static func pow(_ z: Complex, _ n: Int) -> Complex {
    if z.isZero { return n < 0 ? infinity : n == 0 ? one : zero }
    // TODO: this implementation is not quite correct, because n may be
    // rounded in conversion to RealType. This only effects very extreme
    // cases, so we'll leave it alone for now.
    //
    // Note that this does not have the same problems that a similar
    // implementation for a real type would have, because there's no
    // parity/sign interaction in the complex plane.
    return exp(log(z).multiplied(by: RealType(n)))
  }
  
  @inlinable
  public static func sqrt(_ z: Complex) -> Complex {
    let lengthSquared = z.lengthSquared
    if lengthSquared.isNormal {
      // If |z|^2 doesn't overflow, then define u and v by:
      //
      //    u = sqrt((|z|+|x|)/2)
      //    v = y/2u
      //
      // If x is positive, the result is just w = (u, v). If x is negative,
      // the result is (|v|, copysign(u, y)) instead.
      let norm = RealType.sqrt(lengthSquared)
      let u = RealType.sqrt((norm + abs(z.x))/2)
      let v: RealType = z.y / (2*u)
      if z.x.sign == .plus {
        return Complex(u, v)
      } else {
        return Complex(abs(v), RealType(signOf: z.y, magnitudeOf: u))
      }
    }
    // Handle edge cases:
    if z.isZero { return Complex(0, z.y) }
    if !z.isFinite { return z }
    // z is finite but badly-scaled. Rescale and replay by factoring out
    // the larger of x and y.
    let scale = RealType.maximum(abs(z.x), abs(z.y))
    return Complex.sqrt(z.divided(by: scale)).multiplied(by: .sqrt(scale))
  }
  
  @inlinable
  public static func root(_ z: Complex, _ n: Int) -> Complex {
    if z.isZero { return .zero }
    // TODO: this implementation is not quite correct, because n may be
    // rounded in conversion to RealType. This only effects very extreme
    // cases, so we'll leave it alone for now.
    //
    // Note that this does not have the same problems that a similar
    // implementation for a real type would have, because there's no
    // parity/sign interaction in the complex plane.
    return exp(log(z).divided(by: RealType(n)))
  }
}


// ===== Sources/ComplexModule/Complex+Hashable.swift =====
//===--- Complex+Hashable.swift -------------------------------*- swift -*-===//
//
// This source file is part of the Swift Numerics open source project
//
// Copyright (c) 2019-2025 Apple Inc. and the Swift Numerics project authors
// Licensed under Apache License v2.0 with Runtime Library Exception
//
// See https://swift.org/LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

import RealModule

extension Complex: Hashable {
  @_transparent
  public static func ==(a: Complex, b: Complex) -> Bool {
    // Identify all numbers with either component non-finite as a single
    // "point at infinity".
    guard a.isFinite || b.isFinite else { return true }
    // For finite numbers, equality is defined componentwise. Cases where
    // only one of a or b is infinite fall through to here as well, but this
    // expression correctly returns false for them so we don't need to handle
    // them explicitly.
    return a.x == b.x && a.y == b.y
  }
  
  @_transparent
  public func hash(into hasher: inout Hasher) {
    // There are two equivalence classes to which we owe special attention:
    // All zeros should hash to the same value, regardless of sign, and all
    // non-finite numbers should hash to the same value, regardless of
    // representation. The correct behavior for zero falls out for free from
    // the hash behavior of floating-point, but we need to use a
    // representative member for any non-finite values.
    if isFinite {
      hasher.combine(x)
      hasher.combine(y)
    } else {
      hasher.combine(RealType.infinity)
    }
  }
}


// ===== Sources/ComplexModule/Complex+IntegerLiteral.swift =====
//===--- Complex+IntegerLiteral.swift -------------------------*- swift -*-===//
//
// This source file is part of the Swift Numerics open source project
//
// Copyright (c) 2019-2025 Apple Inc. and the Swift Numerics project authors
// Licensed under Apache License v2.0 with Runtime Library Exception
//
// See https://swift.org/LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

extension Complex: ExpressibleByIntegerLiteral {
  public typealias IntegerLiteralType = RealType.IntegerLiteralType
  
  @inlinable
  public init(integerLiteral value: IntegerLiteralType) {
    self.init(RealType(integerLiteral: value))
  }
}

