// Vendored from apple/swift-numerics 1.1.1; see ../LICENSE.txt

// ===== Sources/ComplexModule/Complex+Numeric.swift =====
//===--- Complex+Numeric.swift --------------------------------*- swift -*-===//
//
// This source file is part of the Swift Numerics open source project
//
// Copyright (c) 2019-2025 Apple Inc. and the Swift Numerics project authors
// Licensed under Apache License v2.0 with Runtime Library Exception
//
// See https://swift.org/LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

extension Complex: Numeric {
  
  @_transparent
  public static func *(z: Complex, w: Complex) -> Complex {
    return Complex(z.x*w.x - z.y*w.y, z.x*w.y + z.y*w.x)
  }
  
  @_transparent
  public static func *=(z: inout Complex, w: Complex) {
    z = z * w
  }
  
  /// The complex number with specified real part and zero imaginary part.
  ///
  /// Equivalent to `Complex(RealType(real), 0)`.
  @inlinable
  public init<Other: BinaryInteger>(_ real: Other) {
    self.init(RealType(real), 0)
  }
  
  /// The complex number with specified real part and zero imaginary part,
  /// if it can be constructed without rounding.
  @inlinable
  public init?<Other: BinaryInteger>(exactly real: Other) {
    guard let real = RealType(exactly: real) else { return nil }
    self.init(real, 0)
  }
  
  /// The infinity-norm of the value (a.k.a. "maximum norm" or "Чебышёв
  /// [Chebyshev] norm").
  ///
  /// Equal to `max(abs(real), abs(imaginary))`.
  ///
  /// If you need to work with the Euclidean norm (a.k.a. 2-norm) instead,
  /// use the ``length`` or ``lengthSquared`` properties. If you just need
  /// to know "how big" a number is, use this property.
  ///
  /// **Edge cases:**
  ///
  /// - If `z` is not finite, `z.magnitude` is infinity.
  /// - If `z` is zero, `z.magnitude` is zero.
  /// - Otherwise, `z.magnitude` is finite and non-zero.
  @_transparent
  public var magnitude: RealType {
    guard isFinite else { return .infinity }
    return max(abs(x), abs(y))
  }
}


// ===== Sources/ComplexModule/Complex+StringConvertible.swift =====
//===--- Complex+StringConvertible.swift ----------------------*- swift -*-===//
//
// This source file is part of the Swift Numerics open source project
//
// Copyright (c) 2019-2025 Apple Inc. and the Swift Numerics project authors
// Licensed under Apache License v2.0 with Runtime Library Exception
//
// See https://swift.org/LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

extension Complex: CustomStringConvertible {
  public var description: String {
    guard isFinite else { return "inf" }
    return "(\(x), \(y))"
  }
}

#if compiler(>=6.0)
@_unavailableInEmbedded
#endif
extension Complex: CustomDebugStringConvertible {
  public var debugDescription: String {
    "Complex<\(RealType.self)>(\(String(reflecting: x)), \(String(reflecting: y)))"
  }
}


// ===== Sources/ComplexModule/Complex.swift =====
//===--- Complex.swift ----------------------------------------*- swift -*-===//
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

// A [complex number](https://en.wikipedia.org/wiki/Complex_number).
// See Documentation.docc/Complex.md for more details.
@frozen
public struct Complex<RealType> where RealType: Real {
  //  A note on the `x` and `y` properties
  //
  //  `x` and `y` are the names we use for the raw storage of the real and
  //  imaginary components of our complex number. We also provide public
  //  `.real` and `.imaginary` properties, which wrap this storage and
  //  fixup the semantics for non-finite values.
  
  /// The storage for the real component of the value.
  @usableFromInline @inline(__always)
  internal var x: RealType
  
  /// The storage for the imaginary part of the value.
  @usableFromInline @inline(__always)
  internal var y: RealType
  
  /// A complex number constructed by specifying the real and imaginary parts.
  @_transparent
  public init(_ real: RealType, _ imaginary: RealType) {
    x = real
    y = imaginary
  }
}

extension Complex: Sendable where RealType: Sendable { }

// MARK: - Basic properties
extension Complex {
  /// The real part of this complex value.
  ///
  /// If `z` is not finite, `z.real` is `.nan`.
  public var real: RealType {
    @_transparent
    get { isFinite ? x : .nan }

    @_transparent
    set { x = newValue }
  }
  
  /// The imaginary part of this complex value.
  ///
  /// If `z` is not finite, `z.imaginary` is `.nan`.
  public var imaginary: RealType {
    @_transparent
    get { isFinite ? y : .nan }

    @_transparent
    set { y = newValue }
  }
  
  /// The raw representation of the value.
  ///
  /// Use this when you need the underlying RealType values,
  /// without fixup for NaN or infinity.
  public var rawStorage: (x: RealType, y: RealType) {
    @_transparent
    get { (x, y) }
    @_transparent
    set { (x, y) = newValue }
  }
  
  /// The raw representation of the real part of this value.
  @available(*, deprecated, message: "Use rawStorage")
  @_transparent
  public var _rawX: RealType { x }
  
  /// The raw representation of the imaginary part of this value.
  @available(*, deprecated, message: "Use rawStorage")
  @_transparent
  public var _rawY: RealType { y }
}
  
extension Complex {
  /// The imaginary unit.
  ///
  /// See also ``zero``, ``one`` and ``infinity``.
  @_transparent
  public static var i: Complex {
    Complex(0, 1)
  }
  
  /// The point at infinity.
  ///
  /// See also ``zero``, ``one`` and ``i``.
  @_transparent
  public static var infinity: Complex {
    Complex(.infinity, 0)
  }
  
  /// True if this value is finite.
  ///
  /// A complex value is finite if neither component is an infinity or nan.
  ///
  /// See also ``isNormal``, ``isSubnormal`` and ``isZero``.
  @_transparent
  public var isFinite: Bool {
    x.isFinite && y.isFinite
  }
  
  /// True if this value is normal.
  ///
  /// A complex number is normal if it is finite and *either* the real or
  /// imaginary component is normal. A floating-point number representing
  /// one of the components is normal if its exponent allows a full-precision
  /// representation.
  ///
  /// See also ``isFinite``, ``isSubnormal`` and ``isZero``.
  @_transparent
  public var isNormal: Bool {
    isFinite && (x.isNormal || y.isNormal)
  }
  
  /// True if this value is subnormal.
  ///
  /// A complex number is subnormal if it is finite, not normal, and not zero.
  /// When the result of a computation is subnormal, underflow has occurred and
  /// the result generally does not have full precision.
  ///
  /// See also ``isFinite``, ``isNormal`` and ``isZero``.
  @_transparent
  public var isSubnormal: Bool {
    isFinite && !isNormal && !isZero
  }
  
  /// True if this value is zero.
  ///
  /// A complex number is zero if *both* the real and imaginary components
  /// are zero.
  ///
  /// See also ``isFinite``, ``isNormal`` and ``isSubnormal``.
  @_transparent
  public var isZero: Bool {
    x == 0 && y == 0
  }
  
  /// A "canonical" representation of the value.
  ///
  /// For normal complex numbers with a RealType conforming to
  /// BinaryFloatingPoint (the common case), the result is simply this value
  /// unmodified. For zeros, the result has the representation (+0, +0). For
  /// infinite values, the result has the representation (+inf, +0).
  ///
  /// If the RealType admits non-canonical representations, the x and y
  /// components are canonicalized in the result.
  ///
  /// This is mainly useful for interoperation with other languages, where
  /// you may want to reduce each equivalence class to a single representative
  /// before passing across language boundaries, but it may also be useful
  /// for some serialization tasks. It's also a useful implementation detail
  /// for some primitive operations.
  @_transparent
  public var canonicalized: Self {
    if isZero { return .zero }
    if isFinite { return self.multiplied(by: 1) }
    return .infinity
  }
}

// MARK: - Additional Initializers
extension Complex {
  /// The complex number with specified real part and zero imaginary part.
  ///
  /// Equivalent to `Complex(real, 0)`.
  @inlinable
  public init(_ real: RealType) {
    self.init(real, 0)
  }
  
  /// The complex number with zero real part and specified imaginary part.
  ///
  /// Equivalent to `Complex(0, imaginary)`.
  @inlinable
  public init(imaginary: RealType) {
    self.init(0, imaginary)
  }
}

extension Complex where RealType: BinaryFloatingPoint {
  /// `other` rounded to the nearest representable value of this type.
  @inlinable
  public init<Other: BinaryFloatingPoint>(_ other: Complex<Other>) {
    self.init(RealType(other.x), RealType(other.y))
  }
  
  /// `other`, if it can be represented exactly in this type; otherwise `nil`.
  @inlinable
  public init?<Other: BinaryFloatingPoint>(exactly other: Complex<Other>) {
    guard let x = RealType(exactly: other.x),
          let y = RealType(exactly: other.y) else { return nil }
    self.init(x, y)
  }
}


// ===== Sources/ComplexModule/Polar.swift =====
//===--- Polar.swift ------------------------------------------*- swift -*-===//
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

extension Complex {
  /// The Euclidean norm (a.k.a. 2-norm).
  ///
  /// This property takes care to avoid spurious over- or underflow in
  /// this computation. For example:
  ///
  ///     let x: Float = 3.0e+20
  ///     let x: Float = 4.0e+20
  ///     let naive = sqrt(x*x + y*y) // +Inf
  ///     let careful = Complex(x, y).length // 5.0e+20
  ///
  /// Note that it *is* still possible for this property to overflow,
  /// because the length can be as much as sqrt(2) times larger than
  /// either component, and thus may not be representable in the real type.
  ///
  /// For most use cases, you can use the cheaper ``magnitude``
  /// property (which computes the ∞-norm) instead, which always produces
  /// a representable result. See <doc:Magnitude> for more details.
  ///
  /// Edge cases:
  /// - If a complex value is not finite, its `length` is `infinity`.
  ///
  /// See also ``lengthSquared``, ``phase``, ``polar``
  /// and ``init(length:phase:)``.
  @_transparent
  public var length: RealType {
    let naive = lengthSquared
    guard naive.isNormal else { return carefulLength }
    return .sqrt(naive)
  }
  
  //  Internal implementation detail of ``length``, moving slow path off
  //  of the inline function. Note that even `carefulLength` can overflow
  //  for finite inputs, but only when the result is outside the range
  //  of representable values.
  @usableFromInline
  internal var carefulLength: RealType {
    guard isFinite else { return .infinity }
    return .hypot(x, y)
  }
  
  /// The squared length `(real*real + imaginary*imaginary)`.
  ///
  /// This property is more efficient to compute than ``length``, but is
  /// highly prone to overflow or underflow; for finite values that are
  /// not well-scaled, `lengthSquared` is often either zero or
  /// infinity, even when `length` is a finite number. Use this property
  /// only when you are certain that this value is well-scaled.
  ///
  /// For many cases, ``magnitude`` can be used instead, which is similarly
  /// cheap to compute and always returns a representable value.
  ///
  /// Note that because of how `lengthSquared` is used, it is a primary
  /// design goal that it be as fast as possible. Therefore, it does not
  /// normalize infinities, and may return either `.infinity` or `.nan`
  /// for non-finite values.
  @_transparent
  public var lengthSquared: RealType {
    x*x + y*y
  }
  
  /// The phase (angle, or "argument").
  ///
  /// - Returns: The angle (measured above the real axis) in radians. If
  /// the complex value is zero or infinity, the phase is not defined,
  /// and `nan` is returned.
  ///
  /// See also ``length``, ``polar`` and ``init(length:phase:)``.
  @inlinable
  public var phase: RealType {
    guard isFinite && !isZero else { return .nan }
    return .atan2(y: y, x: x)
  }
  
  /// The length and phase (or polar coordinates) of this value.
  ///
  /// Edge cases:
  /// - If the complex value is zero or non-finite, phase is `.nan`.
  /// - If the complex value is non-finite, length is `.infinity`.
  ///
  /// See also: ``length``, ``phase`` and ``init(length:phase:)``.
  public var polar: (length: RealType, phase: RealType) {
    (length, phase)
  }
  
  /// Creates a complex value specified with polar coordinates.
  ///
  /// Edge cases:
  /// - Negative lengths are interpreted as reflecting the point through the
  ///   origin, i.e.:
  ///   ```
  ///   Complex(length: -r, phase: θ) == -Complex(length: r, phase: θ)
  ///   ```
  /// - For any `θ`, even `.infinity` or `.nan`:
  ///   ```
  ///   Complex(length: .zero, phase: θ) == .zero
  ///   ```
  /// - For any `θ`, even `.infinity` or `.nan`, if `r` is infinite then:
  ///   ```
  ///   Complex(length: r, phase: θ) == .infinity
  ///   ```
  /// - Otherwise, `θ` must be finite, or a precondition failure occurs.
  ///
  /// See also ``length``, ``phase`` and ``polar``.
  @inlinable
  public init(length: RealType, phase: RealType) {
    if phase.isFinite {
      self = Complex(.cos(phase), .sin(phase)).multiplied(by: length)
    } else {
      precondition(
        length.isZero || length.isInfinite,
        "Either phase must be finite, or length must be zero or infinite."
      )
      self = Complex(length)
    }
  }
}


// ===== Sources/ComplexModule/Scale.swift =====
//===--- Scale.swift ------------------------------------------*- swift -*-===//
//
// This source file is part of the Swift Numerics open source project
//
// Copyright (c) 2019-2025 Apple Inc. and the Swift Numerics project authors
// Licensed under Apache License v2.0 with Runtime Library Exception
//
// See https://swift.org/LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

// Policy: deliberately not using the * and / operators for these at the
// moment, because then there's an ambiguity in expressions like 2*z; is
// that Complex(2) * z or is it RealType(2) * z? This is especially
// problematic in type inference: suppose we have:
//
//   let a: RealType = 1
//   let b = 2*a
//
// what is the type of b? If we don't have a type context, it's ambiguous.
// If we have a Complex type context, then b will be inferred to have type
// Complex! Obviously, that doesn't help anyone.

extension Complex {
  /// The result of multiplying this value by the real number `a`.
  ///
  /// Equivalent to `self * Complex(a)`, but may be computed more efficiently.
  @inlinable @inline(__always)
  public func multiplied(by a: RealType) -> Complex {
    Complex(x*a, y*a)
  }
  
  /// The result of dividing this value by the real number `a`.
  ///
  /// More efficient than `self / Complex(a)`. May not produce exactly the
  /// same result, but will always be more accurate if they differ.
  @inlinable @inline(__always)
  public func divided(by a: RealType) -> Complex {
    Complex(x/a, y/a)
  }
}

